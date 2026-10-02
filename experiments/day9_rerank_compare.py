"""Day 9 重排对比：三种检索器各取 20 条候选，加不加 Cross-Encoder 重排，金标召回和耗时各是多少。

用法：
  python -m experiments.day9_rerank_compare --data data/hotpotqa/v1 --split validation \
      --bm25-index indexes/hotpot_pool_v1_bm25 --dense-index indexes/hotpot_pool_v1_e5 \
      --agent-run outputs/runs/<V1 Agent 贪心运行>
  2Wiki 同理（--data data/2wiki/v1 --split analysis，不给 --agent-run）
离线分析，金标只由本脚本（评测侧）读取。进程内加载检索器和重排模型，耗时是模型本身的耗时。

重排只能在候选池里调顺序：候选池（前 20 条）里没有的金标，重排救不回来。所以同时报"候选池召回@20"，
它就是重排的上限；"补上的缺口" = (重排后@3 − 重排前@3) / (候选池@20 − 重排前@3)。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from experiments.compare_runs import paired_bootstrap
from retrieval.bm25 import BM25SearchTool
from retrieval.dense import DenseSearchTool
from retrieval.hybrid import rrf_fuse
from retrieval.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]
BASES = ("bm25", "dense", "hybrid")
CONFIGS = BASES + tuple(f"{b}+rerank" for b in BASES)
KS = (1, 3, 5)
POOL = 20


def _pct(xs: list[float]) -> dict:
    return {"mean": round(float(np.mean(xs)), 2), "p50": round(float(np.percentile(xs, 50)), 2),
            "p95": round(float(np.percentile(xs, 95)), 2)}


class Ranker:
    def __init__(self, bm25, dense, reranker):
        self.bm25, self.dense, self.reranker = bm25, dense, reranker
        self.rerank_ms: list[float] = []

    def lists(self, query: str) -> dict[str, list[str]]:
        """6 种配置各自的 doc_id 排序（长度 ≤ 20）。三个候选池分别重排，每次 20 条，耗时就是真实的一次重排成本。"""
        pools = {"bm25": self.bm25.search(query, POOL), "dense": self.dense.search(query, POOL)}
        pools["hybrid"] = rrf_fuse([pools["bm25"], pools["dense"]], POOL)
        out = {b: [d.doc_id for d in docs] for b, docs in pools.items()}
        for b, docs in pools.items():
            if not docs:
                out[f"{b}+rerank"] = []
                continue
            t0 = time.perf_counter()
            scores = self.reranker.score(query, docs)
            self.rerank_ms.append((time.perf_counter() - t0) * 1000)
            out[f"{b}+rerank"] = [d.doc_id for _, d in sorted(zip(scores, docs), key=lambda x: (-x[0], x[1].doc_id))]
        return out


def _recall(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--bm25-index", required=True)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--reranker", default="/root/autodl-tmp/hf_models/bge-reranker-base")
    ap.add_argument("--rerank-device", default="cuda")
    ap.add_argument("--agent-run", help="V1 Agent 运行目录；给了才做 Agent 查询重放")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("重排对比只在 validation / analysis / debug 上做")

    data = ROOT / args.data
    questions = [json.loads(l) for l in open(data / "questions" / f"{args.split}.jsonl", encoding="utf-8")]
    labels = {l["qid"]: l for l in map(json.loads, open(data / "labels" / f"{args.split}.jsonl", encoding="utf-8"))}
    bm25 = BM25SearchTool(ROOT / args.bm25_index)
    dense = DenseSearchTool(ROOT / args.dense_index, device="cpu")
    reranker = CrossEncoderReranker(args.reranker, device=args.rerank_device)
    ranker = Ranker(bm25, dense, reranker)
    ranker.lists("warmup query")
    ranker.rerank_ms.clear()

    rows = []
    pair_tokens = []
    for q in questions:
        lab = labels[q["qid"]]
        lists = ranker.lists(q["question"])
        rows.append({"qid": q["qid"], "type": lab["type"], "num_gold": len(lab["gold_doc_ids"]),
                     "gold_rank": {c: {g: (lists[c].index(g) + 1 if g in lists[c] else None)
                                       for g in lab["gold_doc_ids"]} for c in CONFIGS}})
    orig_ms = list(ranker.rerank_ms)  # 原问题上每次重排（20 条候选）的耗时

    # 输入长度（查询 + 段落的 token 数）：抽前 100 题的 dense 候选池统计就够
    for q in questions[:100]:
        for d in dense.search(q["question"], POOL):
            pair_tokens.append(len(reranker.tokenizer(q["question"], f"{d.title}\n{d.text}", truncation="only_second",
                                                      max_length=reranker.max_length)["input_ids"]))

    replay_top_k = None
    if args.agent_run:
        run = ROOT / args.agent_run
        replay_top_k = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["budget"]["top_k"]
        by_qid = {r["qid"]: r for r in rows}
        for traj in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8")):
            gold = set(labels[traj["qid"]]["gold_doc_ids"])
            seen = {c: set() for c in CONFIGS}
            for s in traj["steps"]:
                if s["action"] and s["action"]["name"] == "search" and s["observation"] and s["observation"]["ok"]:
                    lists = ranker.lists(s["action"]["arguments"]["query"])
                    for c in CONFIGS:
                        seen[c] |= set(lists[c][:replay_top_k])
            by_qid[traj["qid"]]["agent_replay_recall"] = {c: len(seen[c] & gold) / len(gold) for c in CONFIGS}

    def rec(r, c, k):
        return sum(x is not None and x <= k for x in r["gold_rank"][c].values()) / r["num_gold"]

    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["type"]].append(r)
    metrics: dict = {"num_questions": len(rows), "pool_size": POOL, "by_type": {}, "diffs": {}}
    for g, rs in groups.items():
        out = {"n": len(rs)}
        for c in CONFIGS:
            out[c] = {f"recall@{k}": round(float(np.mean([rec(r, c, k) for r in rs])), 4) for k in KS}
            out[c]["all_gold@3"] = round(float(np.mean([rec(r, c, 3) == 1 for r in rs])), 4)
            if "agent_replay_recall" in rs[0]:
                out[c]["agent_replay_recall"] = round(float(np.mean([r["agent_replay_recall"][c] for r in rs])), 4)
        for b in BASES:
            pool = float(np.mean([rec(r, b, POOL) for r in rs]))
            before, after = out[b]["recall@3"], out[f"{b}+rerank"]["recall@3"]
            out[b]["pool_recall@20"] = round(pool, 4)
            out[b]["no_gold_in_pool"] = round(float(np.mean([rec(r, b, POOL) == 0 for r in rs])), 4)
            out[b]["rerank_gap_closed@3"] = round((after - before) / (pool - before), 4) if pool > before else None
        metrics["by_type"][g] = out

    pairs = [(f"{b}+rerank", b) for b in BASES] + [("dense+rerank", "hybrid+rerank"), ("dense+rerank", "dense")]
    for k in (3, 5):
        for a, b in pairs:
            metrics["diffs"][f"{a} - {b} @{k}"] = [round(x, 4) for x in paired_bootstrap([rec(r, a, k) - rec(r, b, k) for r in rows])]
    if "agent_replay_recall" in rows[0]:
        for a, b in pairs:
            metrics["diffs"][f"{a} - {b} agent_replay"] = [round(x, 4) for x in paired_bootstrap(
                [r["agent_replay_recall"][a] - r["agent_replay_recall"][b] for r in rows])]
        metrics["agent_replay_top_k"] = replay_top_k
    metrics["rerank_latency_ms_per_call"] = _pct(orig_ms)
    metrics["rerank_pair_tokens"] = _pct(pair_tokens)
    metrics["reranker"] = reranker.meta

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    name = Path(args.data).parts[-2] if Path(args.data).name.startswith("v") else Path(args.data).name
    out = ROOT / "outputs/runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-day9-rerank-{name}-{args.split}"
                                   + ("-dirty" if dirty else ""))
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"args": vars(args), "pool": POOL, "ks": list(KS),
                                                     "indexes": {"bm25": bm25.meta, "dense": dense.meta},
                                                     "reranker": reranker.meta}, allow_unicode=True), encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + ("-dirty" if dirty else "") + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
