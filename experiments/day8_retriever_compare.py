"""Day 8 检索器对比：BM25 / Dense / Hybrid 在同样的查询上，金标段落召回、耗时、内存各是多少。

用法：python -m experiments.day8_retriever_compare --split validation --agent-run outputs/runs/<V1 Agent 贪心运行>
进程内加载三个检索器（不走 HTTP）：耗时是检索器本身的耗时，HTTP 开销约 1ms、三种方法一样。
离线分析，金标只由本脚本（评测侧）读取，不改变任何实验结果。

两组查询：
- 原问题：每题一个查询，干净地比较检索器本身（相当于换了检索器的 Static RAG）
- V1 Agent 实际发出的查询：同一批查询换检索器。这些查询是看着 BM25 结果写的，不是真正的反事实
  （换了检索器 Agent 会写出不同的查询），真反事实要换检索器重跑 Agent。
  BM25 重放的结果必须和 V1 轨迹逐条一致，顺带检查索引和 ID 映射没有漂移。
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
from retrieval.dense import QUERY_PREFIX, DenseSearchTool
from retrieval.hybrid import HybridSearchTool

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bm25", "dense", "hybrid")
KS = (3, 5, 10, 20)


def _rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def _gold_ranks(docs, gold: list[str]) -> dict[str, int | None]:
    rank = {d.doc_id: d.rank for d in docs}
    return {g: rank.get(g) for g in gold}


def _recall(ranks: dict[str, int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks.values()) / len(ranks)


def _pct(xs: list[float]) -> dict:
    return {"mean": round(float(np.mean(xs)), 2), "p50": round(float(np.percentile(xs, 50)), 2),
            "p95": round(float(np.percentile(xs, 95)), 2)}


def _agent_queries(run: Path) -> dict[str, list[tuple[str, set[str]]]]:
    """每题 Agent 成功执行的搜索：(查询, V1 当时看到的 doc_id 集合)。"""
    out = {}
    for traj in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8")):
        out[traj["qid"]] = [(s["action"]["arguments"]["query"], {d["doc_id"] for d in s["observation"]["docs"]})
                            for s in traj["steps"]
                            if s["action"] and s["action"]["name"] == "search" and s["observation"]
                            and s["observation"]["ok"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--agent-run", help="V1 Agent 运行目录；给了才做第二组（Agent 查询重放）")
    ap.add_argument("--bm25-index", default="indexes/hotpot_pool_v1_bm25")
    ap.add_argument("--dense-index", default="indexes/hotpot_pool_v1_e5")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("检索器对比只在 validation / debug 上做，test 留给最终评测")

    data_dir = ROOT / "data/hotpotqa/v1"
    questions = [json.loads(l) for l in open(data_dir / "questions" / f"{args.split}.jsonl", encoding="utf-8")]
    labels = {l["qid"]: l for l in map(json.loads, open(data_dir / "labels" / f"{args.split}.jsonl", encoding="utf-8"))}

    memory = {"base_mb": _rss_mb()}
    bm25 = BM25SearchTool(ROOT / args.bm25_index)
    memory["after_bm25_mb"] = _rss_mb()
    dense = DenseSearchTool(ROOT / args.dense_index, device="cpu")
    memory["after_dense_mb"] = _rss_mb()
    tools = {"bm25": bm25, "dense": dense, "hybrid": HybridSearchTool(bm25, dense, args.pool_size, args.rrf_k)}
    memory = {k: round(v) for k, v in memory.items()}
    for t in tools.values():  # 预热：第一次调用有懒加载，不计入耗时
        t.search("warmup query", 5)

    latency = defaultdict(list)

    def timed(method: str, query: str, top_k: int):
        t0 = time.perf_counter()
        docs = tools[method].search(query, top_k)
        latency[method].append((time.perf_counter() - t0) * 1000)
        return docs

    # 第一组：原问题，取前 20 条，记录每个金标段落的名次
    rows = []
    for q in questions:
        lab = labels[q["qid"]]
        row = {"qid": q["qid"], "type": lab["type"], "question": q["question"], "gold_titles": lab["gold_titles"],
               "orig_ranks": {m: _gold_ranks(timed(m, q["question"], max(KS)), lab["gold_doc_ids"]) for m in METHODS}}
        rows.append(row)

    # 向量检索耗时拆成两段：查询编码（CPU，模型前向）和 FAISS 暴力搜索
    breakdown = defaultdict(list)
    for q in questions:
        t0 = time.perf_counter()
        v = dense.encoder.encode([QUERY_PREFIX + q["question"]])
        t1 = time.perf_counter()
        dense.index.search(v, max(KS))
        breakdown["encode_ms"].append((t1 - t0) * 1000)
        breakdown["faiss_ms"].append((time.perf_counter() - t1) * 1000)

    # 第二组：V1 Agent 的查询，top_k 和 V1 运行时一致
    replay_mismatch = None
    if args.agent_run:
        run = ROOT / args.agent_run
        top_k = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["budget"]["top_k"]
        agent_q = _agent_queries(run)
        replay_mismatch = 0
        for row in rows:
            gold = labels[row["qid"]]["gold_doc_ids"]
            seen = {m: set() for m in METHODS}
            for query, v1_seen in agent_q[row["qid"]]:
                for m in METHODS:
                    ids = {d.doc_id for d in timed(m, query, top_k)}
                    seen[m] |= ids
                    if m == "bm25" and ids != v1_seen:
                        replay_mismatch += 1
            row["agent_num_queries"] = len(agent_q[row["qid"]])
            row["agent_recall"] = {m: len(seen[m] & set(gold)) / len(gold) for m in METHODS}

    # 汇总：按题型分组，召回@k、两个金标都找齐的比例；方法差用配对 bootstrap
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["type"]].append(r)
    metrics: dict = {"num_questions": len(rows), "by_type": {}, "diff_vs_bm25": {}}
    for g, rs in groups.items():
        m_out = {"n": len(rs)}
        for m in METHODS:
            m_out[m] = {f"recall@{k}": round(float(np.mean([_recall(r["orig_ranks"][m], k) for r in rs])), 4)
                        for k in KS}
            m_out[m].update({f"all_gold@{k}": round(float(np.mean([_recall(r["orig_ranks"][m], k) == 1 for r in rs])), 4)
                             for k in KS})
            if args.agent_run:
                m_out[m]["agent_replay_recall"] = round(float(np.mean([r["agent_recall"][m] for r in rs])), 4)
        metrics["by_type"][g] = m_out
    diffs = {f"orig_recall@{k}": lambda r, m, k=k: _recall(r["orig_ranks"][m], k) for k in (3, 5)}
    if args.agent_run:
        diffs["agent_replay_recall"] = lambda r, m: r["agent_recall"][m]
    for name, f in diffs.items():
        for m in ("dense", "hybrid"):
            mean, lo, hi = paired_bootstrap([f(r, m) - f(r, "bm25") for r in rows])
            metrics["diff_vs_bm25"][f"{m}:{name}"] = [round(mean, 4), round(lo, 4), round(hi, 4)]

    # 互补性：前 5 条里，每个金标段落被哪一路找到
    overlap = defaultdict(int)
    for r in rows:
        for g in r["orig_ranks"]["bm25"]:
            b = (r["orig_ranks"]["bm25"][g] or 99) <= 5
            d = (r["orig_ranks"]["dense"][g] or 99) <= 5
            overlap["both" if b and d else "bm25_only" if b else "dense_only" if d else "neither"] += 1
    metrics["gold_overlap@5"] = dict(overlap)
    metrics["latency_ms"] = {m: _pct(xs) for m, xs in latency.items()}
    metrics["dense_breakdown_ms"] = {k: _pct(xs) for k, xs in breakdown.items()}
    metrics["memory_rss_mb"] = memory
    metrics["agent_replay_bm25_mismatch"] = replay_mismatch

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    out = ROOT / "outputs/runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-day8-retriever-compare-{args.split}"
                                   + ("-dirty" if dirty else ""))
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(
        {"args": vars(args), "indexes": {m: t.meta for m, t in tools.items()}}, allow_unicode=True), encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + ("-dirty" if dirty else "") + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
