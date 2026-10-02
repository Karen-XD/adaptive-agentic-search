"""检索器路由的上限：按题挑 BM25 / Dense / Hybrid，最多能比固定用一路多找到多少金标？HotpotQA 和 2Wiki 共用这一份代码。

用法：
  python -m experiments.routing_ceiling --data data/hotpotqa/v1 --split validation \
      --bm25-index indexes/hotpot_pool_v1_bm25 --dense-index indexes/hotpot_pool_v1_e5
  python -m experiments.routing_ceiling --data data/2wiki/v1 --split analysis \
      --bm25-index indexes/2wiki_pool_v1_bm25 --dense-index indexes/2wiki_pool_v1_e5
离线分析，金标只由本脚本（评测侧）读取。四个口径从低到高：
  固定        所有题用同一路
  按题型路由  每类题用这一类上最好的一路（题型从问题句式基本能看出来，接近真实路由器能做到的）
  逐题理想    每题事后挑最好的一路（路由的上限，真实路由器达不到）
  三路都没找到的金标：换检索器救不回来，要靠改写 / 拆解问题
理想子查询（只有标签里带 evidences 的数据集，即 2Wiki）：每个三元组（实体, 关系, 值）造一个查询"实体 关系"，
金标是该实体的段落。相当于"Agent 已经拆对了问题、找到了桥接实体"，看每一跳该用哪一路。实体名去掉括号里的消歧义词，
因为 Agent 从段落正文里读到的实体名不带它。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from experiments.compare_runs import paired_bootstrap
from retrieval.bm25 import BM25SearchTool
from retrieval.dense import DenseSearchTool
from retrieval.hybrid import rrf_fuse

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bm25", "dense", "hybrid")
KS = (3, 5, 10)
POOL = 20  # 和检索服务的 hybrid 默认值一致：每路取 20 再融合


def _retrieve(tools: dict, query: str) -> dict[str, list[str]]:
    lists = {m: tools[m].search(query, POOL) for m in ("bm25", "dense")}
    lists["hybrid"] = rrf_fuse([lists["bm25"], lists["dense"]], POOL)
    return {m: [d.doc_id for d in docs] for m, docs in lists.items()}


def _recall(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def _strip_paren(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def _ceilings(recalls: list[dict[str, float]], types: list[str]) -> dict:
    """recalls[i][method] = 第 i 题该方法的召回。返回四个口径和理想路由相对最好固定的增益。"""
    n = len(recalls)
    fixed = {m: float(np.mean([r[m] for r in recalls])) for m in METHODS}
    best_fixed = max(fixed, key=fixed.get)
    by_type = defaultdict(list)
    for r, t in zip(recalls, types):
        by_type[t].append(r)
    type_choice = {t: max(METHODS, key=lambda m: np.mean([r[m] for r in rs])) for t, rs in by_type.items()}
    type_routed = [r[type_choice[t]] for r, t in zip(recalls, types)]
    ideal2 = [max(r["bm25"], r["dense"]) for r in recalls]
    ideal3 = [max(r[m] for m in METHODS) for r in recalls]
    win = Counter("bm25" if r["bm25"] > r["dense"] else "dense" if r["dense"] > r["bm25"] else "tie" for r in recalls)
    gain = paired_bootstrap([i - r[best_fixed] for i, r in zip(ideal3, recalls)])
    return {"n": n, "fixed": {m: round(v, 4) for m, v in fixed.items()}, "best_fixed": best_fixed,
            "type_routed": round(float(np.mean(type_routed)), 4), "type_choice": type_choice,
            "ideal_bm25_dense": round(float(np.mean(ideal2)), 4), "ideal_all3": round(float(np.mean(ideal3)), 4),
            "ideal_gain_vs_best_fixed": [round(x, 4) for x in gain],
            "bm25_vs_dense_wins": dict(win)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--bm25-index", required=True)
    ap.add_argument("--dense-index", required=True)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("路由分析只在 validation / analysis / debug 上做")

    data = ROOT / args.data
    questions = [json.loads(l) for l in open(data / "questions" / f"{args.split}.jsonl", encoding="utf-8")]
    labels = {l["qid"]: l for l in map(json.loads, open(data / "labels" / f"{args.split}.jsonl", encoding="utf-8"))}
    bm25 = BM25SearchTool(ROOT / args.bm25_index)
    dense = DenseSearchTool(ROOT / args.dense_index, device="cpu")
    if [d["doc_id"] for d in bm25.docs] != [d["doc_id"] for d in dense.docs]:
        raise SystemExit("bm25 and dense indexes were built on different corpora")
    tools = {"bm25": bm25, "dense": dense}
    title_to_id = {d["title"]: d["doc_id"] for d in bm25.docs}

    rows = []
    for q in questions:
        lab = labels[q["qid"]]
        ranked = _retrieve(tools, q["question"])
        row = {"qid": q["qid"], "type": lab["type"], "num_gold": len(lab["gold_doc_ids"]),
               "orig": {m: {k: _recall(ranked[m], lab["gold_doc_ids"], k) for k in KS} for m in METHODS},
               "orig_missed_all": {k: len(set(lab["gold_doc_ids"]) - set().union(*(ranked[m][:k] for m in METHODS)))
                                   / len(lab["gold_doc_ids"]) for k in KS}}
        if "evidences" in lab:  # 理想子查询：每个不同的主语实体一跳（同一实体多个关系时只留第一个）
            hops, seen_subj = [], set()
            for subj, rel, _ in lab["evidences"]:
                if subj in seen_subj or subj not in title_to_id:
                    continue
                seen_subj.add(subj)
                r = _retrieve(tools, f"{_strip_paren(subj)} {rel}")
                gid = title_to_id[subj]
                hops.append({"query": f"{_strip_paren(subj)} {rel}",
                             "rank": {m: (r[m].index(gid) + 1 if gid in r[m] else None) for m in METHODS}})
            row["hops"] = hops
            row["evidence_subjects_not_gold"] = sum(title_to_id.get(s) not in lab["gold_doc_ids"]
                                                    for s, _, _ in lab["evidences"])
        rows.append(row)

    types = sorted({r["type"] for r in rows})
    metrics: dict = {"num_questions": len(rows), "orig_question": {}, "orig_missed_by_all3": {}}
    for k in KS:
        metrics["orig_question"][f"@{k}"] = {
            "by_type": {t: _ceilings([r["orig"] and {m: r["orig"][m][k] for m in METHODS} for r in rows if r["type"] == t],
                                     [t] * sum(r["type"] == t for r in rows)) for t in types},
            "all_types": _ceilings([{m: r["orig"][m][k] for m in METHODS} for r in rows], [r["type"] for r in rows]),
        }
        metrics["orig_missed_by_all3"][f"@{k}"] = {
            t: round(float(np.mean([r["orig_missed_all"][k] for r in rows if r["type"] == t])), 4) for t in types}

    if all("hops" in r for r in rows):
        hop_rows = [(r["type"], h) for r in rows for h in r["hops"]]
        metrics["oracle_subqueries"] = {"num_hops": len(hop_rows),
                                        "evidence_subjects_not_gold": sum(r["evidence_subjects_not_gold"] for r in rows)}
        for k in KS:
            hit = lambda h, m: float(h["rank"][m] is not None and h["rank"][m] <= k)
            per_k = {}
            for t in types + ["all_types"]:
                hs = [h for tt, h in hop_rows if t == "all_types" or tt == t]
                recalls = [{m: hit(h, m) for m in METHODS} for h in hs]
                per_k[t] = _ceilings(recalls, [tt for tt, h in hop_rows if t == "all_types" or tt == t])
                per_k[t]["missed_by_all3"] = round(float(np.mean([max(r.values()) == 0 for r in recalls])), 4)
            metrics["oracle_subqueries"][f"@{k}"] = per_k

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    name = Path(args.data).parts[-2] if Path(args.data).name.startswith("v") else Path(args.data).name
    out = ROOT / "outputs/runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-routing-ceiling-{name}-{args.split}"
                                   + ("-dirty" if dirty else ""))
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"args": vars(args), "pool": POOL, "ks": list(KS),
                                                     "indexes": {"bm25": bm25.meta, "dense": dense.meta}},
                                                    allow_unicode=True), encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + ("-dirty" if dirty else "") + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
