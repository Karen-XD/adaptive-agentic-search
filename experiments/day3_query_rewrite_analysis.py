"""Day 3.5 查询改写得失：Agent 自己写的查询，比直接拿原问题搜，找到的金标段落更多还是更少？

用法（先起检索服务）：python -m experiments.day3_query_rewrite_analysis outputs/runs/<agent 运行>
离线分析，读答案文件里的金标段落 ID 是评测侧行为；不改变任何实验结果。

两个口径都按 top_k（和运行时一致）算：
- 第一个查询 vs 原问题：同样只搜一次，改写本身是得是失
- 整条轨迹 vs 原问题：多搜几次之后，总共多找到多少
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from retrieval.client import HttpSearchTool


def coverage(doc_ids: set[str], gold: set[str]) -> float:
    return len(doc_ids & gold) / len(gold)


def main(run_dir: str) -> None:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    top_k = cfg["budget"]["top_k"]
    tool = HttpSearchTool(cfg["retrieval"]["url"])
    labels = {l["qid"]: l for l in map(json.loads, open(Path(cfg["data"]["dir"]) / "labels" / f"{cfg['data']['split']}.jsonl"))}

    rows = []
    for traj in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8")):
        gold = set(labels[traj["qid"]]["gold_doc_ids"])
        queries, seen = [], set()
        for step in traj["steps"]:
            obs = step["observation"]
            if step["action"] and step["action"]["name"] == "search" and obs and obs["ok"]:
                queries.append(step["action"]["arguments"]["query"])
                seen |= {d["doc_id"] for d in obs["docs"]}
        original = {d.doc_id for d in tool.search(traj["question"], top_k)}
        first = {d.doc_id for d in tool.search(queries[0], top_k)} if queries else set()
        rows.append({"type": labels[traj["qid"]]["type"], "n": len(queries),
                     "orig": coverage(original, gold), "first": coverage(first, gold) if queries else None,
                     "traj": coverage(seen, gold), "em": traj["eval"]["em"]})

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
        by_type["all"].append(r)
    print(f"金标覆盖率（top_k={top_k}）：原问题搜一次 / Agent 第一个查询 / Agent 整条轨迹")
    for t, rs in sorted(by_type.items()):
        searched = [r for r in rs if r["first"] is not None]
        better = sum(r["first"] > r["orig"] for r in searched)
        worse = sum(r["first"] < r["orig"] for r in searched)
        mean = lambda k, xs: sum(r[k] for r in xs) / len(xs)
        print(f"  {t:10s} {len(rs):3d} 题 | 原问题 {mean('orig', rs):.2f} | 第一个查询 {mean('first', searched):.2f}"
              f"（变好 {better} / 变差 {worse} / 持平 {len(searched) - better - worse}）| 整条轨迹 {mean('traj', rs):.2f}"
              f" | 平均搜 {mean('n', rs):.2f} 次")


if __name__ == "__main__":
    main(sys.argv[1])
