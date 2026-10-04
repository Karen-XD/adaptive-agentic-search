"""商品搜索的排序指标：nDCG@k（主）、E 类召回@k、第一个 E 的倒数排名。

增益映射 E=3, S=2, C=1, I=0 是**本项目人为设定**，不是官方业务效用（计划 4.2 要求预先声明）。
Complement（互补品）算不算"好结果"是有争议的，所以映射做成参数，敏感性分析时把 C 调成 0 或 1 再跑一遍。

这里的指标都在**固定候选商品集**上算（Setting A）：每个查询的候选就是官方已标注的那批商品，
所以它衡量的是"排序好坏"，不是"全库召回率"。全库设定（Setting B）另算已标注正例的覆盖率。
标签只在评测器里用，绝不进 prompt 或检索工具。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping

GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}
# 敏感性分析用：Complement 不计分（C 是"互补品"，和查询意图不同但可能有搭配价值）
GAIN_NO_COMPLEMENT = {"E": 3, "S": 2, "C": 0, "I": 0}


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_labels: list[str], k: int = 10, gain: Mapping[str, float] = GAIN) -> float:
    """ranked_labels 是按方法打分排好序的 ESCI 标签序列（只取前 k 位）。"""
    gains = [gain[label] for label in ranked_labels[:k]]
    ideal = sorted((gain[label] for label in ranked_labels), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


def recall_at_k(ranked_labels: list[str], k: int = 10, positive: str = "E") -> float:
    """前 k 位里覆盖了多少 E（精确匹配）商品。一个 E 都没有时返回 1.0（没有可错的）。"""
    total = sum(label == positive for label in ranked_labels)
    if total == 0:
        return 1.0
    return sum(label == positive for label in ranked_labels[:k]) / total


def reciprocal_rank(ranked_labels: list[str], positive: str = "E") -> float:
    for i, label in enumerate(ranked_labels):
        if label == positive:
            return 1.0 / (i + 1)
    return 0.0


def aggregate(records: list[dict], gain: Mapping[str, float] = GAIN, k: int = 10) -> dict:
    """records 每项含 ranked_labels（方法排序后的标签序列）和可选的成本字段。失败题计 0 分并留在分母里。"""
    n = len(records)
    if n == 0:
        return {"num_queries": 0}
    ok = [r for r in records if not r.get("error")]
    out = {
        "num_queries": n,
        "ndcg@10": sum(ndcg_at_k(r["ranked_labels"], k, gain) for r in records) / n,
        "recall@10": sum(recall_at_k(r["ranked_labels"], k) for r in records) / n,
        "mrr": sum(reciprocal_rank(r["ranked_labels"]) for r in records) / n,
        "num_errors": n - len(ok),
        "mean_candidates": sum(len(r["ranked_labels"]) for r in records) / n,
        "label_dist": dict(Counter(l for r in records for l in r["ranked_labels"])),
    }
    if any("latency_ms" in r for r in records):
        lat = sorted(r["latency_ms"] for r in records if "latency_ms" in r)
        out["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}
    if any("tokens" in r for r in records):
        toks = [r["tokens"] for r in records if "tokens" in r]
        out["mean_tokens"] = sum(toks) / len(toks) if toks else 0.0
    return out
