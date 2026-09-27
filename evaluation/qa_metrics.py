"""QA 评测指标。答案提取规则必须和自己的 Agent 解析器一致，否则比较不公平。

normalize_answer 沿用 HotpotQA / Search-R1 的标准归一化，让我们的 EM 和论文可比。
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    return normalize_answer(prediction) == normalize_answer(gold)


def aggregate(records: list[dict]) -> dict:
    """records 里每项含 correct / error / stop_reason / 各计数。失败题留在分母里。"""
    n = len(records)
    if n == 0:
        return {"num_questions": 0}
    ok = [r for r in records if not r["error"]]
    total_calls = sum(r["search_calls"] for r in records)
    total_turns = sum(r["turns"] for r in records)
    return {
        "num_questions": n,
        # 主指标：失败题计 0 分并留在分母里（跳过失败题会抬高准确率且让方法间不可比）
        "accuracy": sum(r["correct"] for r in records) / n,
        "num_errors": n - len(ok),
        "failure_rate": (n - len(ok)) / n,
        # 诊断指标：只看跑完的题，用来量化"失败率有多影响结论"
        "accuracy_excluding_errors": (sum(r["correct"] for r in ok) / len(ok)) if ok else None,
        "mean_search_calls": total_calls / n,
        "mean_search_attempts": sum(r["search_attempts"] for r in records) / n,
        "mean_turns": total_turns / n,
        # 格式错误轮次占比：Day 3 接真模型后判断"模型会不会写工具调用"的第一个指标
        "format_error_rate": sum(r["format_errors"] for r in records) / total_turns if total_turns else 0.0,
        "mean_new_docs_per_call": sum(r["new_docs"] for r in records) / total_calls if total_calls else 0.0,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
        **_evidence_metrics(records),
    }


def _evidence_metrics(records: list[dict]) -> dict:
    with_gold = [r for r in records if r.get("evidence_recall") is not None]
    if not with_gold:
        return {}
    return {
        "evidence_recall": sum(r["evidence_recall"] for r in with_gold) / len(with_gold),
        # 多跳题要把所有金标段落都找齐才可能答对，这个比平均召回更能说明问题
        "all_evidence_found": sum(r["evidence_recall"] == 1.0 for r in with_gold) / len(with_gold),
    }
