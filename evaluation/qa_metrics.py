"""QA 评测指标。答案提取规则必须和自己的 Agent 解析器一致，否则比较不公平。

normalize_answer 沿用 HotpotQA / Search-R1 的标准归一化，让我们的 EM 和论文可比。
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_SPECIAL = {"yes", "no", "noanswer"}  # 这几种答案 F1 只认完全一致


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_score(prediction: str | None, gold: str) -> float:
    """词级 F1，照搬 HotpotQA 官方评测脚本：预测和金标各自归一化后按词算精确率、召回率。
    EM 太严（"Atlanta" vs "Atlanta, Georgia" 记 0 分），F1 给部分分。
    yes / no 题只有完全一致才得分：否则答一整句 "Yes, both are directors" 也能靠 yes 拿到部分分。"""
    if prediction is None:
        return 0.0
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _SPECIAL or ref in _SPECIAL) and pred != ref:
        return 0.0
    common = sum((Counter(pred.split()) & Counter(ref.split())).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred.split()), common / len(ref.split())
    return 2 * precision * recall / (precision + recall)


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


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
        "f1": sum(r.get("f1", 0.0) for r in records) / n,
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
        **_cost_metrics(records),
    }


def _cost_metrics(records: list[dict]) -> dict:
    """成本：token 数（输入、输出分开报：多轮 Agent 贵在反复读上下文，延迟却主要由输出决定）和延迟分位数。
    假模型没有 token 数（记 None），整次运行都没有时就不报，避免报成"零成本"。"""
    out = {}
    with_tokens = [r for r in records if r.get("prompt_tokens") is not None]
    if with_tokens:
        out["mean_prompt_tokens"] = sum(r["prompt_tokens"] for r in with_tokens) / len(with_tokens)
        out["mean_completion_tokens"] = sum(r["completion_tokens"] for r in with_tokens) / len(with_tokens)
    if any("latency_ms" in r for r in records):
        # 每题端到端（模型 + 检索），用户实际等的时间；P95 看长尾：多轮的题会拖在尾巴上
        out["latency_ms"] = _percentiles([r["latency_ms"] for r in records if "latency_ms" in r])
        out["llm_call_latency_ms"] = _percentiles([x for r in records for x in r.get("llm_call_ms", [])])
    return out


def _evidence_metrics(records: list[dict]) -> dict:
    with_gold = [r for r in records if r.get("evidence_recall") is not None]
    if not with_gold:
        return {}
    return {
        "evidence_recall": sum(r["evidence_recall"] for r in with_gold) / len(with_gold),
        # 多跳题要把所有金标段落都找齐才可能答对，这个比平均召回更能说明问题
        "all_evidence_found": sum(r["evidence_recall"] == 1.0 for r in with_gold) / len(with_gold),
    }
