"""评测指标：答案归一化与论文一致；模型服务失败的题留在准确率的分母里。

运行：pytest tests/ 或 python -m tests.test_eval
"""
import pytest

from evaluation.qa_metrics import aggregate, exact_match, f1_score


@pytest.mark.parametrize("pred, gold, same", [
    ("The Lighthouse.", "lighthouse", True),  # 冠词、标点、大小写不影响
    ("Port  Edvik", "port edvik", True),
    ("Port Edvik harbor", "Port Edvik", False),
    (None, "Port Edvik", False),              # 没作答
])
def test_exact_match(pred, gold, same):
    assert exact_match(pred, gold) is same


@pytest.mark.parametrize("pred, gold, f1", [
    ("Atlanta", "Atlanta, Georgia", 2 / 3),            # EM 记 0，F1 给部分分
    ("Latvia", "Latvian", 0.0),                         # 不做词干化，和官方脚本一致
    ("Yes, both are directors.", "yes", 0.0),           # yes/no 题只认完全一致
    ("yes", "yes", 1.0),
    (None, "yes", 0.0),
])
def test_f1(pred, gold, f1):
    assert f1_score(pred, gold) == pytest.approx(f1)


def test_cost_metrics():
    recs = [{**_rec(True, "answered"), "prompt_tokens": p, "completion_tokens": 10, "latency_ms": ms,
             "llm_call_ms": [ms]} for p, ms in [(100, 100.0), (300, 200.0), (500, 1000.0)]]
    m = aggregate(recs)
    assert m["mean_prompt_tokens"] == 300 and m["mean_completion_tokens"] == 10
    assert m["latency_ms"]["p50"] == 200.0 and m["latency_ms"]["p95"] == pytest.approx(920.0)


def test_no_token_counts_are_not_reported_as_zero():
    m = aggregate([{**_rec(True, "answered"), "prompt_tokens": None, "completion_tokens": None}])
    assert "mean_prompt_tokens" not in m


def _rec(correct, stop_reason, error=False):
    return {"correct": correct, "error": error, "stop_reason": stop_reason, "turns": 2,
            "search_calls": 1, "search_attempts": 1, "new_docs": 2, "format_errors": 0}


def test_failed_questions_stay_in_denominator():
    m = aggregate([_rec(True, "answered"), _rec(True, "answered"), _rec(False, "no_answer"),
                   _rec(False, "error", error=True)])
    assert m["accuracy"] == 0.5                       # 2 / 4：失败题计 0 分
    assert m["accuracy_excluding_errors"] == 2 / 3    # 诊断指标：去掉失败题
    assert m["failure_rate"] == 0.25
    assert m["stop_reasons"] == {"answered": 2, "no_answer": 1, "error": 1}



def test_evidence_metrics():
    recs = [{**_rec(False, "answered"), "evidence_recall": r} for r in (1.0, 0.5, 0.0, None)]
    m = aggregate(recs)
    assert m["evidence_recall"] == 0.5          # 没有金标的题（None）不计入
    assert m["all_evidence_found"] == 1 / 3

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
