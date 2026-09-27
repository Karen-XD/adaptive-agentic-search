"""评测指标：答案归一化与论文一致；模型服务失败的题留在准确率的分母里。

运行：pytest tests/ 或 python -m tests.test_eval
"""
import pytest

from evaluation.qa_metrics import aggregate, exact_match


@pytest.mark.parametrize("pred, gold, same", [
    ("The Lighthouse.", "lighthouse", True),  # 冠词、标点、大小写不影响
    ("Port  Edvik", "port edvik", True),
    ("Port Edvik harbor", "Port Edvik", False),
    (None, "Port Edvik", False),              # 没作答
])
def test_exact_match(pred, gold, same):
    assert exact_match(pred, gold) is same


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
