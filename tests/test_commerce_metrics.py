"""商品排序指标：nDCG@10 的定义、增益映射的影响、E 召回和 MRR。

运行：pytest tests/ 或 python -m tests.test_commerce_metrics
"""
import math

import pytest

from evaluation.commerce_metrics import GAIN, GAIN_NO_COMPLEMENT, aggregate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ndcg_perfect_and_worst():
    # 完全按 E > S > C > I 排：nDCG = 1
    assert ndcg_at_k(["E", "E", "S", "I"], k=4) == pytest.approx(1.0)
    # 完全倒序：C/E 增益都比 I 高，但 I 在前
    assert ndcg_at_k(["I", "I", "I", "I"], k=4) == pytest.approx(0.0)


def test_ndcg_hand_computed():
    # ["E","S"] 已经是最优排序（增益 3 > 2），nDCG = 1
    assert ndcg_at_k(["E", "S"], k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["E", "S"], k=1) == pytest.approx(1.0)
    # ["S","E"] 把 E 排到了第 2：DCG = 2/log2(2) + 3/log2(3) = 2 + 1.8928
    # 理想排序 ["E","S"]：IDCG = 3/log2(2) + 2/log2(3) = 3 + 1.2619
    dcg = 2 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert ndcg_at_k(["S", "E"], k=2) == pytest.approx(dcg / idcg)


def test_rank_position_matters():
    a = ndcg_at_k(["E", "S", "I", "I"], k=4)
    b = ndcg_at_k(["S", "E", "I", "I"], k=4)
    assert a > b  # E 排第 1 比排第 2 好


def test_complement_sensitivity():
    """把 C 的增益调成 0 后，C 排前面不再有奖励（敏感性分析用的口径）。"""
    assert ndcg_at_k(["C", "E", "S", "I"], k=4, gain=GAIN) > ndcg_at_k(["C", "E", "S", "I"], k=4, gain=GAIN_NO_COMPLEMENT)
    assert ndcg_at_k(["E", "S", "C", "I"], k=4, gain=GAIN_NO_COMPLEMENT) == pytest.approx(1.0)


def test_recall_and_mrr():
    assert recall_at_k(["E", "I", "E", "I"], k=10) == pytest.approx(1.0)
    assert recall_at_k(["E", "I", "E", "I"], k=1) == pytest.approx(0.5)
    assert recall_at_k(["I", "S"], k=10) == pytest.approx(1.0)  # 没有 E，不给惩罚
    assert reciprocal_rank(["I", "S", "E"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["I", "S"]) == 0.0


def test_aggregate_keeps_failures_in_denominator():
    records = [{"ranked_labels": ["E", "S", "I", "I"]}, {"ranked_labels": [], "error": "boom"}]
    out = aggregate(records)
    assert out["num_queries"] == 2
    assert out["num_errors"] == 1
    assert out["ndcg@10"] == pytest.approx(ndcg_at_k(["E", "S", "I", "I"], 10) / 2)
