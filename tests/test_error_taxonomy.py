"""Day 5 错题分类：主因的判断顺序和几条边界规则。

运行：pytest tests/ 或 python -m tests.test_error_taxonomy
"""
import pytest

from experiments.day5_error_taxonomy import answer_form_issue, classify

BUDGET = {"max_search_calls": 3}
GOLD = ["g1", "g2"]


def doc(doc_id, text=""):
    return {"doc_id": doc_id, "title": doc_id, "text": text}


def search_step(docs, new_docs=None):
    return {"action": {"name": "search", "arguments": {"query": "q"}}, "observation": {"ok": True, "docs": docs},
            "num_new_docs": len(docs) if new_docs is None else new_docs}


def traj(steps, answer="wrong", stop="answered", used=None):
    used = sum(1 for s in steps if s["action"]["name"] == "search") if used is None else used
    return {"method": "agent", "question": "Who founded the company?", "final_answer": answer, "stop_reason": stop,
            "context": None, "steps": steps, "budget_state": {"search_calls_used": used}}


def label(answer="Tessa Marrow", type_="bridge"):
    return {"answer": answer, "type": type_, "gold_doc_ids": GOLD}


@pytest.mark.parametrize("pred, gold, expected", [
    ("Michele Bachmann", "Michele Marie Bachmann", True),  # 金标带中间名：按词集合判，不按子串
    ("River Calder", "Calder", True),
    ("French", "yes", True),                                # 问是否却答了实体
    ("no", "yes", False),                                   # yes/no 答反了是真错
    ("Port Edvik", "Tessa Marrow", False),
])
def test_answer_form_issue(pred, gold, expected):
    assert answer_form_issue(pred, gold) == expected


def test_no_answer_wins_over_everything():
    assert classify(traj([search_step([doc("g1"), doc("g2")])], answer=None, stop="no_answer"),
                    label(), BUDGET, None) == "格式非法"


def test_all_gold_found_is_reading_error():
    assert classify(traj([search_step([doc("g1"), doc("g2")])]), label(), BUDGET, None) == "证据够了仍读错"


def test_answer_in_retrieved_text_counts_as_enough_evidence_for_bridge_only():
    steps = [search_step([doc("g1"), doc("x", "founded by Tessa Marrow in 1990")])]
    assert classify(traj(steps), label(), BUDGET, None) == "证据够了仍读错"
    assert classify(traj(steps), label(type_="comparison"), BUDGET, None) != "证据够了仍读错"
    # 纯数字答案容易碰巧命中，不算
    assert classify(traj([search_step([doc("g1"), doc("x", "17 members")])]), label("17"), BUDGET, None) == "过早停止"


def test_stopping_with_budget_left_is_premature():
    assert classify(traj([search_step([doc("g1")])]), label(), BUDGET, None) == "过早停止"


def test_original_question_would_have_found_it():
    steps = [search_step([doc("g1")])] * 3
    assert classify(traj(steps, stop="forced_answer"), label(), BUDGET, {"g2"}) == "查询写得差"


def test_wasted_search_after_budget_runs_out():
    steps = [search_step([doc("g1")]), search_step([doc("g1")], new_docs=0), search_step([doc("x")])]
    assert classify(traj(steps, stop="forced_answer"), label(), BUDGET, None) == "无效重复搜索"


@pytest.mark.parametrize("docs, type_, expected", [
    ([doc("g1")], "bridge", "缺第二跳"),
    ([doc("g1")], "comparison", "缺一个实体"),
    ([doc("x")], "bridge", "检索未命中"),
])
def test_retriever_capability_categories(docs, type_, expected):
    steps = [search_step(docs), search_step([doc("y")]), search_step([doc("z")])]
    assert classify(traj(steps, stop="forced_answer"), label(type_=type_), BUDGET, None) == expected


def test_answering_after_using_all_searches_is_not_premature():
    # 搜满了才作答，证据仍不够：是检索的问题，不是停早了
    steps = [search_step([doc("g1")]), search_step([doc("y")]), search_step([doc("z")])]
    assert classify(traj(steps, stop="answered"), label(), BUDGET, None) == "缺第二跳"
