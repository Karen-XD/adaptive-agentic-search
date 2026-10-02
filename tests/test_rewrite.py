"""Day 10 改写流程：每种计划搜几次、改写看到了什么、证据怎么合并、成本怎么记、解析失败怎么退回。

用按剧本输出的假模型，只验证流程，不验证模型能力。
运行：pytest tests/test_rewrite.py
"""
import pytest

from agent.llm import ScriptedLLM
from agent.methods import check_budget, run_method
from agent.parser import parse_action
from agent.prompts import (ANSWER_ONLY_PROMPTS, EVIDENCE_REWRITE_SYSTEM_PROMPT, STATIC_REWRITE_SYSTEM_PROMPT,
                           render_observation)
from agent.rewrite import PLANS, is_repeat
from agent.schema import Budget, StopReason
from tests.helpers import call
from tests.test_methods import ANSWER, Q, SpyTool


def run(method, rewrites, tool=None, top_k=3):
    tool = tool or SpyTool()
    llm = ScriptedLLM([*rewrites, ANSWER])
    traj = run_method(method, "q-001", Q, llm, tool, Budget(max_search_calls=len(PLANS[method]), top_k=top_k))
    return traj, llm, tool


def test_two_hop_evidence_searches_original_then_rewrite():
    traj, llm, tool = run("two_hop_evidence", [call("search", query="Tessa Marrow birthplace")])
    assert [q for q, _ in tool.calls] == [Q, "Tessa Marrow birthplace"]
    s = traj.context.searches
    assert [r.kind for r in s] == ["original", "evidence_rewrite"] and not s[1].fallback
    # 改写看到的是 Agent 格式：第一次检索写成自己发出的 search 调用 + 工具返回；作答用的是 B3 那份提示词
    rewrite_msgs = llm.seen_messages[0]
    assert [m["role"] for m in rewrite_msgs] == ["system", "user", "assistant", "tool"]
    assert rewrite_msgs[0]["content"] == EVIDENCE_REWRITE_SYSTEM_PROMPT and rewrite_msgs[1]["content"] == f"Question: {Q}"
    assert parse_action(rewrite_msgs[2]["content"]).action.arguments.query == Q  # 和模型自己写的调用同一种格式
    assert rewrite_msgs[3]["content"] == render_observation(traj.context.searches[0].observation)
    assert llm.seen_messages[1][0]["content"] == ANSWER_ONLY_PROMPTS.system
    assert traj.budget_state.search_calls_used == 2 and traj.stop_reason == StopReason.ANSWERED


def test_static_rewrite_sees_only_the_question():
    traj, llm, tool = run("two_hop_static", [call("search", query="Luminara Labs founder")])
    assert llm.seen_messages[0][0]["content"] == STATIC_REWRITE_SYSTEM_PROMPT
    assert llm.seen_messages[0][1]["content"] == f"Question: {Q}"  # 不能看到第一次检索的结果
    assert [r.kind for r in traj.context.searches] == ["original", "static_rewrite"]


def test_rewrite_rag_replaces_the_original_query():
    traj, _, tool = run("rewrite_rag", [call("search", query="Luminara Labs founder")])
    assert [q for q, _ in tool.calls] == ["Luminara Labs founder"]
    assert traj.budget_state.search_calls_used == 1 and traj.context.query == "Luminara Labs founder"


def test_merged_evidence_is_deduplicated_and_renumbered():
    traj, llm, _ = run("two_hop_evidence", [call("search", query="Tessa Marrow")], top_k=2)
    docs = traj.context.observation.docs
    assert [d.rank for d in docs] == list(range(1, len(docs) + 1))
    assert len({d.doc_id for d in docs}) == len(docs)
    first = {d.doc_id for d in traj.context.searches[0].observation.docs}
    assert traj.context.searches[1].num_new_docs == len({d.doc_id for d in traj.context.searches[1].observation.docs} - first)
    assert "[1]" in llm.seen_messages[1][1]["content"]  # 合并后的证据放进了作答的用户消息


def test_unparseable_rewrite_falls_back_to_question():
    traj, _, tool = run("two_hop_evidence", ["I think the answer is Port Edvik."])
    r = traj.context.searches[1]
    assert r.fallback and r.query == Q and [q for q, _ in tool.calls] == [Q, Q]
    assert is_repeat(r, traj.context.searches[:1])  # 退回原问题 = 白搜一次，统计成重复
    assert traj.budget_state.search_calls_used == 2  # 预算照扣：各组成本口径一致


def test_failed_searches_still_answer():
    traj, _, _ = run("two_hop_evidence", [call("search", query="x")], tool=SpyTool(fail=True))
    assert not traj.context.observation.ok and traj.final_answer == "Port Edvik"


@pytest.mark.parametrize("method,n", [("rewrite_rag", 1), ("two_hop_static", 2), ("two_hop_evidence", 2)])
def test_budget_must_match_plan(method, n):
    check_budget(method, Budget(max_search_calls=n))
    with pytest.raises(ValueError):
        check_budget(method, Budget(max_search_calls=n + 1))
