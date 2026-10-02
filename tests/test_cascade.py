"""Day 11 按需升级：门控放不放行、探测结果怎么处理、成本和预算怎么记，以及和 B3 / 两跳证据改写的逐字等价关系。

用按剧本输出的假模型，只验证流程。
运行：pytest tests/test_cascade.py
"""
import pytest

from agent.cascade import CascadeConfig, rerank_gap
from agent.llm import ScriptedLLM
from agent.methods import check_budget, run_method
from agent.prompts import AGENT_PROMPTS, ANSWER_ONLY_PROMPTS, EVIDENCE_REWRITE_SYSTEM_PROMPT
from agent.schema import Budget, Doc, Observation, StopReason
from tests.helpers import call
from tests.test_methods import ANSWER, Q, SpyTool

BUDGET = Budget(max_search_calls=2)


class ScoredTool(SpyTool):
    """MockSearchTool 的分数是词重叠个数；这里改成固定的分数，方便测门控。"""

    def __init__(self, scores, **kw):
        super().__init__(**kw)
        self.scores = scores

    def search(self, query, top_k):
        docs = super().search(query, top_k)
        return [d.model_copy(update={"score": s}) for d, s in zip(docs, self.scores)]


def run(cfg, outputs, tool=None):
    tool = tool or SpyTool()
    llm = ScriptedLLM(list(outputs))
    traj = run_method("cascade", "q-001", Q, llm, tool, BUDGET, cascade=CascadeConfig(**cfg))
    return traj, llm, tool


def test_gate_never_is_b3():
    traj, llm, tool = run({"gate": "never"}, [ANSWER])
    b3, b3_llm = None, ScriptedLLM([ANSWER])
    b3 = run_method("static_rag", "q-001", Q, b3_llm, SpyTool(), Budget(max_search_calls=1))
    assert tool.calls == [(Q, 3)] and len(llm.seen_messages) == 1
    assert llm.seen_messages == b3_llm.seen_messages  # 作答看到的内容和 B3 逐字相同
    assert traj.escalation.outcome == "not_probed" and not traj.escalation.probed
    assert traj.budget_state.search_calls_used == 1 and traj.final_answer == b3.final_answer


def test_probe_rewrite_escalates_like_two_hop_evidence():
    out = [call("search", query="Tessa Marrow birthplace"), ANSWER]
    traj, llm, tool = run({"gate": "always"}, out)
    ref_llm = ScriptedLLM(list(out))
    ref = run_method("two_hop_evidence", "q-001", Q, ref_llm, SpyTool(), BUDGET)
    assert [q for q, _ in tool.calls] == [Q, "Tessa Marrow birthplace"]
    assert llm.seen_messages == ref_llm.seen_messages  # 探测和作答看到的内容都和两跳证据改写逐字相同
    assert llm.seen_messages[0][0]["content"] == EVIDENCE_REWRITE_SYSTEM_PROMPT
    assert traj.escalation.outcome == "escalated" and traj.budget_state.search_calls_used == 2
    assert traj.context.observation.docs == ref.context.observation.docs
    assert traj.steps[0].prompt_tokens is None and len(traj.steps) == 2  # 探测一步 + 作答一步，都记进轨迹


def test_probe_rewrite_declines_and_answers_from_first_hop():
    traj, llm, tool = run({"gate": "always"}, ["The founder was born in Port Edvik.", ANSWER])
    assert tool.calls == [(Q, 3)]  # 不想再搜就不搜：比两跳那组少一次检索
    assert traj.escalation.outcome == "declined" and traj.budget_state.search_calls_used == 1
    assert llm.seen_messages[1][0]["content"] == ANSWER_ONLY_PROMPTS.system
    assert traj.budget_state.turns_used == 2 and [s.turn for s in traj.steps] == [1, 2]


def test_repeating_the_question_is_not_executed():
    traj, _, tool = run({"gate": "always"}, [call("search", query=Q.upper()), ANSWER])
    assert tool.calls == [(Q, 3)] and traj.escalation.outcome == "duplicate"
    assert traj.budget_state.search_calls_used == 1 and traj.budget_state.search_attempts == 2


def test_agent_probe_can_answer_directly():
    traj, llm, tool = run({"gate": "always", "probe": "agent"}, [ANSWER])
    assert llm.seen_messages[0][0]["content"] == AGENT_PROMPTS.system
    assert [m["role"] for m in llm.seen_messages[0]] == ["system", "user", "assistant", "tool"]
    assert len(llm.seen_messages) == 1 and tool.calls == [(Q, 3)]  # 探测时直接作答：省掉一次作答调用
    assert traj.escalation.outcome == "answered" and traj.final_answer == "Port Edvik"
    assert traj.stop_reason == StopReason.ANSWERED


def test_agent_probe_format_error_falls_back_to_first_hop():
    traj, _, tool = run({"gate": "always", "probe": "agent"}, ["Port Edvik", ANSWER])
    assert traj.escalation.outcome == "format_error" and tool.calls == [(Q, 3)]
    assert traj.final_answer == "Port Edvik"


@pytest.mark.parametrize("scores,probed", [((9.0, 2.0, 1.0), True), ((5.0, 4.0, 1.0), False)])
def test_rerank_gap_gate(scores, probed):
    out = ([call("search", query="Tessa Marrow")] if probed else []) + [ANSWER]
    traj, llm, _ = run({"gate": "rerank_gap", "gap_threshold": 3.0}, out, tool=ScoredTool(scores))
    assert traj.escalation.probed is probed and traj.escalation.feature == pytest.approx(scores[0] - scores[1])
    assert len(llm.seen_messages) == 1 + probed


def test_gap_gate_probes_when_too_few_results():
    obs = Observation(ok=True, docs=[Doc(doc_id="a", title="a", text="a", score=1.0, rank=1, source="x")])
    assert rerank_gap(obs) is None and rerank_gap(Observation(ok=False)) is None
    traj, _, _ = run({"gate": "rerank_gap", "gap_threshold": 3.0}, ["no", ANSWER], tool=ScoredTool([1.0]))
    assert traj.escalation.probed and traj.escalation.feature is None  # 拿不准就探测


def test_retrieval_failure_still_answers():
    traj, _, _ = run({"gate": "always"}, [call("search", query="x"), ANSWER], tool=SpyTool(fail=True))
    assert traj.final_answer == "Port Edvik" and traj.escalation.outcome == "escalated"


def test_config_validation():
    with pytest.raises(ValueError):
        CascadeConfig(gate="rerank_gap")  # 门槛必须写在配置里
    with pytest.raises(ValueError):
        CascadeConfig(gate="sometimes")
    check_budget("cascade", Budget(max_search_calls=2))
    with pytest.raises(ValueError):
        check_budget("cascade", Budget(max_search_calls=3))
    with pytest.raises(ValueError):
        run_method("cascade", "q", Q, ScriptedLLM([]), SpyTool(), BUDGET)
