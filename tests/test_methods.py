"""四种方法：提示词、证据、成本口径、预算校验；Oracle 只许在 validation / debug 上跑。

用按剧本输出的假模型，只验证"每种方法喂给模型的是什么、成本怎么记"，不验证模型能力。
运行：pytest tests/ 或 python -m tests.test_methods
"""
import json

import pytest

from agent.llm import ScriptedLLM
from agent.methods import check_budget, run_method
from agent.prompts import AGENT_PROMPTS, ANSWER_ONLY_PROMPTS
from agent.schema import Budget, Doc, ErrorCode, StopReason
from evaluation.oracle import load_gold_docs
from evaluation.run_eval import load_config
from retrieval.mock import MockSearchTool
from tests.helpers import call

Q = "Where was the founder of Luminara Labs born?"
ANSWER = call("final_answer", answer="Port Edvik")


class SpyTool(MockSearchTool):
    def __init__(self, fail=False):
        super().__init__()
        self.calls, self.fail = [], fail

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        if self.fail:
            raise TimeoutError("retriever timeout")
        return super().search(query, top_k)


def run(method, outputs=(ANSWER,), budget=None, tool=None, gold_docs=None):
    budget = budget or Budget(max_search_calls={"direct": 0, "static_rag": 1, "oracle": 0}.get(method, 3))
    llm = ScriptedLLM(list(outputs))
    return run_method(method, "q-001", Q, llm, tool, budget, gold_docs), llm


def test_direct_sees_no_documents_and_no_search_tool():
    traj, llm = run("direct")
    system, user = llm.seen_messages[0]
    assert system["content"] == ANSWER_ONLY_PROMPTS.system and '"name": "search"' not in system["content"]
    assert user["content"] == f"Question: {Q}"
    assert traj.final_answer == "Port Edvik" and traj.budget_state.search_calls_used == 0 and traj.context is None


def test_static_rag_searches_the_original_question_once():
    tool = SpyTool()
    traj, llm = run("static_rag", tool=tool)
    assert tool.calls == [(Q, 3)]  # 原问题、top_k 由 Budget 定
    user = llm.seen_messages[0][1]["content"]
    assert user.startswith("Documents:\n[1] (Title: ") and user.endswith(f"Question: {Q}")
    # 流程替模型搜的那一次照算成本，和 Agent 的搜索次数可比
    assert traj.budget_state.search_calls_used == 1 and traj.budget_state.search_attempts == 1
    assert traj.context.query == Q and len(traj.context.observation.docs) == 3


def test_static_rag_model_cannot_search_more():
    # 就算模型想再搜（工具说明里没有 search，但它可能照写），也会被预算拦下，不会真的检索
    tool = SpyTool()
    traj, _ = run("static_rag", outputs=[call("search", query="Tessa Marrow"), ANSWER], tool=tool)
    assert len(tool.calls) == 1
    assert traj.steps[0].observation.error_code == ErrorCode.BUDGET_EXCEEDED
    assert traj.steps[0].observation.message == ANSWER_ONLY_PROMPTS.forced_notice
    assert traj.stop_reason == StopReason.FORCED_ANSWER


def test_static_rag_retrieval_failure_still_answers():
    traj, llm = run("static_rag", tool=SpyTool(fail=True))
    assert "Documents:\n(none)" in llm.seen_messages[0][1]["content"]
    assert traj.context.observation.error_code == ErrorCode.TOOL_ERROR
    assert traj.final_answer == "Port Edvik" and traj.budget_state.search_calls_used == 1


def test_oracle_gets_gold_docs_and_no_search_cost():
    gold = [Doc(doc_id="d1", title="Tessa Marrow", text="Born in Port Edvik.", score=0.0, rank=1, source="oracle")]
    traj, llm = run("oracle", gold_docs=gold)
    assert "[1] (Title: Tessa Marrow) Born in Port Edvik." in llm.seen_messages[0][1]["content"]
    assert traj.budget_state.search_calls_used == 0 and traj.context.source == "oracle"


def test_answer_only_methods_share_one_prompt():
    # B0 / B1 / Oracle 提示词一字不差，差别只在用户消息里的证据
    systems = set()
    for method in ("direct", "static_rag", "oracle"):
        gold = [Doc(doc_id="d", title="t", text="x", score=0.0, rank=1, source="oracle")]
        _, llm = run(method, tool=SpyTool(), gold_docs=gold)
        systems.add(llm.seen_messages[0][0]["content"])
    assert systems == {ANSWER_ONLY_PROMPTS.system}


def test_agent_path_unchanged():
    tool = SpyTool()
    traj, llm = run("agent", outputs=[call("search", query="Luminara Labs"), ANSWER], tool=tool)
    assert llm.seen_messages[0][0]["content"] == AGENT_PROMPTS.system
    assert llm.seen_messages[0][1]["content"] == f"Question: {Q}"
    assert tool.calls == [("Luminara Labs", 3)] and traj.method == "agent"


@pytest.mark.parametrize("method, max_search_calls", [
    ("direct", 1), ("static_rag", 0), ("static_rag", 3), ("oracle", 1), ("agent", 0), ("browse", 3),
])
def test_budget_must_match_method(method, max_search_calls):
    with pytest.raises(ValueError):
        check_budget(method, Budget(max_search_calls=max_search_calls))


def _write_split(tmp_path, split):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / f"{split}.jsonl").write_text(
        json.dumps({"qid": "q1", "answer": "a", "gold_doc_ids": ["hp-b", "hp-a"]}) + "\n", encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text("".join(
        json.dumps({"doc_id": d, "title": d.upper(), "text": f"text {d}"}) + "\n" for d in ("hp-a", "hp-b", "hp-c")),
        encoding="utf-8")


def test_oracle_loader_sorts_by_doc_id(tmp_path):
    _write_split(tmp_path, "validation")
    docs = load_gold_docs(tmp_path, "validation")["q1"]
    # 标注顺序（常是第一跳在前）会泄漏推理路径，按 doc_id 排
    assert [(d.doc_id, d.rank) for d in docs] == [("hp-a", 1), ("hp-b", 2)]


def test_oracle_refuses_test_split(tmp_path):
    _write_split(tmp_path, "test")
    with pytest.raises(ValueError, match="diagnostic"):
        load_gold_docs(tmp_path, "test")


def test_config_extends_overrides_only_given_fields(tmp_path):
    (tmp_path / "base.yaml").write_text("name: base\nllm: {type: vllm, sampling: {temperature: 0.0, seed: 0}}\n"
                                        "budget: {max_turns: 5, max_search_calls: 3}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nname: child\nbudget: {max_search_calls: 0}\n",
                                         encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["name"] == "child" and cfg["budget"] == {"max_turns": 5, "max_search_calls": 0}
    assert cfg["llm"]["sampling"] == {"temperature": 0.0, "seed": 0}  # 解码参数只写在基础配置里


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
