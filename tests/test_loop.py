"""Agent 循环：策略层（预算、重复查询、强制作答）、执行层（检索出错）、空结果、防泄漏。

用按剧本输出的假模型，只验证控制逻辑，不验证模型能力（要等 Day 3 接真模型）。
运行：pytest tests/ 或 python -m tests.test_loop
"""
import pytest

from agent.llm import RetryingLLM, ScriptedLLM
from agent.loop import normalize_query, run_episode
from agent.prompts import FORCED_ANSWER_NOTICE
from agent.schema import Budget, ErrorCode, StopReason, Trajectory
from retrieval.mock import MockSearchTool
from tests.helpers import call

Q = "Where was the founder of Luminara Labs born?"


class SpyTool(MockSearchTool):
    """记录每次调用的参数。"""

    def __init__(self):
        super().__init__()
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return super().search(query, top_k)


class FailOnceTool(MockSearchTool):
    """第一次调用超时，之后正常。"""

    def __init__(self):
        super().__init__()
        self.failed = False

    def search(self, query, top_k):
        if not self.failed:
            self.failed = True
            raise TimeoutError("retriever timeout")
        return super().search(query, top_k)


def run(outputs, budget=Budget(), tool=None):
    llm = ScriptedLLM(outputs)
    return run_episode("q-001", Q, llm, tool or MockSearchTool(), budget), llm


def test_two_hop_answered():
    traj, _ = run([call("search", query="Luminara Labs founder"),
                   call("search", query="Tessa Marrow born"),
                   call("final_answer", answer="Port Edvik")])
    assert traj.final_answer == "Port Edvik" and traj.stop_reason == StopReason.ANSWERED
    assert traj.steps[-1].budget_state.search_calls_used == 2
    assert [s.num_new_docs for s in traj.steps[:2]] == [2, 2]  # 第二次的 3 条里 mock-001 已见过


def test_format_error_fed_back_costs_turn_not_search():
    traj, llm = run(["The founder is Tessa Marrow.", call("final_answer", answer="Port Edvik")])
    first = traj.steps[0]
    assert first.observation.error_code == ErrorCode.NO_ACTION
    assert (first.budget_state.turns_used, first.budget_state.search_calls_used) == (1, 0)
    assert llm.seen_messages[1][-1] == {"role": "tool", "content": first.observation.message}
    assert traj.stop_reason == StopReason.ANSWERED


def test_prompt_example_is_never_executed():
    # infer.py 的回归测试：提示词里有 tool_call 示例，输出被截断时也不能误执行它。
    # 第一轮不能是强制作答轮，否则误解析出的搜索会被预算拦下，测不出"误执行"
    spy = SpyTool()
    traj, _ = run(['<tool_call>{"name": "search", "arguments": {"query": "Lumi',
                   call("final_answer", answer="Port Edvik")], tool=spy)
    assert spy.calls == [] and traj.steps[0].observation.error_code == ErrorCode.NO_ACTION


@pytest.mark.parametrize("a, b, same", [
    ("Tessa Marrow", "  tessa   MARROW? ", True),
    ("Ｔｅｓｓａ Marrow", "tessa marrow", True),           # 全角字符
    ("Tessa Marrow", "Tessa Marrow birthplace", False),  # 追加限定词是正当改写，不能拦
])
def test_normalize_query(a, b, same):
    assert (normalize_query(a) == normalize_query(b)) is same


def test_duplicate_query_blocked_without_charging_search():
    spy = SpyTool()
    traj, _ = run([call("search", query="Luminara Labs founder"),
                   call("search", query="  luminara labs FOUNDER? "),
                   call("final_answer", answer="Port Edvik")], tool=spy)
    assert traj.steps[1].observation.error_code == ErrorCode.DUPLICATE_QUERY
    s = traj.steps[1].budget_state
    assert (s.turns_used, s.search_attempts, s.search_calls_used) == (2, 2, 1)
    assert len(spy.calls) == 1


def test_search_over_budget_blocked_then_forced_answer():
    traj, _ = run([call("search", query="Luminara Labs"), call("search", query="Tessa Marrow"),
                   call("final_answer", answer="Port Edvik")], Budget(max_turns=5, max_search_calls=1))
    assert traj.steps[1].observation.error_code == ErrorCode.BUDGET_EXCEEDED
    assert traj.steps[2].forced and traj.stop_reason == StopReason.FORCED_ANSWER
    assert traj.steps[-1].budget_state.search_calls_used == 1


def test_not_forced_right_after_search_budget_runs_out():
    # 搜索次数刚用完、模型本来就打算作答：算 answered，不算 forced_answer
    traj, _ = run([call("search", query="Luminara Labs"), call("final_answer", answer="Port Edvik")],
                  Budget(max_turns=5, max_search_calls=1))
    assert not traj.steps[1].forced and traj.stop_reason == StopReason.ANSWERED


def test_last_turn_forced_and_notified():
    traj, llm = run([call("search", query="Luminara Labs"), call("final_answer", answer="Port Edvik")],
                    Budget(max_turns=2))
    assert traj.steps[1].forced and traj.stop_reason == StopReason.FORCED_ANSWER
    assert llm.seen_messages[1][-1] == {"role": "user", "content": FORCED_ANSWER_NOTICE}


def test_max_turns_without_answer():
    traj, _ = run([call("search", query=q) for q in ["Luminara Labs", "Tessa Marrow", "Port Edvik"]],
                  Budget(max_turns=3, max_search_calls=5))
    assert traj.stop_reason == StopReason.NO_ANSWER and traj.final_answer is None
    assert len(traj.steps) == 3
    assert traj.steps[-1].observation.error_code == ErrorCode.BUDGET_EXCEEDED  # 最后一轮还想搜


def test_empty_results_are_ok_not_error():
    traj, llm = run([call("search", query="qwerty zzz"), call("final_answer", answer="unknown")])
    obs = traj.steps[0].observation
    assert obs.ok and obs.docs == [] and obs.error_code is None
    assert "No results" in llm.seen_messages[1][-1]["content"]


def test_tool_error_counts_as_call_and_allows_retry():
    traj, _ = run([call("search", query="Luminara Labs"), call("search", query="Luminara Labs"),
                   call("final_answer", answer="Tessa Marrow")], tool=FailOnceTool())
    assert traj.steps[0].observation.error_code == ErrorCode.TOOL_ERROR
    assert traj.steps[1].observation.ok  # 原样重试不算重复查询
    assert traj.steps[1].budget_state.search_calls_used == 2


def test_qid_never_reaches_model_or_tool():
    # qid 是查标签的钥匙，不能出现在模型上下文或检索参数里；top_k 来自 Budget
    spy = SpyTool()
    llm = ScriptedLLM([call("search", query="Luminara Labs"), call("final_answer", answer="Port Edvik")])
    run_episode("SECRET-QID-42", Q, llm, spy, Budget(top_k=2))
    assert all("SECRET-QID-42" not in m["content"] for msgs in llm.seen_messages for m in msgs)
    assert spy.calls == [("Luminara Labs", 2)]


class CostLLM(ScriptedLLM):
    """带 token 数的假模型，检查成本是否按轮记进轨迹。"""

    def generate(self, messages):
        gen = super().generate(messages)
        gen.prompt_tokens, gen.completion_tokens, gen.finish_reason = 100 * len(messages), 20, "stop"
        return gen


def test_tokens_recorded_per_step():
    llm = CostLLM([call("search", query="Luminara Labs"), call("final_answer", answer="Port Edvik")])
    traj = run_episode("q-001", Q, llm, MockSearchTool(), Budget())
    # 第二轮上下文多了 assistant + tool 两条：多轮的输入 token 随轮数增长，是 Agent 的主要成本
    assert [s.prompt_tokens for s in traj.steps] == [200, 400]
    assert [s.completion_tokens for s in traj.steps] == [20, 20]
    assert traj.steps[0].finish_reason == "stop"


def test_scripted_llm_has_no_token_counts():
    # 假模型没有 token 数：记 None 而不是 0，免得被统计成"零成本"
    traj, _ = run([call("final_answer", answer="Port Edvik")])
    assert traj.steps[0].prompt_tokens is None


def test_trajectory_json_roundtrip():
    # 1.6 要从 JSONL 重放轨迹：序列化后必须能原样读回
    traj, _ = run([call("search", query="Luminara Labs"), "oops", call("final_answer", answer="Port Edvik")])
    assert Trajectory.model_validate_json(traj.model_dump_json()) == traj



class FlakyLLM:
    """前 n 次调用抛指定异常，之后按剧本输出。"""

    def __init__(self, n_failures, exc, outputs):
        self.n_failures, self.exc, self.calls = n_failures, exc, 0
        self.inner = ScriptedLLM(outputs)

    def generate(self, messages):
        self.calls += 1
        if self.calls <= self.n_failures:
            raise self.exc
        return self.inner.generate(messages)


def test_model_error_recorded_and_steps_kept():
    llm = FlakyLLM(0, None, [call("search", query="Luminara Labs")])  # 第二轮剧本耗尽 → 模型报错
    traj = run_episode("q-001", Q, llm, MockSearchTool(), Budget())
    assert traj.stop_reason == StopReason.ERROR and traj.final_answer is None
    assert len(traj.steps) == 1 and "RuntimeError" in traj.error
    assert traj.budget_state.search_calls_used == 1


def test_retry_transient_errors_only():
    ok = RetryingLLM(FlakyLLM(2, TimeoutError("slow"), ["out"]), max_retries=3, base_delay_s=0)
    assert ok.generate([]).text == "out" and ok.llm.calls == 3 and ok.num_retries == 2
    exhausted = RetryingLLM(FlakyLLM(9, TimeoutError("slow"), ["out"]), max_retries=2, base_delay_s=0)
    with pytest.raises(TimeoutError):
        exhausted.generate([])
    assert exhausted.llm.calls == 3
    permanent = RetryingLLM(FlakyLLM(9, ValueError("context too long"), ["out"]), base_delay_s=0)
    with pytest.raises(ValueError):
        permanent.generate([])
    assert permanent.llm.calls == 1  # 重试也没用的错误不重试

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
