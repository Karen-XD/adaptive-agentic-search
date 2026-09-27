"""Action Schema、Budget 与解析器：语法层和结构层的错误要落到正确的错误码。

运行：pytest tests/ 或 python -m tests.test_action_schema
"""
import pytest
from pydantic import ValidationError

from agent.parser import parse_action
from agent.schema import Budget, ErrorCode
from tests.helpers import call


@pytest.mark.parametrize("text, name, args", [
    (call("search", query="Luminara Labs founder"), "search", {"query": "Luminara Labs founder"}),
    ("Let me answer.\n" + call("final_answer", answer=" Port Edvik "), "final_answer", {"answer": "Port Edvik"}),
])
def test_valid_actions(text, name, args):
    r = parse_action(text)
    assert r.error is None and r.num_tool_calls == 1
    assert r.action.name == name and r.action.arguments.model_dump() == args


@pytest.mark.parametrize("text, code", [
    ("The answer is Port Edvik.", ErrorCode.NO_ACTION),  # 严格模式：纯文字不当答案
    ('<tool_call>{"name": "search", "arguments": {"query": "Lumi', ErrorCode.NO_ACTION),  # 输出被截断
    ('<tool_call>{"name": "search", "arguments": {"query": "x"},}</tool_call>', ErrorCode.INVALID_JSON),
    ('<tool_call>```json\n{"name": "search", "arguments": {"query": "x"}}\n```</tool_call>', ErrorCode.INVALID_JSON),
    ('<tool_call>["search", "x"]</tool_call>', ErrorCode.INVALID_ARGS),
    ('<tool_call>{"name": "search"}</tool_call>', ErrorCode.INVALID_ARGS),
    (call("browse", url="x"), ErrorCode.UNKNOWN_TOOL),
    (call("search", query="   "), ErrorCode.INVALID_ARGS),          # 空查询
    (call("search", query="x", top_k=10), ErrorCode.INVALID_ARGS),  # top_k 不交给模型
    (call("final_answer"), ErrorCode.INVALID_ARGS),
])
def test_invalid_actions(text, code):
    r = parse_action(text)
    assert r.action is None and not r.error.ok and r.error.error_code == code
    assert "<tool_call>" in r.error.message  # 报错里附正确写法，模型下一轮才知道怎么改


def test_only_first_of_multiple_tool_calls_is_used():
    r = parse_action(call("search", query="a") + call("search", query="b"))
    assert r.action.arguments.query == "a" and r.num_tool_calls == 2


def test_budget_is_validated_and_frozen():
    with pytest.raises(ValidationError):
        Budget(max_turns=0)
    b = Budget()
    with pytest.raises(ValidationError):
        b.max_turns = 100


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
