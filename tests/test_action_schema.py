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


@pytest.mark.parametrize("text", [
    # 3.3 真模型实测：先写推理文字，JSON 写完就结束，漏了结尾标签
    'The answer is X.\n<tool_call>\n{"name": "final_answer", "arguments": {"answer": "X"}}',
    '<tool_call>{"name": "final_answer", "arguments": {"answer": "X"}}\n',
])
def test_unclosed_but_complete_tool_call_is_accepted(text):
    r = parse_action(text)
    assert r.action.name == "final_answer" and r.action.arguments.answer == "X"
    assert r.unclosed and r.num_tool_calls == 1 and r.ignored_suffix == ""


@pytest.mark.parametrize("text, suffix, unclosed", [
    # 3.5 validation 强制轮实测：JSON 写完后跟乱码 / 半个新调用，没写结尾标签
    ('<tool_call>\n{"name": "final_answer", "arguments": {"answer": "X"}} hendrix', "hendrix", True),
    ('<tool_call>\n{"name": "final_answer", "arguments": {"answer": "X"}}[]\n<tool_call>', "[]\n<tool_call>", True),
    # 闭合的块里同理：只取第一个完整 JSON
    ('<tool_call>{"name": "final_answer", "arguments": {"answer": "X"}} I think</tool_call>', "I think", False),
])
def test_first_complete_json_wins_and_suffix_is_recorded(text, suffix, unclosed):
    r = parse_action(text)
    assert r.action.name == "final_answer" and r.action.arguments.answer == "X"
    assert r.ignored_suffix == suffix and r.unclosed == unclosed


@pytest.mark.parametrize("text, code", [
    ('<tool_call>{"name": "final_answer", "arguments": {"answer": "X"', ErrorCode.NO_ACTION),       # 截断
    ('<tool_call>{"name": "browse", "arguments": {}}', ErrorCode.UNKNOWN_TOOL),  # 补上标签后照常走结构层校验
    ('<tool_call> I will search for it', ErrorCode.NO_ACTION),                   # 标签后面不是 JSON
    # 3.5 实测：} 写成 )，JSON 不完整，不修（模型重试一次就能改对）
    ('<tool_call>{"name": "final_answer", "arguments": {"answer": "Columbus"})</tool_call>', ErrorCode.INVALID_JSON),
])
def test_unclosed_repair_is_narrow(text, code):
    r = parse_action(text)
    assert r.action is None and r.error.error_code == code


@pytest.mark.parametrize("text, name, value", [
    # 3.5 validation 实测：答案里的引号没转义，同一题重试 4 次都这样写
    ('<tool_call>{"name": "final_answer", "arguments": {"answer": "an angel who is "effective in love""}}</tool_call>',
     "final_answer", 'an angel who is "effective in love"'),
    ('<tool_call>{"name": "search", "arguments": {"query": "the film "Zazel" 1995"}}', "search", 'the film "Zazel" 1995'),
])
def test_unescaped_quotes_repaired_for_single_string_arg(text, name, value):
    r = parse_action(text)
    assert r.action.name == name and list(r.action.arguments.model_dump().values()) == [value]
    assert r.repaired_quotes


@pytest.mark.parametrize("text, code", [
    # 值里像有第二个参数：不是单参数骨架，不修
    ('<tool_call>{"name": "search", "arguments": {"query": "a", "top_k": "b "c""}}</tool_call>', ErrorCode.INVALID_JSON),
    # 修出来的动作照常走结构层：参数名不对仍然报错
    ('<tool_call>{"name": "final_answer", "arguments": {"reply": "say "hi""}}</tool_call>', ErrorCode.INVALID_ARGS),
])
def test_quote_repair_is_narrow(text, code):
    r = parse_action(text)
    assert r.action is None and r.error.error_code == code


def test_closed_call_wins_over_unclosed_tail():
    r = parse_action(call("search", query="a") + '<tool_call>{"name": "final_answer", "arguments": {"answer": "b"}}')
    assert r.action.arguments.query == "a" and not r.unclosed


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
