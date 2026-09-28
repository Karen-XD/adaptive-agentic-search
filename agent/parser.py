"""把模型本轮生成的文字解析成 Action，负责四层校验里的语法层和结构层。

调用方只传入本轮新生成的内容，不要带上 prompt 和历史轮次：
infer.py 在整段文本上解析，截断时会静默拿到旧查询或 prompt 里的示例。
策略层（重复查询、预算）由循环负责，执行层（工具出错）由检索网关负责。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import TypeAdapter, ValidationError

from agent.schema import TOOL_NAMES, Action, ErrorCode, Observation

# Qwen2.5 原生工具调用格式：标签负责定位，JSON 负责内容
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# 结尾标签缺失：最后一个 <tool_call> 之后直到输出结束的全部内容
_UNCLOSED_TAIL = re.compile(r"<tool_call>((?:(?!<tool_call>).)*)\Z", re.DOTALL)
_ACTION = TypeAdapter(Action)

# 出错时附在观察里给模型看，告诉它正确写法（提示词是英文，这里也用英文）
_FORMAT_HINT = (
    "Use exactly one tool call per turn, e.g. "
    '<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call> or '
    '<tool_call>{"name": "final_answer", "arguments": {"answer": "..."}}</tool_call>'
)


@dataclass
class ParseResult:
    action: Optional[Action] = None      # 解析成功时有值
    error: Optional[Observation] = None  # 解析失败时有值，作为本轮观察拼回给模型
    num_tool_calls: int = 0              # 本轮写了几个 tool_call；只执行第一个，个数记进轨迹
    unclosed: bool = False               # 缺结尾标签但按下面的规则补上了；记进轨迹，统计这类输出有多少


def _fail(code: ErrorCode, what: str, num_tool_calls: int = 0) -> ParseResult:
    message = f"Invalid action: {what}. {_FORMAT_HINT}"
    return ParseResult(error=Observation(ok=False, error_code=code, message=message),
                       num_tool_calls=num_tool_calls)


def parse_action(generated: str) -> ParseResult:
    blocks = _TOOL_CALL.findall(generated)
    n = len(blocks)

    # 语法层：严格模式，没有 tool_call 的纯文字不当答案；所有方法共用这条规则
    if n == 0:
        if "<tool_call>" in generated:
            # 唯一的放宽：缺 </tool_call>，但标签后面直到结尾恰好是一个完整的 JSON 对象，意图没有歧义，照常解析。
            # Qwen2.5-3B 先写推理文字、再写调用时，常在 JSON 写完后直接结束（3.3 实测 5 个格式错误里 4 个是这样）。
            # 输出被截断时 JSON 不完整，解析不了，仍然报错
            tail = _UNCLOSED_TAIL.search(generated)
            if tail and _is_json_object(tail.group(1)):
                result = _parse_block(tail.group(1), 1)
                result.unclosed = True
                return result
            return _fail(ErrorCode.NO_ACTION, "found <tool_call> without a closing </tool_call>")
        return _fail(ErrorCode.NO_ACTION, "no tool call found")
    return _parse_block(blocks[0], n)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def _parse_block(block: str, n: int) -> ParseResult:
    # 每轮只执行一个动作，取第一个（与 Search-R1 训练代码一致）
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as e:
        return _fail(ErrorCode.INVALID_JSON, f"tool call is not valid JSON ({e.msg})", n)

    # 结构层：先单独判断工具名，把"未知工具"和"参数写错"区分开
    if not isinstance(obj, dict) or "name" not in obj:
        return _fail(ErrorCode.INVALID_ARGS, "tool call must be a JSON object with 'name' and 'arguments'", n)
    if obj["name"] not in TOOL_NAMES:
        return _fail(ErrorCode.UNKNOWN_TOOL,
                     f"unknown tool {obj['name']!r}, available tools: {', '.join(TOOL_NAMES)}", n)
    try:
        return ParseResult(action=_ACTION.validate_python(obj), num_tool_calls=n)
    except ValidationError as e:
        err = e.errors()[0]
        field = ".".join(str(x) for x in err["loc"])
        return _fail(ErrorCode.INVALID_ARGS, f"{field}: {err['msg']}", n)
