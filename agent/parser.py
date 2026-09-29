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

from agent.prompts import AGENT_FORMAT_HINT
from agent.schema import TOOL_NAMES, Action, ErrorCode, Observation

# Qwen2.5 原生工具调用格式：标签负责定位，JSON 负责内容
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# 引号没转义时的固定骨架。两个工具都只有一个字符串参数，值只能到最后一个 "}} 为止，边界没有歧义
_ONE_STRING_ARG = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*\{\s*"(\w+)"\s*:\s*"(.*)"\s*\}\s*\}',
                             re.DOTALL)
_ANOTHER_ARG = re.compile(r'"\s*,\s*"\w+"\s*:')  # 值里像是还有第二个参数：不是单参数骨架，不修
_ACTION = TypeAdapter(Action)


@dataclass
class ParseResult:
    action: Optional[Action] = None      # 解析成功时有值
    error: Optional[Observation] = None  # 解析失败时有值，作为本轮观察拼回给模型
    num_tool_calls: int = 0              # 本轮写了几个 tool_call；只执行第一个，个数记进轨迹
    # 下面三个是放宽解析的标记，都记进轨迹，统计每类放宽救回了多少轮
    unclosed: bool = False               # 缺结尾标签
    ignored_suffix: str = ""             # JSON 对象后面被丢掉的内容
    repaired_quotes: bool = False        # 参数值里的引号没转义，按单参数骨架取值


def _fail(code: ErrorCode, what: str, num_tool_calls: int = 0) -> ParseResult:
    return ParseResult(error=Observation(ok=False, error_code=code, message=f"Invalid action: {what}."),
                       num_tool_calls=num_tool_calls)


def parse_action(generated: str, format_hint: str = AGENT_FORMAT_HINT) -> ParseResult:
    """解析规则所有方法都一样；只有报错时附的正确写法随方法可用的工具变（format_hint，见 agent/prompts.py）。"""
    result = _parse(generated)
    if result.error is not None:
        result.error.message = f"{result.error.message} {format_hint}"
    return result


def _parse(generated: str) -> ParseResult:
    blocks = _TOOL_CALL.findall(generated)
    n = len(blocks)

    # 语法层：严格模式，没有 tool_call 的纯文字不当答案；所有方法共用这条规则
    if n == 0:
        if "<tool_call>" not in generated:
            return _fail(ErrorCode.NO_ACTION, "no tool call found")
        # 缺 </tool_call>：取第一个 <tool_call> 之后的内容，照样按"第一个完整 JSON 对象"解析。
        # Qwen2.5-3B 常在 JSON 写完后直接结束（3.3），或跟一段乱码 / 半个新调用（3.5 强制轮 4 次）。
        # 输出被截断时 JSON 不完整，解析不了，仍然报错
        result = _parse_block(generated.split("<tool_call>", 1)[1], 1)
        if result.error is not None and result.error.error_code == ErrorCode.INVALID_JSON:
            return _fail(ErrorCode.NO_ACTION, "found <tool_call> without a closing </tool_call>")
        result.unclosed = True
        return result
    return _parse_block(blocks[0], n)


def _decode(block: str) -> tuple[object, str, bool]:
    """返回 (JSON 值, 被丢掉的后缀, 是否修过引号)；解析不了抛 JSONDecodeError。

    只取第一个完整的 JSON 值，后面的内容丢掉：和停止词的语义一致——模型若写了 </tool_call>，
    停止词本来就会截掉后面的一切，只执行第一个动作；只因为漏了结尾标签就判成没作答，口径不一致。
    """
    text = block.strip()
    try:
        obj, end = json.JSONDecoder().raw_decode(text)
        return obj, text[end:].strip(), False
    except json.JSONDecodeError:
        m = _ONE_STRING_ARG.match(text)
        # 引号修复：模型在参数值里原样写了双引号（validation 上同一题 Static RAG 连续 4 次重试都这样写，重试救不回来）
        if m is None or _ANOTHER_ARG.search(m.group(3)):
            raise
        return {"name": m.group(1), "arguments": {m.group(2): m.group(3)}}, text[m.end():].strip(), True


def _parse_block(block: str, n: int) -> ParseResult:
    # 每轮只执行一个动作，取第一个（与 Search-R1 训练代码一致）
    try:
        obj, suffix, repaired = _decode(block)
    except json.JSONDecodeError as e:
        return _fail(ErrorCode.INVALID_JSON, f"tool call is not valid JSON ({e.msg})", n)

    # 结构层：先单独判断工具名，把"未知工具"和"参数写错"区分开
    if not isinstance(obj, dict) or "name" not in obj:
        return _fail(ErrorCode.INVALID_ARGS, "tool call must be a JSON object with 'name' and 'arguments'", n)
    if obj["name"] not in TOOL_NAMES:
        return _fail(ErrorCode.UNKNOWN_TOOL,
                     f"unknown tool {obj['name']!r}, available tools: {', '.join(TOOL_NAMES)}", n)
    try:
        return ParseResult(action=_ACTION.validate_python(obj), num_tool_calls=n,
                           ignored_suffix=suffix, repaired_quotes=repaired)
    except ValidationError as e:
        err = e.errors()[0]
        field = ".".join(str(x) for x in err["loc"])
        return _fail(ErrorCode.INVALID_ARGS, f"{field}: {err['msg']}", n)
