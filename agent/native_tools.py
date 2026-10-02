"""Qwen2.5 原生工具说明：`# Tools` + `<tools>` 函数签名 + `<tool_call>` 格式说明。

2026-09-28 实测：自己写的列表式工具说明格式有效率 0/8，这份写法 8/8。
Instruct 模型的工具调用能力是用固定模板微调出来的，偏离模板它就只会模仿提示词的字面样子。

为什么在客户端用代码生成，而不是请求时把 tools 参数交给服务端去套模板：
1. 模型看到的每个字都在我们的代码里、进 git，换 vLLM / transformers 版本不会悄悄改变实验输入。
2. 我们用自己的解析器处理输出（四层校验、错误码），不依赖服务端的工具调用解析。
代价：换模型要重新核对。`tests/test_prompt_native.py` 用模型目录里的对话模板重新生成，逐字比对。
"""
from __future__ import annotations

import json

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search",
    "description": "Search a document collection with keywords. Returns the top matching passages.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "Search keywords"}},
                   "required": ["query"]}}}

FINAL_ANSWER_TOOL = {"type": "function", "function": {
    "name": "final_answer",
    "description": "Give the final answer and end. The answer should be a short phrase such as a name, "
                   "date or number, not a sentence.",
    "parameters": {"type": "object",
                   "properties": {"answer": {"type": "string", "description": "Short final answer"}},
                   "required": ["answer"]}}}


# 静态拆解（Day 10.4）：一次写出多个子查询。参数是列表，所以单独做一个工具，而不是让模型一轮写多个 search：
# 生成在第一个 </tool_call> 处就停（每轮一个动作），一轮写不出多个调用
DECOMPOSE_TOOL = {"type": "function", "function": {
    "name": "decompose",
    "description": "Split the question into simpler sub-questions and search for each of them. "
                   "Returns the top matching passages for every sub-question.",
    "parameters": {"type": "object",
                   "properties": {"subqueries": {"type": "array", "items": {"type": "string"},
                                                 "description": "One search query per sub-question"}},
                   "required": ["subqueries"]}}}


def native_system_prompt(task: str, tools: list[dict]) -> str:
    """复现 Qwen2.5 对话模板在传入 tools 时生成的 system 内容（模板里 tojson = json.dumps(ensure_ascii=False)）。"""
    signatures = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
    return (f"{task}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>{signatures}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
