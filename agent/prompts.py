"""提示词和观察的渲染。当前是占位版本，Day 3 起只在 validation 上调。

提示词里的 tool_call 示例不会被误解析：解析器只看模型本轮新生成的内容（对比 infer.py 的问题）。
"""
from __future__ import annotations

from agent.schema import Observation

SYSTEM_PROMPT = """Answer the question by searching a document collection.

In every turn, think briefly, then call exactly one tool:
- Search: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
- Answer: <tool_call>{"name": "final_answer", "arguments": {"answer": "..."}}</tool_call>

Search results are shown after each search. Search again if the information is not enough.
The final answer should be a short phrase such as a name, date or number, not a sentence."""

# 不训练的模型没法从奖励里学会"最后一轮不能搜"（Search-R1 靠 RL 学），只能明确告诉它
FORCED_ANSWER_NOTICE = ("No more searches are allowed. Call final_answer now "
                        "with your best answer based on the information above.")


def render_observation(obs: Observation) -> str:
    if not obs.ok:
        return obs.message
    if not obs.docs:
        # 检索成功但没结果不算错误，但要让模型知道，它才会换关键词
        return "No results found. Try different keywords."
    return "\n".join(f"[{d.rank}] (Title: {d.title}) {d.text}" for d in obs.docs)
