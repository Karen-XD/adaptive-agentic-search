"""提示词和观察的渲染。只在 validation 上调。

系统提示词 = 任务说明 + Qwen2.5 原生工具说明（见 agent/native_tools.py：为什么用原生写法、为什么在代码里生成）。
提示词里的 tool_call 格式说明不会被误解析：解析器只看模型本轮新生成的内容（对比 infer.py 的问题）。
"""
from __future__ import annotations

from agent.native_tools import FINAL_ANSWER_TOOL, SEARCH_TOOL, native_system_prompt
from agent.schema import Observation

TASK = ("Answer the question by searching a document collection. Call exactly one function per turn. "
        "Search again with different keywords if the information is not enough; call final_answer when you can answer.")
TOOLS = [SEARCH_TOOL, FINAL_ANSWER_TOOL]
SYSTEM_PROMPT = native_system_prompt(TASK, TOOLS)

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
