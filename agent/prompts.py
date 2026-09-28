"""提示词和观察的渲染。只在 validation 上调。

系统提示词 = 任务说明 + Qwen2.5 原生工具说明（见 agent/native_tools.py：为什么用原生写法、为什么在代码里生成）。
提示词里的 tool_call 格式说明不会被误解析：解析器只看模型本轮新生成的内容（对比 infer.py 的问题）。
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.native_tools import FINAL_ANSWER_TOOL, SEARCH_TOOL, native_system_prompt
from agent.schema import Observation

TASK = ("Answer the question by searching a document collection. Call exactly one function per turn. "
        "Search again with different keywords if the information is not enough; call final_answer when you can answer.")
TOOLS = [SEARCH_TOOL, FINAL_ANSWER_TOOL]
SYSTEM_PROMPT = native_system_prompt(TASK, TOOLS)

# 不训练的模型没法从奖励里学会"最后一轮不能搜"（Search-R1 靠 RL 学），只能明确告诉它
FORCED_ANSWER_NOTICE = ("No more searches are allowed. Call final_answer now "
                        "with your best answer based on the information above.")

# 只能作答（B0 Direct / B1 Static RAG / Oracle 共用）：同一种原生写法，只是不给 search 工具。
# 三者只差用户消息里有没有、有什么证据，提示词一字不差，比较的才是"证据"这一个变量
ANSWER_ONLY_TASK = ("Answer the question. If documents are provided, base your answer on them. "
                    "Call final_answer with your answer.")
ANSWER_ONLY_SYSTEM_PROMPT = native_system_prompt(ANSWER_ONLY_TASK, [FINAL_ANSWER_TOOL])
ANSWER_ONLY_FORCED_NOTICE = "Call final_answer now with your best answer."


@dataclass(frozen=True)
class Prompts:
    system: str
    forced_notice: str


AGENT_PROMPTS = Prompts(SYSTEM_PROMPT, FORCED_ANSWER_NOTICE)
ANSWER_ONLY_PROMPTS = Prompts(ANSWER_ONLY_SYSTEM_PROMPT, ANSWER_ONLY_FORCED_NOTICE)


def render_observation(obs: Observation) -> str:
    if not obs.ok:
        return obs.message
    if not obs.docs:
        # 检索成功但没结果不算错误，但要让模型知道，它才会换关键词
        return "No results found. Try different keywords."
    return "\n".join(f"[{d.rank}] (Title: {d.title}) {d.text}" for d in obs.docs)


def render_context(obs: Observation) -> str:
    """答题前给的证据，放在问题前面。段落格式和 Agent 看到的检索结果一致。"""
    docs = render_observation(obs) if obs.ok and obs.docs else "(none)"  # 检索失败也照常答题，只是没有证据
    return f"Documents:\n{docs}"
