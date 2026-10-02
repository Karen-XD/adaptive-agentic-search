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

# 格式出错时附在报错里的正确写法。先给作答：纯文字写出答案、忘了调用是最常见的格式错误，
# 旧提示先举 search 的例子，模型就又去搜了一遍（3.5 在 debug 上数到 4 次，其中一题因此没答上）
AGENT_FORMAT_HINT = ("Use exactly one tool call per turn. If you know the answer, call "
                     '<tool_call>{"name": "final_answer", "arguments": {"answer": "..."}}</tool_call>; '
                     'if you need more information, call '
                     '<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>')

# 只能作答（B0 Direct / B1 Static RAG / Oracle 共用）：同一种原生写法，只是不给 search 工具。
# 三者只差用户消息里有没有、有什么证据，提示词一字不差，比较的才是"证据"这一个变量
ANSWER_ONLY_TASK = ("Answer the question. If documents are provided, base your answer on them. "
                    "Call final_answer with your answer.")
ANSWER_ONLY_SYSTEM_PROMPT = native_system_prompt(ANSWER_ONLY_TASK, [FINAL_ANSWER_TOOL])
ANSWER_ONLY_FORCED_NOTICE = "Call final_answer now with your best answer."
# 这三种方法没有 search 工具，报错里不能举 search 的例子
ANSWER_ONLY_FORMAT_HINT = ('Use exactly one tool call per turn: '
                           '<tool_call>{"name": "final_answer", "arguments": {"answer": "..."}}</tool_call>')


# 查询改写（Day 10）：只给 search 一个工具，模型只能写查询。改写和作答分开调用，作答仍用 ANSWER_ONLY_PROMPTS，
# 这样各组只差"证据从哪些查询来"，作答提示词一字不差
STATIC_REWRITE_TASK = ("Write one search query for a Wikipedia paragraph search engine. The query should retrieve "
                       "the paragraph that contains the information needed to answer the question. "
                       "Use the key names and terms in the question. Call search exactly once.")
# 证据条件改写用 Agent 的对话格式：之前的检索写成模型自己发出的 search 调用，结果以工具返回（<tool_response>）给它。
# debug 上试过把段落放进用户消息（render_rewrite_user 的 evidence 分支）：50 题 0 次用上证据里的新实体、0 次先写推理，
# 几乎都在复述原问题；同一个模型在 Agent 循环里第二次搜索前 76% 会先写推理、35% 用上证据里的实体 → 顺着微调格式
EVIDENCE_REWRITE_TASK = ("Answer the question by searching a document collection. Call exactly one function per turn. "
                         "The information found so far is not enough: search again with different keywords "
                         "for what is still missing.")
STATIC_REWRITE_SYSTEM_PROMPT = native_system_prompt(STATIC_REWRITE_TASK, [SEARCH_TOOL])
EVIDENCE_REWRITE_SYSTEM_PROMPT = native_system_prompt(EVIDENCE_REWRITE_TASK, [SEARCH_TOOL])
REWRITE_FORMAT_HINT = 'Use exactly one tool call: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>'


@dataclass(frozen=True)
class Prompts:
    system: str
    forced_notice: str
    format_hint: str


AGENT_PROMPTS = Prompts(SYSTEM_PROMPT, FORCED_ANSWER_NOTICE, AGENT_FORMAT_HINT)
ANSWER_ONLY_PROMPTS = Prompts(ANSWER_ONLY_SYSTEM_PROMPT, ANSWER_ONLY_FORCED_NOTICE, ANSWER_ONLY_FORMAT_HINT)


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


def render_rewrite_user(question: str) -> str:
    """静态改写的用户消息：只给问题。证据条件改写的上下文见 agent/rewrite.py（Agent 的对话格式）。"""
    return f"Question: {question}"
