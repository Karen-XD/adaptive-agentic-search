"""Agent 主循环：生成 → 解析 → 策略检查 → 执行检索 → 观察拼回，直到作答或轮数用完。

四层校验的分工：语法层、结构层在 agent/parser.py；策略层（预算、重复查询）和执行层（检索出错）在这里。
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Protocol

from agent.llm import LLM
from agent.parser import parse_action
from agent.prompts import AGENT_PROMPTS, Prompts, render_context, render_observation
from agent.schema import (Budget, BudgetState, Context, Doc, ErrorCode, FinalAnswerAction, Observation,
                          SearchAction, Step, StopReason, Trajectory)


class SearchTool(Protocol):
    # 只收 query 和 top_k：工具拿不到题目 ID 和标准答案
    def search(self, query: str, top_k: int) -> list[Doc]: ...


def normalize_query(query: str) -> str:
    # 只差大小写、标点、空白的查询检索结果一样（BM25 分词也会丢掉这些），算同一个查询。
    # 不做语义去重：误拦一次正当的改写，代价远大于漏放一次重复查询
    query = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", query).split())


def _error(code: ErrorCode, message: str) -> Observation:
    return Observation(ok=False, error_code=code, message=message)


def run_episode(qid: str, question: str, llm: LLM, tool: SearchTool | None, budget: Budget, *,
                prompts: Prompts = AGENT_PROMPTS, context: Context | None = None,
                method: str = "agent") -> Trajectory:
    """各方法怎么调用它见 agent/methods.py。"""
    user = f"Question: {question}"
    if context is not None:
        user = f"{render_context(context.observation)}\n\n{user}"
    messages = [{"role": "system", "content": prompts.system}, {"role": "user", "content": user}]
    state = BudgetState()
    steps: list[Step] = []
    searched: dict[str, int] = {}  # 归一化后的查询 → 第几轮搜的
    seen_doc_ids: set[str] = set()
    if context is not None:
        seen_doc_ids |= {d.doc_id for d in context.observation.docs}
        if context.query is not None:
            # Static RAG：流程替模型拿原问题搜了一次。成本照算，和 Agent 的搜索次数放在同一把尺子上
            state.search_attempts += 1
            state.search_calls_used += 1
            searched[normalize_query(context.query)] = 0
    budget_hit = False  # 搜索次数用完后模型仍想搜：之后只许作答
    notified = False    # 是否已经告诉过模型"只许作答"
    answer, stop_reason, error = None, StopReason.NO_ANSWER, None

    while state.can_take_turn(budget):
        # 只在"最后一轮"或"模型想超预算搜索"之后强制作答；搜索次数刚用完时不提前强制，
        # 这样 forced_answer 统计的才是"还想搜但被截停"，而不是"本来就打算停"
        forced = budget_hit or state.turns_used == budget.max_turns - 1
        if forced and not notified:
            messages.append({"role": "user", "content": prompts.forced_notice})
            notified = True

        t0 = time.perf_counter()
        try:
            gen = llm.generate(messages)
        except Exception as e:
            # 模型服务重试后仍失败：记成 error，已跑完的轮次照样保留；是否重试由 RetryingLLM 负责
            error, stop_reason = f"{type(e).__name__}: {e}", StopReason.ERROR
            break
        llm_ms = (time.perf_counter() - t0) * 1000
        generated = gen.text
        state.turns_used += 1

        parsed = parse_action(generated)  # 只传本轮新生成的内容
        action, obs = parsed.action, parsed.error
        tool_ms, num_new_docs = 0.0, 0

        if isinstance(action, FinalAnswerAction):
            answer = action.arguments.answer
            stop_reason = StopReason.FORCED_ANSWER if forced else StopReason.ANSWERED
        elif isinstance(action, SearchAction):
            state.search_attempts += 1
            key = normalize_query(action.arguments.query)
            if forced or not state.can_search(budget):
                obs = _error(ErrorCode.BUDGET_EXCEEDED, prompts.forced_notice)
                budget_hit = notified = True  # 报错信息里已经告诉过它了
            elif key in searched:
                # 没有真正执行，不扣搜索次数；但这一轮生成已经花了，照样算轮数（max_turns 兜底防死循环）
                obs = _error(ErrorCode.DUPLICATE_QUERY,
                             f"You already searched this in turn {searched[key]}; the results are above. "
                             "Search with different keywords or call final_answer.")
            else:
                t0 = time.perf_counter()
                try:
                    docs = tool.search(action.arguments.query, budget.top_k)
                    obs = Observation(ok=True, docs=docs)
                    searched[key] = state.turns_used
                    ids = {d.doc_id for d in docs}
                    num_new_docs = len(ids - seen_doc_ids)
                    seen_doc_ids |= ids
                except Exception as e:
                    # 执行层：请求已经发出，算一次实际执行；不记入"已搜过"，允许模型原样重试
                    obs = _error(ErrorCode.TOOL_ERROR,
                                 f"Search failed ({type(e).__name__}). Retry or call final_answer.")
                tool_ms = (time.perf_counter() - t0) * 1000
                state.search_calls_used += 1

        steps.append(Step(turn=state.turns_used, generated=generated, forced=forced,
                          num_tool_calls=parsed.num_tool_calls, unclosed_tool_call=parsed.unclosed,
                          action=action, observation=obs,
                          num_new_docs=num_new_docs, llm_latency_ms=llm_ms, tool_latency_ms=tool_ms,
                          prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                          finish_reason=gen.finish_reason, budget_state=state.model_copy()))
        if answer is not None:
            break
        messages.append({"role": "assistant", "content": generated})
        # role="tool"：Qwen2.5 的对话模板把它渲染成 user 轮次 + <tool_response>…</tool_response>（3.2 已核对）
        messages.append({"role": "tool", "content": render_observation(obs)})

    return Trajectory(qid=qid, question=question, method=method, context=context, budget=budget,
                      steps=steps, budget_state=state,
                      final_answer=answer, stop_reason=stop_reason, error=error)
