"""查询改写的固定流程（Day 10）：按计划好的顺序检索若干次，每次的查询要么是原问题、要么让模型改写，最后合并证据交给作答。

| 方法 | 检索计划 | 改写看到什么 |
|---|---|---|
| static_rag（B3） | 原问题 | — |
| rewrite_rag | 静态改写 | 只有问题 |
| two_hop_static | 原问题 → 静态改写 | 只有问题 |
| two_hop_evidence | 原问题 → 证据条件改写 | 问题 + 第一次检索（以"自己搜过、收到工具返回"的格式） |

为什么用固定流程、不让模型自己决定搜几次：Day 10 只比"查询写得好不好"。Agent 循环里查询、停止、作答缠在一起，
改了查询停止行为也跟着变（Day 6 见过：改一处，77% 的轨迹都变了）。这里两跳的组每题都恰好搜 2 次，预算完全相同。
改写只给 search 一个工具；解析失败就退回原问题并记 fallback，不重试（重试次数会让各组成本不一样）。
fallback 的检索照样执行、照样计费：固定流程的预算必须相同。代价是这次检索白花（和第一次重复）。

证据条件改写的 fallback 本身是信号（debug 50 题实测）：模型拒绝写第二个查询、直接写出答案的 24 题里，
21 题第一跳已经证据全齐；写出了查询的 26 题里只有 8 题证据全齐 → "看完第一跳还想不想搜"可以当按需升级的信号，
在 Day 11 用（这里不拦截，固定流程要保证两跳的组每题都搜 2 次）。
"""
from __future__ import annotations

import json
import time

from agent.llm import LLM
from agent.loop import SearchTool, normalize_query
from agent.parser import parse_action
from agent.prompts import (EVIDENCE_REWRITE_SYSTEM_PROMPT, REWRITE_FORMAT_HINT, STATIC_REWRITE_SYSTEM_PROMPT,
                           render_observation, render_rewrite_user)
from agent.schema import Context, ErrorCode, Observation, SearchAction, SearchRecord

PLANS = {
    "rewrite_rag": ("static_rewrite",),
    "two_hop_static": ("original", "static_rewrite"),
    "two_hop_evidence": ("original", "evidence_rewrite"),
}


def _merge(observations: list[Observation]) -> Observation:
    """按检索顺序拼接、按 doc_id 去重，名次按展示顺序重排：模型看到的是 [1]..[n]，不会出现两个 [1]。"""
    seen, docs = set(), []
    for obs in observations:
        for d in obs.docs:
            if d.doc_id not in seen:
                seen.add(d.doc_id)
                docs.append(d.model_copy(update={"rank": len(docs) + 1}))
    if not docs and observations and not any(o.ok for o in observations):
        return observations[0]  # 全部检索都失败：保留第一个错误，作答时照常答题（和 Static RAG 口径一致）
    return Observation(ok=True, docs=docs)


def _search(tool: SearchTool, query: str, top_k: int) -> tuple[Observation, float]:
    t0 = time.perf_counter()
    try:
        obs = Observation(ok=True, docs=tool.search(query, top_k))
    except Exception as e:
        obs = Observation(ok=False, error_code=ErrorCode.TOOL_ERROR, message=f"{type(e).__name__}: {e}")
    return obs, (time.perf_counter() - t0) * 1000


def _tool_call_text(query: str) -> str:
    # 和模型自己写的调用逐字同一种写法（Qwen2.5 原生格式，见 agent/native_tools.py）
    return f'<tool_call>\n{json.dumps({"name": "search", "arguments": {"query": query}}, ensure_ascii=False)}\n</tool_call>'


def _rewrite_messages(kind: str, question: str, previous: list[SearchRecord]) -> list[dict]:
    if kind == "static_rewrite":  # 只看问题：不能看到任何检索结果
        return [{"role": "system", "content": STATIC_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": render_rewrite_user(question)}]
    # 证据条件改写：之前的检索写成"模型自己搜过"，和 Agent 循环里看到的上下文同一种格式
    messages = [{"role": "system", "content": EVIDENCE_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"}]
    for r in previous:
        messages += [{"role": "assistant", "content": _tool_call_text(r.query)},
                     {"role": "tool", "content": render_observation(r.observation)}]
    return messages


def _rewrite(llm: LLM, kind: str, question: str, previous: list[SearchRecord]) -> tuple[str | None, dict]:
    t0 = time.perf_counter()
    gen = llm.generate(_rewrite_messages(kind, question, previous))
    info = {"generated": gen.text, "llm_latency_ms": (time.perf_counter() - t0) * 1000,
            "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens}
    action = parse_action(gen.text, REWRITE_FORMAT_HINT).action
    return (action.arguments.query if isinstance(action, SearchAction) else None), info


def retrieve_with_plan(method: str, question: str, tool: SearchTool, llm: LLM, top_k: int) -> Context:
    t0 = time.perf_counter()
    records: list[SearchRecord] = []
    seen: set[str] = set()
    for kind in PLANS[method]:
        info: dict = {}
        query, fallback = question, False
        if kind != "original":
            rewritten, info = _rewrite(llm, kind, question, records)
            if rewritten is None:
                fallback = True
            else:
                query = rewritten
        obs, tool_ms = _search(tool, query, top_k)
        ids = {d.doc_id for d in obs.docs}
        records.append(SearchRecord(kind=kind, query=query, fallback=fallback, observation=obs,
                                    num_new_docs=len(ids - seen), tool_latency_ms=tool_ms, **info))
        seen |= ids
    return Context(source="retrieval", query=records[0].query, observation=_merge([r.observation for r in records]),
                   latency_ms=(time.perf_counter() - t0) * 1000, searches=records)


def is_repeat(record: SearchRecord, earlier: list[SearchRecord]) -> bool:
    """改写出来的查询和之前某次检索只差大小写 / 标点：白花了一次检索。只统计，不拦截（固定流程要保证预算相同）。"""
    return any(normalize_query(record.query) == normalize_query(r.query) for r in earlier)
