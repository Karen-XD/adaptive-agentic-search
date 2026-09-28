"""四种方法：共用同一个循环、同一个解析器、同一个 Budget 定义，只在"证据从哪来""模型能不能搜"上不同。

| 方法 | 答题前给的证据 | 模型能搜 | 工具 | Budget.max_search_calls |
|---|---|---|---|---|
| agent（B2 起） | 无 | 能 | search + final_answer | ≥ 1 |
| direct（B0） | 无 | 不能 | final_answer | 0 |
| static_rag（B1） | 原问题检索 top_k 条 | 不能 | final_answer | 1（流程替模型用掉） |
| oracle（诊断上限） | 金标段落 | 不能 | final_answer | 0（不算检索成本） |

direct / static_rag / oracle 的提示词一字不差，只差用户消息里的证据 → 三者之差就是"证据"带来的差别：
B0 → B1 是一次检索的收益，B1 → Oracle 是检索还差多少，Oracle → 100% 是阅读（和标签噪声）的损失。

Oracle 要把金标段落放进提示词，是防泄漏规则的唯一例外（CLAUDE.md）：金标由评测侧读取后传进来，
这里不碰答案文件；只许在 validation / debug 上跑，见 evaluation/oracle.py。
"""
from __future__ import annotations

import time

from agent.llm import LLM
from agent.loop import SearchTool, run_episode
from agent.prompts import ANSWER_ONLY_PROMPTS
from agent.schema import Budget, Context, Doc, ErrorCode, Observation, Trajectory

METHODS = ("agent", "direct", "static_rag", "oracle")
NEEDS_RETRIEVAL = {"agent", "static_rag"}


def check_budget(method: str, budget: Budget) -> None:
    """搜索次数写在配置里，和方法对不上就拒绝运行：成本统计按实际执行的检索算，配错了表就不可比。"""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, choose from {METHODS}")
    expected = {"direct": 0, "static_rag": 1, "oracle": 0}.get(method)
    if expected is not None and budget.max_search_calls != expected:
        raise ValueError(f"{method} needs budget.max_search_calls={expected}, got {budget.max_search_calls}")
    if method == "agent" and budget.max_search_calls < 1:
        raise ValueError("agent needs budget.max_search_calls >= 1; use method=direct for no search")


def retrieve_context(question: str, tool: SearchTool, top_k: int) -> Context:
    """Static RAG：拿原问题搜一次。检索出错不中断，记下错误、没有证据照常答题（和 Agent 的执行层口径一致）。"""
    t0 = time.perf_counter()
    try:
        obs = Observation(ok=True, docs=tool.search(question, top_k))
    except Exception as e:
        obs = Observation(ok=False, error_code=ErrorCode.TOOL_ERROR, message=f"{type(e).__name__}: {e}")
    return Context(source="retrieval", query=question, observation=obs,
                   latency_ms=(time.perf_counter() - t0) * 1000)


def oracle_context(gold_docs: list[Doc]) -> Context:
    return Context(source="oracle", observation=Observation(ok=True, docs=gold_docs))


def run_method(method: str, qid: str, question: str, llm: LLM, tool: SearchTool | None, budget: Budget,
               gold_docs: list[Doc] | None = None) -> Trajectory:
    if method == "agent":
        return run_episode(qid, question, llm, tool, budget)
    if method == "direct":
        return run_episode(qid, question, llm, None, budget, prompts=ANSWER_ONLY_PROMPTS, method=method)
    if method == "static_rag":
        context = retrieve_context(question, tool, budget.top_k)
    elif method == "oracle":
        if gold_docs is None:
            raise ValueError("oracle needs gold_docs")
        context = oracle_context(gold_docs)
    else:
        raise ValueError(f"unknown method {method!r}")
    # tool 传 None：这三种方法的模型搜不了（max_search_calls 已用完或为 0），就算它想搜也会先被预算拦下
    return run_episode(qid, question, llm, None, budget, prompts=ANSWER_ONLY_PROMPTS, context=context,
                       method=method)
