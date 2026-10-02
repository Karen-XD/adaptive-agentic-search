"""按需升级（级联，Day 11）：先走 B3（原问题搜一次 + 重排），再决定这道题要不要升级成两跳。

流程：
  1. 原问题检索一次（和 B3 完全相同）
  2. 门控（gate）：要不要花一次模型调用去"探测"
       never       从不探测 = B3
       always      每题都探测
       rerank_gap  重排第 1 名和第 2 名的分差 > 门槛才探测（不调模型，几乎零成本）
  3. 探测（probe）：把第一跳结果交给模型，看它想不想再搜
       rewrite  Day 10 的证据改写提示词（只有 search 工具）；不写查询 = 拒绝再搜
       agent    Agent 提示词（search + final_answer）；调用 final_answer = 拒绝再搜，而且这个答案直接用，省一次作答调用
  4. 想再搜：用模型写的查询搜第二次，用两跳证据作答；不想：用第一跳证据作答（= B3 的答案）

作答步和 B3 / 两跳证据改写用同一份提示词。所以 gate=never 逐字等于 B3，gate=always + probe=rewrite 的答案逐字等于
two_hop_evidence（拒绝再搜时两跳那组退回原问题又搜一遍，证据和第一跳相同）。贪心解码下，门控类策略的结果可以用
B3 和 two_hop_evidence 两次运行离线精确推算（Day 11 先离线画了质量–成本曲线，再在线跑验证）。

为什么门控用"重排分差"：分差大 = 只有一段和问题强相关，另一段金标多半是问题里没点名的桥接实体，单次检索够不着；
分差小 = 两段都和问题相关（比如比较题两个实体各一段），证据多半已经齐了。离线 AUC 约 0.74（两个数据集都是）。
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from agent.llm import LLM
from agent.loop import SearchTool, normalize_query, run_episode
from agent.parser import parse_action
from agent.prompts import AGENT_PROMPTS, ANSWER_ONLY_PROMPTS, REWRITE_FORMAT_HINT, render_observation
from agent.rewrite import _merge, _rewrite_messages, _search, _tool_call_text
from agent.schema import (Budget, BudgetState, Context, ErrorCode, Escalation, FinalAnswerAction, Observation,
                          SearchAction, SearchRecord, Step, StopReason, Trajectory)


class CascadeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gate: Literal["never", "always", "rerank_gap"] = "always"
    gap_threshold: Optional[float] = None  # rerank_gap 用：分差大于它才探测。只能从 validation 得到，写进配置
    probe: Literal["rewrite", "agent"] = "rewrite"

    def model_post_init(self, _) -> None:
        if self.gate == "rerank_gap" and self.gap_threshold is None:
            raise ValueError("gate=rerank_gap needs gap_threshold")


def rerank_gap(obs: Observation) -> Optional[float]:
    """重排第 1 名和第 2 名的分差。不到 2 条结果时返回 None（拿不准，交给探测）。"""
    if not obs.ok or len(obs.docs) < 2:
        return None
    return obs.docs[0].score - obs.docs[1].score


def _probe_messages(probe: str, question: str, first: SearchRecord) -> list[dict]:
    if probe == "rewrite":
        return _rewrite_messages("evidence_rewrite", question, [first])
    # Agent 提示词：和 Agent 循环里"第一次搜了原问题之后"看到的上下文逐字相同
    return [{"role": "system", "content": AGENT_PROMPTS.system},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": _tool_call_text(first.query)},
            {"role": "tool", "content": render_observation(first.observation)}]


def run_cascade(qid: str, question: str, llm: LLM, tool: SearchTool, budget: Budget,
                cfg: CascadeConfig) -> Trajectory:
    obs1, ms1 = _search(tool, question, budget.top_k)
    records = [SearchRecord(kind="original", query=question, observation=obs1,
                            num_new_docs=len(obs1.docs), tool_latency_ms=ms1)]
    feature = rerank_gap(obs1)
    probed = cfg.gate == "always" or (cfg.gate == "rerank_gap" and (feature is None or feature > cfg.gap_threshold))

    probe_step, direct_answer, outcome = None, None, "not_probed"
    if probed:
        t0 = time.perf_counter()
        gen = llm.generate(_probe_messages(cfg.probe, question, records[0]))
        llm_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_action(gen.text, AGENT_PROMPTS.format_hint if cfg.probe == "agent" else REWRITE_FORMAT_HINT)
        action, obs, tool_ms, new_docs = parsed.action, None, 0.0, 0
        if isinstance(action, SearchAction) and normalize_query(action.arguments.query) != normalize_query(question):
            obs, tool_ms = _search(tool, action.arguments.query, budget.top_k)
            new_docs = len({d.doc_id for d in obs.docs} - {d.doc_id for d in obs1.docs})
            records.append(SearchRecord(kind="evidence_rewrite", query=action.arguments.query, observation=obs,
                                        num_new_docs=new_docs, tool_latency_ms=tool_ms))
            outcome = "escalated"
        elif isinstance(action, SearchAction):
            # 原样重搜原问题：结果和第一跳一样，白花一次检索 → 当成拒绝再搜，不执行
            obs = Observation(ok=False, error_code=ErrorCode.DUPLICATE_QUERY, message="same as the first query")
            outcome = "duplicate"
        elif isinstance(action, FinalAnswerAction):  # 只有 agent 探测有 final_answer 工具
            direct_answer, outcome = action.arguments.answer, "answered"
        elif cfg.probe == "agent":
            obs, outcome = parsed.error, "format_error"  # Agent 格式写错了：记成格式错误，不重试，用第一跳证据作答
        else:
            outcome = "declined"  # 改写提示词下没写查询（大多是直接写了答案）：Day 10 的"拒绝再搜"
        probe_step = Step(turn=1, generated=gen.text, forced=False, num_tool_calls=parsed.num_tool_calls,
                          unclosed_tool_call=parsed.unclosed, ignored_suffix=parsed.ignored_suffix,
                          repaired_quotes=parsed.repaired_quotes, action=action, observation=obs,
                          num_new_docs=new_docs, llm_latency_ms=llm_ms, tool_latency_ms=tool_ms,
                          prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                          finish_reason=gen.finish_reason,
                          budget_state=BudgetState(turns_used=1, search_attempts=1 + isinstance(action, SearchAction),
                                                   search_calls_used=len(records)))

    # context.latency_ms 只算第一次检索：第二次检索和探测的耗时记在探测那一步里，run_eval 两者相加时不重复
    context = Context(source="retrieval", query=question, observation=_merge([r.observation for r in records]),
                      latency_ms=ms1, searches=records)
    escalation = Escalation(gate=cfg.gate, feature=feature, probed=probed, outcome=outcome)
    if direct_answer is not None:
        return Trajectory(qid=qid, question=question, method="cascade", context=context, budget=budget,
                          steps=[probe_step], budget_state=probe_step.budget_state, final_answer=direct_answer,
                          stop_reason=StopReason.ANSWERED, escalation=escalation)
    # 作答：搜索上限设成实际搜过的次数，模型就算想再搜也会被预算拦下（和 Static RAG 一样），不会碰到 tool=None
    answer_budget = Budget(max_turns=budget.max_turns, max_search_calls=len(records), top_k=budget.top_k)
    ans = run_episode(qid, question, llm, None, answer_budget, prompts=ANSWER_ONLY_PROMPTS, context=context,
                      method="cascade")
    offset = 1 if probe_step else 0
    steps = ([probe_step] if probe_step else []) + [
        s.model_copy(update={"turn": s.turn + offset,
                             "budget_state": s.budget_state.model_copy(update={"turns_used": s.budget_state.turns_used + offset})})
        for s in ans.steps]
    # 想搜但没执行的那次（和原问题重复）也算一次尝试；真的搜了的已经算在 context 的检索次数里
    state = ans.budget_state.model_copy(update={"turns_used": ans.budget_state.turns_used + offset,
                                                "search_attempts": ans.budget_state.search_attempts
                                                + (outcome == "duplicate")})
    return Trajectory(qid=qid, question=question, method="cascade", context=context, budget=budget, steps=steps,
                      budget_state=state, final_answer=ans.final_answer, stop_reason=ans.stop_reason,
                      error=ans.error, escalation=escalation)
