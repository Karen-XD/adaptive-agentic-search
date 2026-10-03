"""Day 11.4 诊断：同一份证据，换一种"交给模型的格式"，作答准确率差多少。

起因：cascade 的 agent 探测（configs/qwen3b_cascade_always_agent.yaml）比 B3 高 8 个点，其中模型直接作答的 92 题
证据和 B3 逐字相同、却多答对 11 题。两者的区别有两处，本脚本做 2×2 把它们拆开：
  证据位置  user：放在用户消息里（"Documents: ... Question: ..."，B3 的写法）
            tool：写成模型自己发出的 search(原问题) + 工具返回（Agent 循环 / agent 探测的写法）
  系统提示  answer_only：只能作答（final_answer 一个工具，B3 的写法）
            agent：Agent 提示词（search + final_answer）；模型想搜就按预算用完处理，让它作答（和 Agent 循环一样）
user + answer_only 就是 B3 本身，要和 B3 的运行逐字一致（检查复用证据的方式没走样）。

用法（先起 vLLM，不需要检索服务；证据直接取已有运行轨迹里存下的检索结果）：
  python -m experiments.day11_answer_format_probe --context-run outputs/runs/<B3 运行> \
      --labels data/hotpotqa/v1/labels/validation.jsonl
诊断用，只在 validation / debug 上跑；金标只在最后打分时读。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from agent.llm import RetryingLLM
from agent.loop import _error, run_episode
from agent.parser import parse_action
from agent.prompts import AGENT_PROMPTS, ANSWER_ONLY_PROMPTS, Prompts, render_observation
from agent.rewrite import _tool_call_text
from agent.schema import Budget, Context, ErrorCode, FinalAnswerAction, SearchAction
from evaluation.qa_metrics import exact_match, f1_score
from evaluation.run_eval import build_llm
from experiments.compare_runs import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"user_answer_only": ("user", ANSWER_ONLY_PROMPTS), "user_agent": ("user", AGENT_PROMPTS),
        "tool_answer_only": ("tool", ANSWER_ONLY_PROMPTS), "tool_agent": ("tool", AGENT_PROMPTS)}


def _tool_format(question: str, context: Context, llm, prompts: Prompts, max_turns: int) -> dict:
    """证据写成已经发生过的 search 调用和工具返回，接着让模型作答。不能再搜：想搜就回预算用完的报错。"""
    records = context.searches or []
    pairs = [(r.query, r.observation) for r in records] or [(context.query, context.observation)]
    messages = [{"role": "system", "content": prompts.system}, {"role": "user", "content": f"Question: {question}"}]
    for q, obs in pairs:
        messages += [{"role": "assistant", "content": _tool_call_text(q)},
                     {"role": "tool", "content": render_observation(obs)}]
    wanted_search = format_errors = 0
    for _ in range(max_turns):
        gen = llm.generate(messages)
        parsed = parse_action(gen.text, prompts.format_hint)
        if isinstance(parsed.action, FinalAnswerAction):
            return {"answer": parsed.action.arguments.answer, "wanted_search": wanted_search,
                    "format_errors": format_errors}
        if isinstance(parsed.action, SearchAction):
            wanted_search += 1
            obs = _error(ErrorCode.BUDGET_EXCEEDED, prompts.forced_notice)
        else:
            format_errors += 1
            obs = parsed.error
        messages += [{"role": "assistant", "content": gen.text}, {"role": "tool", "content": render_observation(obs)}]
    return {"answer": None, "wanted_search": wanted_search, "format_errors": format_errors}


def _user_format(qid: str, question: str, context: Context, llm, prompts: Prompts, max_turns: int) -> dict:
    """B3 的写法：直接复用 run_episode。搜索上限 = 已经搜过的次数，模型想搜会被预算拦下再作答。"""
    budget = Budget(max_turns=max_turns, max_search_calls=context.num_searches(), top_k=len(context.observation.docs))
    t = run_episode(qid, question, llm, None, budget, prompts=prompts, context=context, method="probe")
    return {"answer": t.final_answer,
            "wanted_search": sum(isinstance(s.action, SearchAction) for s in t.steps),
            "format_errors": sum(s.action is None for s in t.steps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-run", required=True, help="提供证据的已有运行（B3 / 两跳等只作答的方法）")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", default="configs/qwen3b_base.yaml", help="取 vLLM 地址、采样参数、max_turns")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, help="只跑前 N 题（冒烟测试）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    base_llm, llm_info = build_llm(cfg["llm"])  # 和 run_eval 同一个构造方式（含预热、服务端信息）
    llm = RetryingLLM(base_llm, **cfg["llm"].get("retry", {}))
    max_turns = cfg["budget"]["max_turns"]
    run = Path(args.context_run)
    trajs = [json.loads(line) for line in open(run / "trajectories.jsonl", encoding="utf-8")][:args.limit]
    arms = args.arms.split(",")

    rows = []
    for i, t in enumerate(trajs):
        context = Context.model_validate(t["context"])
        row = {"qid": t["qid"], "source_answer": t["final_answer"]}
        for arm in arms:
            where, prompts = ARMS[arm]
            row[arm] = (_tool_format(t["question"], context, llm, prompts, max_turns) if where == "tool"
                        else _user_format(t["qid"], t["question"], context, llm, prompts, max_turns))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(trajs)}", flush=True)

    # 打分：到这里才读金标
    gold = {r["qid"]: r["answer"] for r in map(json.loads, open(args.labels, encoding="utf-8"))}
    for r in rows:
        r["gold"] = gold[r["qid"]]
        for arm in arms:
            r[arm]["em"] = float(exact_match(r[arm]["answer"], r["gold"]))
            r[arm]["f1"] = f1_score(r[arm]["answer"], r["gold"])
    yn = lambda a: (a or "").strip().lower().rstrip(".")
    metrics = {"context_run": run.name, "num_questions": len(rows), "arms": {}}
    for arm in arms:
        answers = [yn(r[arm]["answer"]) for r in rows]
        metrics["arms"][arm] = {
            "em": round(sum(r[arm]["em"] for r in rows) / len(rows), 4),
            "f1": round(sum(r[arm]["f1"] for r in rows) / len(rows), 4),
            "wanted_search_rate": round(sum(r[arm]["wanted_search"] > 0 for r in rows) / len(rows), 4),
            "format_error_rate": round(sum(r[arm]["format_errors"] > 0 for r in rows) / len(rows), 4),
            "no_answer": sum(r[arm]["answer"] is None for r in rows),
            "true_false_answers": sum(a in ("true", "false") for a in answers),
            "same_as_source_run": sum(r[arm]["answer"] == r["source_answer"] for r in rows),
            "em_minus_user_answer_only": [round(x, 4) for x in paired_bootstrap(
                [r[arm]["em"] - r["user_answer_only"]["em"] for r in rows])] if "user_answer_only" in arms else None,
        }
    print(json.dumps(metrics["arms"], indent=1, ensure_ascii=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    split = Path(args.labels).stem
    out = ROOT / "outputs/runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-day11-answer-format-{run.name[16:]}"
                                   + ("-dirty" if dirty else ""))
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"args": vars(args), "split": split, "llm": cfg["llm"], "llm_server": llm_info,
                                                     "max_turns": max_turns}, allow_unicode=True), encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + ("-dirty" if dirty else "") + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
