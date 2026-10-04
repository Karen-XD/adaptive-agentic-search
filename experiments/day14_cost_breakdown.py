"""Day 14 工具成本全表：每组 test 运行的大模型调用（探测 / 改写 vs 作答分开）、token、检索和重排次数、各段耗时。

模型侧（精确，来自轨迹）：每次调用的输入 / 输出 token 和耗时。cascade 被探测的题，第一步是探测（rewrite 或 agent）调用，
其余是作答调用（agent 探测直接作答时，探测就是作答，记在"探测"里）。
检索侧：检索次数和检索总耗时来自轨迹；"召回 vs 重排"的耗时拆分和重排输入 token 用 day14_rerank_cost.py 回放测出的
单次均值乘检索次数（--rerank-cost，每个数据集一个 json）。不重排的组（B1 / B2 用 BM25）重排成本记 0。

用法：python -m experiments.day14_cost_breakdown --rerank-cost hotpotqa=<json> 2wiki=<json> --out docs/day14_cost_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RUNS = Path("outputs/runs")
ARMS = [("B0 不检索", "direct"), ("B1 BM25 搜一次", "static-rag"), ("B2 BM25 多轮 Agent", "agent"),
        ("B3 Dense+重排 搜一次", "static-rag-dense-rerank"), ("全量多轮 Agent（Dense+重排）", "agent-dense-rerank"),
        ("cascade 每题探测（rewrite）", "cascade-always"), ("cascade 分差门控（rewrite）", "cascade-gap"),
        ("cascade 每题 agent 探测", "cascade-always-agent")]
PREFIX = {"hotpotqa": "qwen3b-", "2wiki": "qwen3b-2wiki-"}
RERANK_ARMS = {"static-rag-dense-rerank", "agent-dense-rerank", "cascade-always", "cascade-gap", "cascade-always-agent"}


def find_test_run(name: str) -> Path | None:
    pat = re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(name)}-test$")
    runs = [p for p in RUNS.iterdir() if pat.match(p.name) and (p / "metrics.json").exists()]
    if len(runs) > 1:
        raise SystemExit(f"{name}: {len(runs)} 个 test 运行，test 不该重跑")
    return runs[0] if runs else None


def summarize(run: Path, rerank_unit: dict | None) -> dict:
    ts = [json.loads(l) for l in open(run / "trajectories.jsonl", encoding="utf-8")]
    m = json.load(open(run / "metrics.json"))
    n = len(ts)
    probe_calls = probe_in = probe_out = probe_ms = 0.0
    ans_calls = ans_in = ans_out = ans_ms = 0.0
    searches = tool_ms = 0.0
    for t in ts:
        steps = t["steps"]
        esc = t.get("escalation") or {}
        probe_idx = 0 if esc.get("probed") else None
        for i, s in enumerate(steps):
            pin, pout = s.get("prompt_tokens") or 0, s.get("completion_tokens") or 0
            if i == probe_idx:
                probe_calls += 1; probe_in += pin; probe_out += pout; probe_ms += s.get("llm_latency_ms") or 0
            else:
                ans_calls += 1; ans_in += pin; ans_out += pout; ans_ms += s.get("llm_latency_ms") or 0
            tool_ms += s.get("tool_latency_ms") or 0
        ctx = t.get("context") or {}
        if ctx:
            # context.latency_ms 只算流程替模型做的第一次检索；cascade 的第二次检索记在探测那一步的 tool_latency_ms
            # （决策记录 2026-10-02），上面逐步累加时已经算过，不要再从 context.searches 里加一遍
            tool_ms += ctx.get("latency_ms") or 0
        searches += t["budget_state"]["search_calls_used"]
    row = {
        "run": run.name, "n": n, "em": m["accuracy"], "f1": m["f1"],
        "llm_calls": (probe_calls + ans_calls) / n,
        "probe_calls": probe_calls / n, "probe_input_tokens": probe_in / n, "probe_output_tokens": probe_out / n,
        "answer_calls": ans_calls / n, "answer_input_tokens": ans_in / n, "answer_output_tokens": ans_out / n,
        "input_tokens": (probe_in + ans_in) / n, "output_tokens": (probe_out + ans_out) / n,
        "llm_ms": (probe_ms + ans_ms) / n, "probe_ms": probe_ms / n,
        "search_calls": searches / n, "search_ms": tool_ms / n,
        "latency_p50": m["latency_ms"]["p50"], "latency_p95": m["latency_ms"]["p95"],
    }
    if rerank_unit:
        row["rerank_calls"] = row["search_calls"]
        row["rerank_ms_est"] = row["search_calls"] * rerank_unit["rerank_ms"]["mean"]
        row["rerank_input_tokens_est"] = row["search_calls"] * rerank_unit["rerank_input_tokens_per_search"]["mean"]
    else:
        row["rerank_calls"] = row["rerank_ms_est"] = row["rerank_input_tokens_est"] = 0.0
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-cost", nargs="*", default=[], help="数据集=回放 json，例如 hotpotqa=outputs/...json")
    ap.add_argument("--out")
    args = ap.parse_args()
    unit = {k: json.load(open(v)) for k, v in (x.split("=", 1) for x in args.rerank_cost)}
    result = {}
    for ds, prefix in PREFIX.items():
        print(f"\n== {ds} test ==")
        print(f"{'组':28s} {'EM':>6s} {'调用':>5s} {'探测入':>6s} {'作答入':>6s} {'总入':>6s} {'总出':>5s}"
              f" {'模型ms':>7s} {'检索':>5s} {'检索ms':>7s} {'重排ms*':>7s} {'重排tok*':>8s} {'p50':>6s} {'p95':>6s}")
        rows = {}
        for label, key in ARMS:
            run = find_test_run(prefix + key)
            if run is None:
                print(f"{label:28s} （缺）")
                continue
            r = summarize(run, unit.get(ds) if key in RERANK_ARMS else None)
            rows[label] = r
            print(f"{label:28s} {r['em']:6.3f} {r['llm_calls']:5.2f} {r['probe_input_tokens']:6.0f} {r['answer_input_tokens']:6.0f}"
                  f" {r['input_tokens']:6.0f} {r['output_tokens']:5.0f} {r['llm_ms']:7.0f} {r['search_calls']:5.2f}"
                  f" {r['search_ms']:7.1f} {r['rerank_ms_est']:7.1f} {r['rerank_input_tokens_est']:8.0f}"
                  f" {r['latency_p50']:6.0f} {r['latency_p95']:6.0f}")
        result[ds] = {"rows": rows, "rerank_unit": unit.get(ds)}
    print("\n* 重排耗时 / token 为回放测得的单次均值 × 检索次数（day14_rerank_cost.py）")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
