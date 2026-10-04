"""Day 12 test 主表：预先登记的各组（docs/DAY12_PREREG.md）和 B3 逐题配对，报质量和成本；2Wiki 另按题型报 EM。

用法：python -m experiments.day12_test_summary [--out outputs/day12_test_summary.json]
按运行名找 outputs/runs 下的 test 运行（每组只应有一个带 metrics.json 的；多于一个就报错，test 不该重跑）。
离线推算组"分差门控 + agent 探测"= B3 + 每题 agent 探测逐题拼接（day11_agent_probe_breakdown.simulate，贪心下精确）。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.compare_runs import load, paired_bootstrap
from experiments.day11_agent_probe_breakdown import simulate

RUNS = Path("outputs/runs")
ARMS = [("B3", "static-rag-dense-rerank"), ("全量多轮 Agent", "agent-dense-rerank"),
        ("cascade 每题探测（rewrite）", "cascade-always"), ("cascade 分差门控（rewrite）", "cascade-gap"),
        ("cascade 每题 agent 探测", "cascade-always-agent")]
DATASETS = {"hotpotqa": ("qwen3b-", "data/hotpotqa/v1/labels/test.jsonl", 4.19),
            "2wiki": ("qwen3b-2wiki-", "data/2wiki/v1/labels/test.jsonl", 5.69)}


def find_test_run(name: str) -> Path:
    pat = re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(name)}-test$")
    runs = [p for p in RUNS.iterdir() if pat.match(p.name) and (p / "metrics.json").exists()]
    if len(runs) != 1:
        raise SystemExit(f"{name}: 找到 {len(runs)} 个 test 运行，应该正好 1 个")
    return runs[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    result = {}
    for ds, (prefix, labels_path, gap) in DATASETS.items():
        labels = {r["qid"]: r for r in map(json.loads, open(labels_path, encoding="utf-8"))}
        runs = {label: find_test_run(prefix + key) for label, key in ARMS}
        trajs = {label: load(p) for label, p in runs.items()}
        b3 = trajs["B3"]
        for label, t in trajs.items():
            if set(t) != set(labels):
                raise SystemExit(f"{ds} {label}: 题目集合和标签文件不一致")
        b3_tok = json.load(open(runs["B3"] / "metrics.json"))["mean_prompt_tokens"]
        print(f"\n== {ds} test（{len(labels)} 题） ==")
        print(f"{'组':28s} {'EM':>6s} {'−B3 [95%]':>22s} {'F1':>6s} {'召回':>6s} {'检索':>5s} {'输入':>6s} {'倍数':>5s} {'p50/p95 ms':>12s}")
        rows = {}
        for label, p in runs.items():
            m = json.load(open(p / "metrics.json"))
            diff = paired_bootstrap([trajs[label][q]["eval"]["em"] - b3[q]["eval"]["em"] for q in b3])
            rows[label] = {"run": p.name, "em": m["accuracy"], "f1": m["f1"], "evidence_recall": m["evidence_recall"],
                           "search_calls": m["mean_search_calls"], "prompt_tokens": m["mean_prompt_tokens"],
                           "latency_p50": m["latency_ms"]["p50"], "latency_p95": m["latency_ms"]["p95"],
                           "em_minus_b3": diff, "probe_rate": m.get("probe_rate"), "escalation_rate": m.get("escalation_rate")}
            sig = " *" if label != "B3" and diff[1] > 0 else "  "
            print(f"{label:28s} {m['accuracy']:6.3f} {diff[0] * 100:+6.1f} [{diff[1] * 100:+5.1f},{diff[2] * 100:+5.1f}]{sig}"
                  f" {m['f1']:6.3f} {m['evidence_recall']:6.3f} {m['mean_search_calls']:5.2f} {m['mean_prompt_tokens']:6.0f}"
                  f" {m['mean_prompt_tokens'] / b3_tok:5.2f} {m['latency_ms']['p50']:5.0f}/{m['latency_ms']['p95']:5.0f}")
        v = simulate(b3, trajs["cascade 每题 agent 探测"], gap, True)
        rows[f"分差门控 {gap} + agent 探测（离线）"] = v
        print(f"{'分差门控 + agent 探测（离线）':28s} {v['em']:6.3f} {v['em_minus_b3'][0] * 100:+6.1f}"
              f" [{v['em_minus_b3'][1] * 100:+5.1f},{v['em_minus_b3'][2] * 100:+5.1f}]{' *' if v['em_minus_b3'][1] > 0 else '  '}"
              f" {'':6s} {'':6s} {v['searches']:5.2f} {v['prompt_tokens']:6.0f} {v['prompt_tokens'] / b3_tok:5.2f}   探测 {v['probe_rate']:.2f}")
        types = sorted({l["type"] for l in labels.values()})
        by_type = {}
        if len(types) > 1 and ds == "2wiki":
            print("  按题型 EM：" + "  ".join(types))
            for label in runs:
                acc = defaultdict(list)
                for q, t in trajs[label].items():
                    acc[labels[q]["type"]].append(t["eval"]["em"])
                by_type[label] = {k: sum(v) / len(v) for k, v in acc.items()}
                print(f"  {label:28s} " + "  ".join(f"{by_type[label][k]:.3f}" for k in types))
        result[ds] = {"rows": rows, "by_type": by_type}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
