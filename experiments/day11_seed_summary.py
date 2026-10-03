"""Day 11 多 seed 汇总：cascade 每个 seed 和同一 seed 的 B3 逐题配对，报 EM 差的 bootstrap 区间，按"多数 seed 显著才算显著"。

用法：python -m experiments.day11_seed_summary [--temperature 0.7]
按运行名自动找 outputs/runs 下的 seed 运行（同名多次取最新、有 metrics.json 的那次），缺的 seed 打印"缺"。
同一 seed 配对：采样噪声里有一部分来自 B3 那一跳的作答本身，配同 seed 的 B3 才是"同一次采样下升级带来的变化"。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiments.compare_runs import load, paired_bootstrap

RUNS = Path("outputs/runs")
DATASETS = {
    "hotpotqa": ("qwen3b-static-rag-dense-rerank", ["qwen3b-cascade-gap", "qwen3b-cascade-always-agent"]),
    "2wiki": ("qwen3b-2wiki-static-rag-dense-rerank", ["qwen3b-2wiki-cascade-gap", "qwen3b-2wiki-cascade-always-agent"]),
}
SEEDS = (1, 2, 3)


def find_run(name: str, suffix: str) -> Path | None:
    pat = re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(name)}-validation{re.escape(suffix)}$")
    runs = sorted(p for p in RUNS.iterdir() if pat.match(p.name) and (p / "metrics.json").exists())
    return runs[-1] if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", default="0.7")
    args = ap.parse_args()
    for dataset, (b3_name, methods) in DATASETS.items():
        print(f"\n== {dataset} ==")
        for seed in SEEDS:
            b3 = find_run(b3_name, f"-t{args.temperature}-s{seed}")
            print(f"  B3 s{seed}: " + (f"EM {json.load(open(b3 / 'metrics.json'))['accuracy']:.3f}  {b3.name}" if b3 else "缺"))
        for name in methods:
            sig, done = 0, 0
            for seed in SEEDS:
                b3 = find_run(b3_name, f"-t{args.temperature}-s{seed}")
                run = find_run(name, f"-t{args.temperature}-s{seed}")
                if not (b3 and run):
                    print(f"  {name} s{seed}: 缺")
                    continue
                old, new = load(b3), load(run)
                if set(old) != set(new):
                    raise SystemExit(f"{run.name} 和 {b3.name} 题目不同，不能配对")
                mean, lo, hi = paired_bootstrap([new[q]["eval"]["em"] - old[q]["eval"]["em"] for q in old])
                m = json.load(open(run / "metrics.json"))
                done += 1
                sig += lo > 0
                print(f"  {name} s{seed}: EM {m['accuracy']:.3f}  −B3 {mean * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                      f"{' *' if lo > 0 else '  '}  探测 {m.get('probe_rate', 0):.2f}  再搜 {m.get('escalation_rate', 0):.2f}"
                      f"  检索 {m['mean_search_calls']:.2f}  输入 {m['mean_prompt_tokens']:.0f}")
            print(f"  -> {name}: {sig}/{done} 个 seed 显著优于 B3" + ("（多数显著）" if done and sig * 2 > len(SEEDS) else ""))


if __name__ == "__main__":
    main()
