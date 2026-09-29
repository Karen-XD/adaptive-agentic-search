"""两次运行逐题对比：主要指标的变化 + 逐题 EM 配对 bootstrap 区间 + 变好 / 变差的题。

用法：python -m experiments.compare_runs outputs/runs/<旧运行> outputs/runs/<新运行> [--show N]
两次运行必须是同一个划分、同一批题（逐题配对才有意义）。离线分析，不改变任何实验结果。

为什么报配对 bootstrap 区间而不只报均值：200 题上几个点的差距常在噪声范围内（Day 3.5 的 +3 个点区间跨过 0）。
配对（同一道题新旧相减再重采样）比两组独立比较的区间窄，因为题目本身难度的差异被抵消了。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

METRICS = ["accuracy", "f1", "evidence_recall", "format_error_rate", "mean_search_calls", "mean_turns",
           "mean_prompt_tokens", "mean_completion_tokens"]


def load(run: Path) -> dict[str, dict]:
    return {t["qid"]: t for t in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8"))}


def paired_bootstrap(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """均值和 95% 区间。seed 固定：同样的输入每次报同样的区间。"""
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    return sum(diffs) / len(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--show", type=int, default=10, help="变好 / 变差的题各打印几道")
    args = ap.parse_args()
    old_run, new_run = Path(args.old), Path(args.new)
    old, new = load(old_run), load(new_run)
    if set(old) != set(new):
        raise SystemExit(f"两次运行的题目不同（{len(old)} vs {len(new)}），不能逐题配对")
    mo, mn = (json.loads((r / "metrics.json").read_text(encoding="utf-8")) for r in (old_run, new_run))

    print(f"{old_run.name} → {new_run.name}（{len(old)} 题）")
    for k in METRICS:
        if k in mo and k in mn:
            print(f"  {k:24s} {mo[k]:8.3f} → {mn[k]:8.3f}  ({mn[k] - mo[k]:+.3f})")
    print(f"  {'stop_reasons':24s} {mo['stop_reasons']} → {mn['stop_reasons']}")

    qids = sorted(old)
    for name, key in (("EM", "em"), ("F1", "f1")):
        mean, lo, hi = paired_bootstrap([float(new[q]["eval"][key]) - float(old[q]["eval"][key]) for q in qids])
        print(f"  {name} 新 − 旧（逐题配对 bootstrap 95%）: {mean * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]")

    better = [q for q in qids if new[q]["eval"]["em"] and not old[q]["eval"]["em"]]
    worse = [q for q in qids if old[q]["eval"]["em"] and not new[q]["eval"]["em"]]
    print(f"  逐题：变好 {len(better)} 题，变差 {len(worse)} 题")
    for tag, qs in (("变好", better), ("变差", worse)):
        for q in qs[:args.show]:
            print(f"    {tag} {q} | {old[q]['question'][:60]!r} | 金标 {new[q]['eval']['gold'][:30]!r} | "
                  f"{old[q]['final_answer']!r} → {new[q]['final_answer']!r}")


if __name__ == "__main__":
    main()
