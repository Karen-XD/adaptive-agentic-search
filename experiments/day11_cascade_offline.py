"""Day 11 按需升级的离线推算：用 B3 和两跳证据改写两次已有的运行，逐题推算各种门控策略的质量和成本。

用法：
  python -m experiments.day11_cascade_offline --labels data/hotpotqa/v1/labels/validation.jsonl \
      --b3 outputs/runs/<B3 运行> --two-hop outputs/runs/<two_hop_evidence 运行>
为什么能离线推算：agent/cascade.py 的探测和两跳证据改写的改写步看到的内容逐字相同、作答步也相同（tests/test_cascade.py），
贪心解码下输出也相同。所以一道题如果被送去探测，结果就是两跳那组这道题的结果；没送去探测，就是 B3 的结果。
拒绝再搜的题，两跳那组退回原问题又搜了一遍（证据和第一跳相同），cascade 不会再搜，成本按 B3 + 一次探测算。
推算完要在线跑一遍验证（experiments/run_cascade.sh）。

门控特征：第一跳重排第 1 名和第 2 名的分差（rerank_gap）；对照：前 3 名平均分、第 1 名分数、随机。
门槛的选法：在 21 个分位点里选"EM 不低于每题都探测"的最省的那个。只在 validation 上选。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

FEATURES = {
    "rerank_gap": lambda sc: sc[0] - sc[1],      # 大 → 探测
    "neg_mean3": lambda sc: -float(np.mean(sc)),  # 前 3 名平均分低 → 探测
    "neg_top1": lambda sc: -sc[0],                # 第 1 名分数低 → 探测
}


def _load(run: Path) -> dict:
    return {t["qid"]: t for t in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8"))}


def _answer_cost(t: dict) -> tuple[float, float, float]:
    return (sum(s["prompt_tokens"] for s in t["steps"]), sum(s["completion_tokens"] for s in t["steps"]),
            sum(s["llm_latency_ms"] for s in t["steps"]))


def build_rows(b3: dict, two: dict) -> list[dict]:
    rows = []
    for q, tb in b3.items():
        te = two[q]
        r2 = te["context"]["searches"][1]
        docs = tb["context"]["observation"]["docs"]
        sc = [d["score"] for d in docs] + [-20.0] * (3 - len(docs))  # 不到 3 条时补一个很低的分
        b_in, b_out, b_ms = _answer_cost(tb)
        e_in, e_out, e_ms = _answer_cost(te)
        rows.append({
            "qid": q, "b3_em": float(tb["eval"]["em"]), "two_em": float(te["eval"]["em"]),
            "b3_f1": tb["eval"]["f1"], "two_f1": te["eval"]["f1"],
            "declined": r2["fallback"], "features": {k: f(sc) for k, f in FEATURES.items()},
            "b3": {"in": b_in, "out": b_out, "ms": b_ms + tb["context"]["latency_ms"], "ret_ms": tb["context"]["latency_ms"]},
            "probe": {"in": r2["prompt_tokens"], "out": r2["completion_tokens"], "ms": r2["llm_latency_ms"]},
            "esc": {"in": e_in, "out": e_out, "ms": e_ms, "ret_ms": r2["tool_latency_ms"]},
        })
    return rows


def simulate(rows: list[dict], probe_mask: list[bool]) -> dict:
    em = f1 = tin = tout = calls = rets = ms = 0.0
    for r, probed in zip(rows, probe_mask):
        b, p, e = r["b3"], r["probe"], r["esc"]
        if not probed:
            em += r["b3_em"]; f1 += r["b3_f1"]; tin += b["in"]; tout += b["out"]; calls += 1; rets += 1; ms += b["ms"]
        elif r["declined"]:  # 用第一跳证据作答 = B3 的答案
            em += r["b3_em"]; f1 += r["b3_f1"]; tin += p["in"] + b["in"]; tout += p["out"] + b["out"]
            calls += 2; rets += 1; ms += b["ms"] + p["ms"]
        else:
            em += r["two_em"]; f1 += r["two_f1"]; tin += p["in"] + e["in"]; tout += p["out"] + e["out"]
            calls += 2; rets += 2; ms += b["ret_ms"] + p["ms"] + e["ret_ms"] + e["ms"]
    n = len(rows)
    return {"em": em / n, "f1": f1 / n, "prompt_tokens": tin / n, "completion_tokens": tout / n,
            "llm_calls": calls / n, "searches": rets / n, "latency_ms": ms / n, "probe_rate": sum(probe_mask) / n}


def curve(rows: list[dict], feature: str, n_points: int = 21) -> list[dict]:
    vals = np.array([r["features"][feature] for r in rows])
    out = []
    for thr in np.quantile(vals, np.linspace(0, 1, n_points)):
        mask = [v > thr for v in vals]
        out.append({"threshold": float(thr), **simulate(rows, mask)})
    out.append({"threshold": float("-inf"), **simulate(rows, [True] * len(rows))})
    return out


def pick_threshold(points: list[dict], target_em: float) -> dict | None:
    ok = [p for p in points if p["em"] >= target_em - 1e-9]
    return min(ok, key=lambda p: p["prompt_tokens"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="只用来确认题目集合一致；EM 直接读两次运行里评测器算好的结果")
    ap.add_argument("--b3", required=True)
    ap.add_argument("--two-hop", required=True)
    ap.add_argument("--apply-gap", type=float, help="额外报告用这个分差门槛（比如另一个数据集上选的）的结果")
    ap.add_argument("--out")
    args = ap.parse_args()
    if "test" in Path(args.labels).name:
        raise SystemExit("门槛只在 validation 上选")
    b3, two = _load(Path(args.b3)), _load(Path(args.two_hop))
    qids = {json.loads(l)["qid"] for l in open(args.labels, encoding="utf-8")}
    if set(b3) != qids or set(two) != qids:
        raise SystemExit("两次运行的题目集合和标签文件不一致")
    rows = build_rows(b3, two)
    never, always = simulate(rows, [False] * len(rows)), simulate(rows, [True] * len(rows))
    ideal = simulate(rows, [r["two_em"] > r["b3_em"] for r in rows])
    result = {"n": len(rows), "never_probe_B3": never, "always_probe": always, "oracle_gate": ideal,
              "declined": sum(r["declined"] for r in rows), "curves": {}, "picked": {}}
    gain = always["em"] - never["em"]
    for f in FEATURES:
        pts = curve(rows, f)
        result["curves"][f] = pts
        best = pick_threshold(pts, always["em"])
        result["picked"][f] = best
    rnd = random.Random(0)
    for frac in (0.2, 0.3, 0.4, 0.5):
        gap_thr = float(np.quantile([r["features"]["rerank_gap"] for r in rows], 1 - frac))
        g = simulate(rows, [r["features"]["rerank_gap"] > gap_thr for r in rows])
        rand_em = np.mean([simulate(rows, [rnd.random() < frac for _ in rows])["em"] for _ in range(200)])
        result.setdefault("vs_random", []).append({
            "probe_rate": frac, "gap_share_of_gain": (g["em"] - never["em"]) / gain if gain else None,
            "random_share_of_gain": (rand_em - never["em"]) / gain if gain else None})
    if args.apply_gap is not None:
        result["applied_gap"] = {"threshold": args.apply_gap,
                                 **simulate(rows, [r["features"]["rerank_gap"] > args.apply_gap for r in rows])}
    print(f"n={len(rows)}  拒绝再搜 {result['declined']}")
    for k in ("never_probe_B3", "always_probe", "oracle_gate"):
        v = result[k]
        print(f"  {k:15s} EM {v['em']:.3f} F1 {v['f1']:.3f} | 输入 {v['prompt_tokens']:6.0f} | 调用 {v['llm_calls']:.2f}"
              f" 检索 {v['searches']:.2f} | {v['latency_ms']:5.0f}ms | 探测 {v['probe_rate']:.2f}")
    for f, p in result["picked"].items():
        print(f"  门槛[{f}]: " + (f"thr={p['threshold']:.2f} 探测 {p['probe_rate']:.2f} EM {p['em']:.3f}"
                                  f" 输入 {p['prompt_tokens']:.0f} {p['latency_ms']:.0f}ms" if p else "无"))
    for v in result["vs_random"]:
        print(f"  探测 {v['probe_rate']:.0%}: 分差门控拿到升级收益的 {v['gap_share_of_gain']:.0%}，随机 {v['random_share_of_gain']:.0%}")
    if "applied_gap" in result:
        v = result["applied_gap"]
        print(f"  套用门槛 {v['threshold']}: 探测 {v['probe_rate']:.2f} EM {v['em']:.3f}"
              f"（拿到升级收益的 {(v['em'] - never['em']) / gain:.0%}）输入 {v['prompt_tokens']:.0f} {v['latency_ms']:.0f}ms")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
