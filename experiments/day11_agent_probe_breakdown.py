"""Day 11.4 拆 agent 探测的收益：和 B3 同题配对，按探测结果分组；再离线推算几种 agent 探测的变体。

用法：
  python -m experiments.day11_agent_probe_breakdown --labels data/hotpotqa/v1/labels/validation.jsonl \
      --b3 outputs/runs/<B3 运行> --agent outputs/runs/<cascade gate=always probe=agent 运行> --gap 4.19 5.69
为什么能离线推算：agent 探测看到的内容（B3 的第一跳 + Agent 提示词）和门控无关，贪心解码下同一道题探测的输出不变。
所以"分差门控 + agent 探测"= 门控放过的题用 B3 的结果、送去探测的题用每题都探测那次运行的结果；
"不再搜"变体 = 探测时直接作答的题用探测的答案，其余用 B3 的答案（只花一次检索）。
EM 直接读两次运行里评测器算好的结果；标签文件只用来确认题目集合和取题型。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.compare_runs import load, paired_bootstrap


def _tokens(t: dict) -> float:
    return sum(s["prompt_tokens"] for s in t["steps"])


def _gap(t: dict) -> float | None:
    docs = t["context"]["observation"]["docs"]
    return docs[0]["score"] - docs[1]["score"] if len(docs) >= 2 else None


def simulate(b3: dict, ag: dict, gap: float | None, escalate: bool) -> dict:
    """gap=None 表示每题都探测。成本只算输入 token 和检索次数（延迟在线跑时再看）。"""
    ems, tin, probed, searches = [], 0.0, 0, 0
    for q, tb in b3.items():
        g, ta = _gap(tb), ag[q]
        if gap is not None and g is not None and g <= gap:  # 门控放过（和 agent/cascade.py 一样：拿不准就探测）
            ems.append(tb["eval"]["em"]); tin += _tokens(tb); searches += 1
            continue
        probed += 1
        outcome, probe_in = ta["escalation"]["outcome"], ta["steps"][0]["prompt_tokens"]
        if outcome == "answered":
            ems.append(ta["eval"]["em"]); tin += probe_in; searches += 1
        elif outcome == "escalated" and escalate:
            ems.append(ta["eval"]["em"]); tin += _tokens(ta); searches += 2
        else:  # 不再搜 / 格式错误 / 重复：用第一跳证据作答 = B3 的答案，多花一次探测
            ems.append(tb["eval"]["em"]); tin += probe_in + _tokens(tb); searches += 1
    n = len(ems)
    diff = paired_bootstrap([e - tb["eval"]["em"] for e, tb in zip(ems, b3.values())])
    return {"em": sum(ems) / n, "em_minus_b3": diff, "probe_rate": probed / n,
            "searches": searches / n, "prompt_tokens": tin / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--b3", required=True)
    ap.add_argument("--agent", required=True, help="cascade gate=always probe=agent 的运行")
    ap.add_argument("--gap", type=float, nargs="*", default=[], help="额外推算这些分差门槛 + agent 探测")
    ap.add_argument("--out")
    args = ap.parse_args()
    labels = {r["qid"]: r for r in map(json.loads, open(args.labels, encoding="utf-8"))}
    b3, ag = load(Path(args.b3)), load(Path(args.agent))
    if set(b3) != set(labels) or set(ag) != set(labels):
        raise SystemExit("两次运行的题目集合和标签文件不一致")

    groups: dict[str, list[str]] = {}
    for q, t in ag.items():
        groups.setdefault(t["escalation"]["outcome"], []).append(q)
    result = {"n": len(b3), "by_outcome": {}, "variants": {}}
    print(f"n={len(b3)}   B3 EM {sum(t['eval']['em'] for t in b3.values()) / len(b3):.3f}")
    print(f"{'outcome':12s} {'n':>4s} {'B3':>4s} {'agent':>5s} {'赢':>3s} {'输':>3s}  B3 召回  题型")
    for o, qs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        win = sum(ag[q]["eval"]["em"] > b3[q]["eval"]["em"] for q in qs)
        loss = sum(ag[q]["eval"]["em"] < b3[q]["eval"]["em"] for q in qs)
        row = {"n": len(qs), "b3_correct": sum(b3[q]["eval"]["em"] for q in qs),
               "agent_correct": sum(ag[q]["eval"]["em"] for q in qs), "wins": win, "losses": loss,
               "b3_evidence_recall": sum(b3[q]["eval"]["evidence_recall"] for q in qs) / len(qs),
               "types": dict(Counter(labels[q]["type"] for q in qs))}
        if args.gap:
            row["wins_gap_le"] = {str(g): sum(ag[q]["eval"]["em"] > b3[q]["eval"]["em"] and _gap(b3[q]) is not None
                                              and _gap(b3[q]) <= g for q in qs) for g in args.gap}
        result["by_outcome"][o] = row
        print(f"{o:12s} {len(qs):4d} {row['b3_correct']:4.0f} {row['agent_correct']:5.0f} {win:3d} {loss:3d}"
              f"  {row['b3_evidence_recall']:.3f}  {row['types']}" + (f"  赢的题里分差≤门槛 {row['wins_gap_le']}"
                                                                      if args.gap else ""))
    variants = {"always_agent_escalate": (None, True), "always_agent_no_second_search": (None, False)}
    for g in args.gap:
        variants[f"gap{g}_agent_escalate"] = (g, True)
        variants[f"gap{g}_agent_no_second_search"] = (g, False)
    for name, (g, esc) in variants.items():
        v = simulate(b3, ag, g, esc)
        result["variants"][name] = v
        m, lo, hi = v["em_minus_b3"]
        print(f"  {name:34s} EM {v['em']:.3f}  −B3 {m * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
              f"  探测 {v['probe_rate']:.2f}  检索 {v['searches']:.2f}  输入 {v['prompt_tokens']:.0f}")
    online = sum(t["eval"]["em"] for t in ag.values()) / len(ag)
    if abs(result["variants"]["always_agent_escalate"]["em"] - online) > 1e-9:
        raise SystemExit(f"推算的每题都探测 EM 和在线运行 {online:.3f} 对不上，检查运行是否配对")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
