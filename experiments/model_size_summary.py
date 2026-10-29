"""模型尺寸对照（3B vs 7B，validation）：同一台机器（RTX 4090）、V2 的 5 组只换模型。

用法：python -m experiments.model_size_summary --since 20261029-000000 [--out docs/model_size_summary.json]
按运行名找 outputs/runs 下 --since 之后的 validation 运行（每组正好一个带 metrics.json 的；多了少了都报错），
并核对服务端实际加载的模型目录，防止"配置写的 7B、服务端其实是 3B"。

例外 PINNED：3B 在 2Wiki 上的 B3 / 分差门控 / agent 探测不重跑，用 10 月初同一台 4090 上的旧运行（用户 2026-10-30 决定）。
依据：3B 在 HotpotQA 上今天重跑的 5 组和 10 月初的旧运行逐字一致（1000/1000 条生成相同），
旧运行的 commit 到现在 QA 代码只新增了级联方法和指标，没改这三组的代码路径。

回答三个问题：
1. 同一组换成 7B 涨多少（7B − 3B，同一批题逐题配对）
2. 7B 上级联还比 B3 好吗（−B3，和 V2 主表同一口径）
3. 损失拆解：B0 →（检索）B3 →（按需升级）级联 → Oracle →（阅读和标签噪声）100%。
   Oracle 是读了金标的诊断上限，不是方法，不参与方法比较。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from experiments.compare_runs import load, paired_bootstrap

RUNS = Path("outputs/runs")
ARMS = [("B0 Direct", "direct"), ("B3 搜一次 + 重排", "static-rag-dense-rerank"),
        ("cascade 分差门控", "cascade-gap"), ("cascade 每题 agent 探测", "cascade-always-agent"),
        ("Oracle（诊断上限）", "oracle")]
SIZES = {"3b": "Qwen2.5-3B-Instruct", "7b": "Qwen2.5-7B-Instruct"}
DATASETS = {"hotpotqa": ("", "data/hotpotqa/v1/labels/validation.jsonl"),
            "2wiki": ("2wiki-", "data/2wiki/v1/labels/validation.jsonl")}
PINNED = {"qwen3b-2wiki-static-rag-dense-rerank": "20261002-164844-qwen3b-2wiki-static-rag-dense-rerank-validation",
          "qwen3b-2wiki-cascade-gap": "20261003-144150-qwen3b-2wiki-cascade-gap-validation",
          "qwen3b-2wiki-cascade-always-agent": "20261003-144949-qwen3b-2wiki-cascade-always-agent-validation"}


def find_run(name: str, since: str) -> Path:
    if name in PINNED:
        return RUNS / PINNED[name]
    pat = re.compile(rf"^(\d{{8}}-\d{{6}})-{re.escape(name)}-validation$")
    runs = [p for p in RUNS.iterdir() if (m := pat.match(p.name)) and m.group(1) >= since
            and (p / "metrics.json").exists()]
    if len(runs) != 1:
        raise SystemExit(f"{name}: {since} 之后找到 {len(runs)} 个完整的 validation 运行，应该正好 1 个：{sorted(runs)}")
    return runs[0]


def check_model(run: Path, size: str) -> None:
    root = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))["run"]["llm_server"]["root"]
    if Path(root).name != SIZES[size]:
        raise SystemExit(f"{run.name}: 服务端模型是 {root}，应该是 {SIZES[size]}")


def ci(d: tuple[float, float, float]) -> str:
    return f"{d[0] * 100:+5.1f} [{d[1] * 100:+5.1f},{d[2] * 100:+5.1f}]{'*' if d[1] > 0 or d[2] < 0 else ' '}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="只认这个时间之后的运行（运行名前缀 YYYYMMDD-HHMMSS），排除 4090 上的旧运行")
    ap.add_argument("--out")
    args = ap.parse_args()
    result = {}
    for ds, (infix, labels_path) in DATASETS.items():
        labels = {r["qid"]: r for r in map(json.loads, open(labels_path, encoding="utf-8"))}
        runs, trajs, metrics = {}, {}, {}
        for size in SIZES:
            for label, key in ARMS:
                p = find_run(f"qwen{size}-{infix}{key}", args.since)
                check_model(p, size)
                t = load(p)
                if set(t) != set(labels):
                    raise SystemExit(f"{p.name}: 题目集合和标签文件不一致")
                m = json.load(open(p / "metrics.json", encoding="utf-8"))
                if not m["valid"]:
                    raise SystemExit(f"{p.name}: metrics.valid = false")
                runs[size, label], trajs[size, label], metrics[size, label] = p.name, t, m
        commits = sorted({(RUNS / r / "git_commit.txt").read_text().strip()[:7] for r in runs.values()})

        print(f"\n== {ds} validation（{len(labels)} 题；commit {', '.join(commits)}）==")
        print("−B3 / 7B−3B 是逐题配对 bootstrap 的 EM 差（百分点）和 95% 区间，* = 区间不含 0；token 是平均输入 token")
        print(f"{'组':24s} {'3B EM':>6s} {'7B EM':>6s} {'7B−3B':>21s} {'3B −B3':>21s} {'7B −B3':>21s}"
              f" {'3B F1':>6s} {'7B F1':>6s} {'3B tok':>7s} {'7B tok':>7s}")
        rows = {}
        b3 = "B3 搜一次 + 重排"
        for label, _ in ARMS:
            em = {s: [trajs[s, label][q]["eval"]["em"] for q in labels] for s in SIZES}
            gain = paired_bootstrap([b - a for a, b in zip(em["3b"], em["7b"])])
            vs_b3 = {s: paired_bootstrap([trajs[s, label][q]["eval"]["em"] - trajs[s, b3][q]["eval"]["em"] for q in labels])
                     for s in SIZES}
            m3, m7 = metrics["3b", label], metrics["7b", label]
            rows[label] = {"runs": {s: runs[s, label] for s in SIZES},
                           **{f"{k}_{s}": metrics[s, label][k] for s in SIZES
                              for k in ("accuracy", "f1", "evidence_recall", "mean_prompt_tokens", "mean_search_calls",
                                        "format_error_rate", "probe_rate", "escalation_rate") if k in metrics[s, label]},
                           **{f"latency_p50_{s}": metrics[s, label]["latency_ms"]["p50"] for s in SIZES},
                           **{f"latency_p95_{s}": metrics[s, label]["latency_ms"]["p95"] for s in SIZES},
                           "em_7b_minus_3b": gain, **{f"em_minus_b3_{s}": vs_b3[s] for s in SIZES}}
            dash = f"{'—':>21s}"
            print(f"{label:24s} {m3['accuracy']:6.3f} {m7['accuracy']:6.3f} {ci(gain):>21s}"
                  f" {dash if label == b3 else ci(vs_b3['3b']):>21s} {dash if label == b3 else ci(vs_b3['7b']):>21s}"
                  f" {m3['f1']:6.3f} {m7['f1']:6.3f} {m3['mean_prompt_tokens']:7.0f} {m7['mean_prompt_tokens']:7.0f}")

        # 损失拆解：每一段是相邻两级的 EM 差；最后一段是 Oracle 到满分的距离（阅读错误 + EM 口径 / 标签噪声）
        print("  损失拆解（EM 点）：检索收益 B3−B0 | 升级收益 级联(agent 探测)−B3 | 检索还差 Oracle−级联 | 阅读+噪声 1−Oracle")
        decomp = {}
        for s in SIZES:
            acc = {label: metrics[s, label]["accuracy"] for label, _ in ARMS}
            d = {"retrieval_gain": acc[b3] - acc["B0 Direct"],
                 "escalation_gain": acc["cascade 每题 agent 探测"] - acc[b3],
                 "retrieval_gap_to_oracle": acc["Oracle（诊断上限）"] - acc["cascade 每题 agent 探测"],
                 "reading_and_noise": 1 - acc["Oracle（诊断上限）"]}
            decomp[s] = d
            print(f"  {s}: " + " | ".join(f"{v * 100:+5.1f}" for v in d.values()))

        types = sorted({l["type"] for l in labels.values()})
        by_type = {}
        print("  按题型 EM（3B → 7B）：" + "  ".join(f"{t}({sum(l['type'] == t for l in labels.values())})" for t in types))
        for label, _ in ARMS:
            by_type[label] = {}
            for s in SIZES:
                acc = defaultdict(list)
                for q, t in trajs[s, label].items():
                    acc[labels[q]["type"]].append(t["eval"]["em"])
                by_type[label][s] = {k: sum(v) / len(v) for k, v in acc.items()}
            print(f"  {label:24s} " + "  ".join(f"{by_type[label]['3b'][t]:.3f}→{by_type[label]['7b'][t]:.3f}" for t in types))

        # 分差门控只看第一跳的重排分数，和模型无关：两个模型被送去探测的应是同一批题，差别只来自模型
        gap = "cascade 分差门控"
        probed = {s: {q for q, t in trajs[s, gap].items() if t["escalation"]["probed"]} for s in SIZES}
        same = len(probed["3b"] & probed["7b"]), len(probed["3b"] | probed["7b"])
        print(f"  分差门控探测的题：两个模型重合 {same[0]} / 并集 {same[1]}")

        # agent 探测相对 B3 的收益，按探测时模型的选择拆开：直接作答（修读错）/ 再搜一次（补证据）
        agent = "cascade 每题 agent 探测"
        probe = {}
        print("  agent 探测按模型选择拆开（净修对题数 = 错→对 − 对→错；召回 = 再搜那批题的证据召回 B3 → 再搜后）")
        for s in SIZES:
            outcome = {q: t["escalation"]["outcome"] for q, t in trajs[s, agent].items()}
            net = defaultdict(int)
            for q in labels:
                net[outcome[q]] += trajs[s, agent][q]["eval"]["em"] - trajs[s, b3][q]["eval"]["em"]
            esc = [q for q in labels if outcome[q] == "escalated"]
            rec = (sum(trajs[s, b3][q]["eval"]["evidence_recall"] for q in esc) / len(esc),
                   sum(trajs[s, agent][q]["eval"]["evidence_recall"] for q in esc) / len(esc))
            probe[s] = {"counts": dict(Counter(outcome.values())), "net_fixed": dict(net),
                        "escalated_recall_b3": rec[0], "escalated_recall_after": rec[1],
                        "net_fixed_per_escalation": net["escalated"] / len(esc)}
            print(f"  {s}: 直接作答 {probe[s]['counts'].get('answered', 0)} 题 净 {net['answered']:+d}；"
                  f"再搜 {len(esc)} 题 净 {net['escalated']:+d}（每次再搜 {net['escalated'] / len(esc):.3f}），"
                  f"召回 {rec[0]:.3f} → {rec[1]:.3f}")

        # 两种花算力的方式直接比：换大模型（7B 只搜一次 / 分差门控） vs 小模型 + 每题探测
        cross = {}
        for big in (b3, gap):
            d = paired_bootstrap([trajs["7b", big][q]["eval"]["em"] - trajs["3b", agent][q]["eval"]["em"] for q in labels])
            cross[f"7b {big} − 3b {agent}"] = d
            print(f"  7B {big} − 3B {agent}：{ci(d)}")
        result[ds] = {"rows": rows, "decomposition": decomp, "by_type": by_type, "commits": commits,
                      "gap_probed_overlap": same, "agent_probe": probe, "cross": cross}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
