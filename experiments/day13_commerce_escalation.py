"""商品域的按需升级（Day 13 收尾）：固定候选池上，"Dense 排序" vs "LLM 重排"，用信号决定哪些查询值得升级。

和 Day 11 的 QA 级联同构，换了一个领域：
  - 便宜的一路：e5 向量检索给候选打分排序（零模型调用，Setting A 实测 nDCG@10 0.849）
  - 贵的一路：把查询和 Dense 前 20 名商品给 Qwen2.5-3B，让它**逐条判相关性等级**（3/2/1/0），再按分数重排；
    池外商品保持 Dense 顺序跟在后面。平均约 880 token / 查询
  - 门控信号：Dense 第 1、2 名的分数差（分差小 = 拿不准 → 升级）。这是 QA 上分差信号的直接对照，
    另比三个候选信号：查询词数、第 1 名绝对分数、候选池大小

为什么用点式打分而不是让模型直接输出排序：3B 模型直接给排列时，25 条查询里严格合法的排列只有 6 条（有重复、
跳号、" -1,-2,-3" 之类的填充），重排后反而比 Dense 差（7 好 16 坏）。改成逐条判等级后覆盖率 92%、
nDCG@10 0.847 → 0.858。**两种问法都留在下面的对比里**（`--mode`），因为"哪种问法能让小模型当好重排器"
本身就是可报告的结论。

贪心下离线精确推算：升级的查询用 LLM 重排的结果，不升级的用 Dense 结果，两条路各自在线跑出来，
所以任何门槛的质量-成本曲线都能直接拼出来（同 Day 11）。在线只需跑"从不升级"和"每题都升级"两个端点。

标签只在评测器里用；LLM 看到的只有查询词和商品标题/品牌，看不到 ESCI 标签。

用法：
  python -m experiments.day13_commerce_escalation --split validation
  python -m experiments.day13_commerce_escalation --split test --final
输出：outputs/runs/<时间>-esci-escalation-<split>/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from agent.llm import VLLMClient
from evaluation.commerce_metrics import GAIN, aggregate, dcg
from retrieval.dense import E5Encoder, PASSAGE_PREFIX, QUERY_PREFIX

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/root/autodl-tmp/hf_models/e5-base-v2"
LLM_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "qwen2.5-3b-instruct"
POOL = 20  # 只重排 Dense 前 20 名：nDCG@10 只看前 10，池外商品没有机会进来

SYSTEM_POINTWISE = """You rate how relevant each product is to a shopping query.
For each product give one integer:
3 = exactly the product the shopper asked for
2 = a close substitute the shopper would likely accept
1 = related or complementary, but not the product asked for
0 = not relevant
Answer with a JSON object mapping the product number to its score."""

SYSTEM_LISTWISE = """You are a search relevance ranker for an e-commerce query.
You get a shopping query and a numbered list of products (title and brand).
Rank the products from most relevant to least relevant for what the shopper wants.
Answer with a JSON array of the numbers in order, best first, each number exactly once, written as integers."""


def build_prompt(query: str, docs: list[dict], mode: str) -> str:
    lines = [f"Query: {query}", "", "Products:"]
    for i, d in enumerate(docs, 1):
        brand = d.get("_brand")
        lines.append(f"[{i}] {d['title']}" + (f" (Brand: {brand})" if brand else ""))
    lines.append("")
    if mode == "pointwise":
        lines.append(f'Answer with a JSON object, e.g. {{"1": 3, "2": 0, ...}}, every product exactly once:')
    else:
        lines.append(f"Rank all {len(docs)} products from most to least relevant. "
                     f"Answer with a JSON array of integers:")
    return "\n".join(lines)


def parse_listwise(text: str, n: int) -> tuple[list[int], bool]:
    """宽松解析：按出现顺序取合法编号并去重，缺的补在最后。返回（0-based 下标顺序, 是否严格合法）。"""
    nums = [int(x) for x in re.findall(r"\d+", text)]
    seen: list[int] = []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.append(x)
    order = [x - 1 for x in seen] + [i for i in range(n) if i not in {x - 1 for x in seen}]
    exact = len(nums) == n and nums == seen  # 没有重复、没有跳号、没有多余的内容
    return order, exact


def parse_pointwise(text: str, n: int) -> tuple[dict[int, int], bool]:
    """宽松解析 {"3": 2, ...}；也接受 "3": 2 或 [3] = 2 这类写法。返回（{1-based 编号: 分数}, 覆盖是否完整）。"""
    scores: dict[int, int] = {}
    for m in re.finditer(r'"?\[?(\d{1,3})\]?"?\s*[:=]\s*(-?\d)', text):
        i, v = int(m.group(1)), int(m.group(2))
        if 1 <= i <= n and 0 <= v <= 3 and i not in scores:
            scores[i] = v
    return scores, len(scores) == n


def ndcg_of(labels: list[str], k: int = 10) -> float:
    gain = [GAIN[x] for x in labels]
    best = dcg(sorted(gain, reverse=True)[:k])
    return dcg(gain[:k]) / best if best > 0 else 0.0


def auc(signal: list[float], pos: list[bool]) -> float:
    """信号越大越该升级：正例应排前面。并列算 0.5。用秩和公式算。"""
    p = sum(pos)
    n = len(pos) - p
    if p == 0 or n == 0:
        return float("nan")
    order = sorted(range(len(pos)), key=lambda i: -signal[i])
    # 排名相同的信号值算并列，给平均秩
    ranks = [0.0] * len(pos)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and signal[order[j + 1]] == signal[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based 平均秩
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(pos)) if pos[i])
    return 1 - (rank_pos - p * (p + 1) / 2) / (p * n)


def paired_bootstrap(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    from experiments.compare_runs import paired_bootstrap as pb
    return pb(diffs, n_boot, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--final", action="store_true", help="跑 test 必须加：test 只在最后评测时跑")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", default="pointwise", choices=["pointwise", "listwise"])
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="outputs/runs")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("test 只在最后评测时跑，要跑请加 --final")

    labels = [json.loads(l) for l in open(ROOT / f"data/esci/v1/labels/{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        labels = labels[:args.limit]
    corpus = {}
    with open(ROOT / "data/esci/v1/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["doc_id"]] = d
    import pandas as pd
    prod = pd.read_parquet(ROOT / "data/raw/esci/shopping_queries_dataset_products.parquet",
                           columns=["product_id", "product_brand"])
    brand = dict(zip(prod.product_id, prod.product_brand))

    enc = E5Encoder(MODEL, device="cuda", max_length=256)
    llm = VLLMClient(LLM_URL, LLM_MODEL, temperature=0.0, max_tokens=args.max_tokens)
    print(f"warmup {llm.warmup():.0f}ms  mode={args.mode}  pool={args.pool}", flush=True)

    records = []
    t0 = time.time()
    for n, lab in enumerate(labels):
        cands = lab["candidates"]
        docs = [corpus[c["doc_id"]] for c in cands]
        for d, c in zip(docs, cands):
            d["_brand"] = brand.get(c["product_id"])
        label_of = {c["doc_id"]: c["esci_label"] for c in cands}
        query = lab["question"]

        qv = enc.encode([QUERY_PREFIX + query])[0]
        dv = enc.encode([PASSAGE_PREFIX + f"{d['title']}\n{d['text']}" for d in docs])
        scores = (dv @ qv).tolist()
        dense_order = sorted(range(len(docs)), key=lambda i: (-scores[i], docs[i]["doc_id"]))
        gap = (scores[dense_order[0]] - scores[dense_order[1]]) if len(docs) > 1 else 0.0
        pool, rest = dense_order[:args.pool], dense_order[args.pool:]

        gen = llm.generate([{"role": "system", "content": SYSTEM_POINTWISE if args.mode == "pointwise"
                             else SYSTEM_LISTWISE},
                            {"role": "user", "content": build_prompt(query, [docs[i] for i in pool], args.mode)}])
        if args.mode == "pointwise":
            sc, exact = parse_pointwise(gen.text, len(pool))
            within = sorted(sc, key=lambda i: (-sc[i], i))
            llm_order = [pool[i - 1] for i in within] + [pool[i - 1] for i in range(1, len(pool) + 1) if i not in sc] + rest
            coverage = len(sc) / len(pool)
        else:
            order, exact = parse_listwise(gen.text, len(pool))
            llm_order = [pool[i] for i in order] + rest
            coverage = 1.0
        records.append({
            "qid": lab["qid"], "question": query, "num_candidates": len(docs),
            "gap": gap, "gap_ratio": gap / (abs(scores[dense_order[0]]) + 1e-9),
            "top1_score": scores[dense_order[0]], "query_words": len(query.split()),
            "dense_labels": [label_of[docs[i]["doc_id"]] for i in dense_order],
            "llm_labels": [label_of[docs[i]["doc_id"]] for i in llm_order],
            "exact_format": exact, "coverage": coverage,
            "llm_tokens": (gen.prompt_tokens or 0) + (gen.completion_tokens or 0),
            "llm_prompt_tokens": gen.prompt_tokens, "llm_completion_tokens": gen.completion_tokens,
        })
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(labels)}  {time.time() - t0:.0f}s", flush=True)

    for r in records:
        r["dense_ndcg"], r["llm_ndcg"] = ndcg_of(r["dense_labels"]), ndcg_of(r["llm_labels"])
        r["uplift"] = r["llm_ndcg"] - r["dense_ndcg"]
    dense_res = aggregate([{"ranked_labels": r["dense_labels"]} for r in records])
    llm_res = aggregate([{"ranked_labels": r["llm_labels"]} for r in records])
    mean, lo, hi = paired_bootstrap([r["uplift"] for r in records])
    tokens = sum(r["llm_tokens"] for r in records) / len(records)
    print(f"\nn={len(records)}  Dense nDCG@10 {dense_res['ndcg@10']:.4f}  LLM 重排 {llm_res['ndcg@10']:.4f}"
          f"  −Dense {mean * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]")
    print(f"严格合法输出 {sum(r['exact_format'] for r in records)}/{len(records)}"
          f"  打分覆盖率 {sum(r['coverage'] for r in records) / len(records):.1%}"
          f"  平均 token {tokens:.0f}  变好 {sum(r['uplift'] > 1e-9 for r in records)}"
          f" / 变差 {sum(r['uplift'] < -1e-9 for r in records)}")

    # ---- 门控信号的 AUC ----
    pos = [r["uplift"] > 1e-9 for r in records]
    signals = {"gap_neg（分差小→升级）": [-r["gap"] for r in records],
               "gap_ratio_neg": [-r["gap_ratio"] for r in records],
               "top1_neg（第 1 名分低→升级）": [-r["top1_score"] for r in records],
               "query_words（词多→升级）": [r["query_words"] for r in records],
               "pool_size（池大→升级）": [r["num_candidates"] for r in records]}
    aucs = {}
    print(f"\n升级有益的查询 {sum(pos)}/{len(pos)}")
    print(f"{'门控信号':30s} AUC")
    for name, sig in signals.items():
        aucs[name] = auc(list(sig), pos)
        print(f"{name:30s} {aucs[name]:.3f}")

    # ---- 质量-成本曲线（贪心下精确） ----
    print(f"\n{'门控信号':30s} {'最优 nDCG@10':>12s} {'升级率':>7s} {'token/查询':>11s}")
    curves = {}
    for name, sig in signals.items():
        order = sorted(range(len(records)), key=lambda i: -sig[i])
        pts = []
        for frac in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            k = int(round(frac * len(records)))
            idx = set(order[:k])
            nd = sum(records[i]["llm_ndcg"] if i in idx else records[i]["dense_ndcg"] for i in range(len(records))) / len(records)
            pts.append({"frac": frac, "ndcg": nd, "tokens": sum(records[i]["llm_tokens"] for i in idx) / len(records)})
        curves[name] = pts
        best = max(pts, key=lambda p: p["ndcg"])
        print(f"{name:30s} {best['ndcg']:12.4f} {best['frac']:7.0%} {best['tokens']:11.0f}")
    # 随机门控对照
    rng = random.Random(0)
    rand_pts = []
    for frac in (0.1, 0.3, 0.5, 1.0):
        acc = []
        for _ in range(200):
            k = int(round(frac * len(records)))
            idx = set(rng.sample(range(len(records)), k))
            acc.append(sum(records[i]["llm_ndcg"] if i in idx else records[i]["dense_ndcg"] for i in range(len(records))) / len(records))
        rand_pts.append({"frac": frac, "ndcg": sum(acc) / len(acc)})
    print("随机门控对照：" + "  ".join(f"{p['frac']:.0%}→{p['ndcg']:.4f}" for p in rand_pts))

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-esci-escalation-{args.mode}-{args.split}" + ("-dirty" if dirty else "")
    out = ROOT / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "git_commit.txt").write_text(
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        + ("\n(dirty working tree)" if dirty else "") + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(
        {"argv": sys.argv, "split": args.split, "limit": args.limit, "mode": args.mode, "pool": args.pool,
         "max_tokens": args.max_tokens, "dense_model": MODEL, "llm_model": LLM_MODEL, "llm_url": LLM_URL,
         "system_prompt": SYSTEM_POINTWISE if args.mode == "pointwise" else SYSTEM_LISTWISE,
         "gain_mapping": GAIN, "setting": "A: fixed annotated candidates"}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({
        "split": args.split, "num_queries": len(records), "mode": args.mode, "pool": args.pool,
        "never_escalate_dense": {**dense_res, "mean_llm_tokens": 0.0, "llm_call_rate": 0.0},
        "always_escalate_llm": {**llm_res, "mean_llm_tokens": tokens, "llm_call_rate": 1.0,
                                "exact_format_rate": sum(r["exact_format"] for r in records) / len(records),
                                "mean_coverage": sum(r["coverage"] for r in records) / len(records)},
        "llm_minus_dense": {"mean": mean, "lo": lo, "hi": hi,
                            "better": sum(r["uplift"] > 1e-9 for r in records),
                            "worse": sum(r["uplift"] < -1e-9 for r in records)},
        "gate_auc": aucs, "curves": curves, "random_gate": rand_pts,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
