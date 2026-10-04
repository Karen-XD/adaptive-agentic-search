"""Day 15：Setting A 上评测微调后的商品重排模型，并离线推算"Dense → 微调重排"的按需升级。

同一批固定候选（每个查询的全部已标注商品），比三种排序：
  dense          e5 向量相似度（Day 13 最强的通用方法，零模型调用之外的成本）
  generic_rerank 通用 bge-reranker-base 给全部候选打分
  ft_rerank      微调后的重排模型给全部候选打分
三者都对完整候选池排序；78% 的查询候选不超过 20 个，Day 13 的"Hybrid 取前 20 再重排"在这些查询上就是全池重排。
重排的 max_length 和训练时相同（256）；通用模型也用 256，对比才公平。

按需升级（离线、精确：两条路的打分都是确定性的）：不升级的查询用 Dense 排序，升级的用微调重排。
门控信号沿用 Day 13 的五个（Dense 第 1、2 名分差等）；另报"理想门控"（只升级有益的查询）上限和随机门控。
每个查询的重排耗时单独计时（一个查询的候选一批），用来算升级的真实成本。

用法：
  python -m experiments.day15_eval_reranker --model /root/autodl-tmp/checkpoints/esci_reranker_v1/best --split validation
  python -m experiments.day15_eval_reranker --model ... --split test --final
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.commerce_metrics import GAIN, GAIN_NO_COMPLEMENT, aggregate, ndcg_at_k
from experiments.compare_runs import paired_bootstrap
from experiments.day13_commerce_escalation import auc
from experiments.day15_finetune_reranker import load_corpus_text, load_eval_split
from retrieval.dense import E5Encoder, PASSAGE_PREFIX, QUERY_PREFIX

ROOT = Path(__file__).resolve().parents[1]
E5 = "/root/autodl-tmp/hf_models/e5-base-v2"
GENERIC = "/root/autodl-tmp/hf_models/bge-reranker-base"


@torch.inference_mode()
def rerank_scores(model, tok, query: str, texts: list[str], max_length: int) -> tuple[list[float], float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = tok([(query, t) for t in texts], padding=True, truncation="only_second", max_length=max_length,
                return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = model(**batch).logits[:, 0].float().cpu().tolist()
    torch.cuda.synchronize()
    return s, (time.perf_counter() - t0) * 1000


def rank(scores: list[float], doc_ids: list[str]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], doc_ids[i]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="微调后的检查点目录")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--final", action="store_true", help="跑 test 必须加：test 只在最后评测时跑")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", default="outputs/runs")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("test 只在最后评测时跑，要跑请加 --final")

    groups = load_eval_split(args.split)
    text = load_corpus_text()
    enc = E5Encoder(E5, device="cuda", max_length=256)
    tok = AutoTokenizer.from_pretrained(GENERIC)
    generic = AutoModelForSequenceClassification.from_pretrained(GENERIC).to("cuda").eval()
    ft_tok = AutoTokenizer.from_pretrained(args.model)
    ft = AutoModelForSequenceClassification.from_pretrained(args.model).to("cuda").eval()
    # 热身：第一次前向要初始化 CUDA kernel，不算进耗时
    rerank_scores(ft, ft_tok, "warmup", ["warmup"] * 8, args.max_length)
    rerank_scores(generic, tok, "warmup", ["warmup"] * 8, args.max_length)

    rows = []
    for n, g in enumerate(groups):
        texts = [text[d] for d in g["doc_ids"]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qv = enc.encode([QUERY_PREFIX + g["query"]])[0]
        dv = enc.encode([PASSAGE_PREFIX + t for t in texts])
        dense = (dv @ qv).tolist()
        torch.cuda.synchronize()
        dense_ms = (time.perf_counter() - t0) * 1000
        gen_s, gen_ms = rerank_scores(generic, tok, g["query"], texts, args.max_length)
        ft_s, ft_ms = rerank_scores(ft, ft_tok, g["query"], texts, args.max_length)
        orders = {"dense": rank(dense, g["doc_ids"]), "generic_rerank": rank(gen_s, g["doc_ids"]),
                  "ft_rerank": rank(ft_s, g["doc_ids"])}
        d_sorted = sorted(dense, reverse=True)
        rows.append({
            "qid": g["qid"], "query": g["query"], "num_candidates": len(texts),
            "labels": {m: [g["labels"][i] for i in o] for m, o in orders.items()},
            "dense_gap": d_sorted[0] - d_sorted[1] if len(d_sorted) > 1 else 0.0,
            "dense_top1": d_sorted[0], "query_words": len(g["query"].split()),
            "dense_ms": dense_ms, "generic_ms": gen_ms, "ft_ms": ft_ms,
        })
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(groups)}", flush=True)

    methods = ["dense", "generic_rerank", "ft_rerank"]
    results = {}
    for m in methods:
        recs = [{"ranked_labels": r["labels"][m]} for r in rows]
        res = aggregate(recs, GAIN)
        res["ndcg@10_no_complement"] = aggregate(recs, GAIN_NO_COMPLEMENT)["ndcg@10"]
        results[m] = res
    for r in rows:
        for m in methods:
            r[f"{m}_ndcg"] = ndcg_at_k(r["labels"][m], 10, GAIN)
    comps = {}
    for a, b in [("ft_rerank", "dense"), ("ft_rerank", "generic_rerank"), ("generic_rerank", "dense")]:
        comps[f"{a}-{b}"] = paired_bootstrap([r[f"{a}_ndcg"] - r[f"{b}_ndcg"] for r in rows])
    lat = {m: {"mean": float(np.mean([r[k] for r in rows])), "p50": float(np.percentile([r[k] for r in rows], 50)),
               "p95": float(np.percentile([r[k] for r in rows], 95))}
           for m, k in (("dense", "dense_ms"), ("generic_rerank", "generic_ms"), ("ft_rerank", "ft_ms"))}

    print(f"\n{'方法':16s} {'nDCG@10':>8s} {'(C=0)':>7s} {'E召回@10':>9s} {'MRR':>6s} {'耗时/查询':>10s}")
    for m in methods:
        r = results[m]
        print(f"{m:16s} {r['ndcg@10']:8.4f} {r['ndcg@10_no_complement']:7.4f} {r['recall@10']:9.4f} {r['mrr']:6.3f}"
              f" {lat[m]['mean']:8.1f}ms")
    for k, (mean, lo, hi) in comps.items():
        print(f"  {k:28s} {mean * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{' *' if lo > 0 or hi < 0 else ''}")

    # ---- 按需升级：Dense → 微调重排 ----
    up = [r["ft_rerank_ndcg"] - r["dense_ndcg"] for r in rows]
    pos = [u > 1e-9 for u in up]
    signals = {"dense_gap_neg（分差小→升级）": [-r["dense_gap"] for r in rows],
               "dense_top1_neg（第 1 名分低→升级）": [-r["dense_top1"] for r in rows],
               "query_words（词多→升级）": [r["query_words"] for r in rows],
               "pool_size（池大→升级）": [r["num_candidates"] for r in rows]}
    aucs = {k: auc(v, pos) for k, v in signals.items()}
    n = len(rows)
    base = float(np.mean([r["dense_ndcg"] for r in rows]))
    full = float(np.mean([r["ft_rerank_ndcg"] for r in rows]))
    ideal = float(np.mean([max(r["dense_ndcg"], r["ft_rerank_ndcg"]) for r in rows]))
    print(f"\n升级有益的查询 {sum(pos)}/{n}  有害 {sum(u < -1e-9 for u in up)}  不变 {sum(abs(u) <= 1e-9 for u in up)}")
    print(f"从不升级 {base:.4f}  每条都升级 {full:.4f}  理想门控 {ideal:.4f}")
    print(f"{'门控信号':34s} AUC")
    for k, v in aucs.items():
        print(f"{k:34s} {v:.3f}")
    fracs = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    curves = {}
    for k, sig in signals.items():
        order = sorted(range(n), key=lambda i: -sig[i])
        pts = []
        for f in fracs:
            idx = set(order[:int(round(f * n))])
            nd = float(np.mean([rows[i]["ft_rerank_ndcg"] if i in idx else rows[i]["dense_ndcg"] for i in range(n)]))
            pts.append({"frac": f, "ndcg": nd, "gain_share": (nd - base) / (full - base) if full != base else None})
        curves[k] = pts
    rng = random.Random(0)
    rand = []
    for f in fracs:
        acc = [float(np.mean([rows[i]["ft_rerank_ndcg"] if i in idx else rows[i]["dense_ndcg"] for i in range(n)]))
               for idx in (set(rng.sample(range(n), int(round(f * n)))) for _ in range(200))]
        rand.append({"frac": f, "ndcg": float(np.mean(acc))})
    print("升级比例            " + "  ".join(f"{f:>6.0%}" for f in fracs))
    for k, pts in curves.items():
        print(f"{k[:18]:18s}  " + "  ".join(f"{p['ndcg']:.4f}" for p in pts))
    print(f"{'随机':18s}  " + "  ".join(f"{p['ndcg']:.4f}" for p in rand))

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-esci-ft-rerank-{args.split}" + ("-dirty" if dirty else "")
    out = ROOT / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "git_commit.txt").write_text(commit + ("\n(dirty working tree)" if dirty else "") + "\n", encoding="utf-8")
    train_meta = Path(args.model).parent / "train_meta.json"
    (out / "config.json").write_text(json.dumps(
        {"argv": sys.argv, "split": args.split, "model": args.model, "max_length": args.max_length,
         "generic": GENERIC, "dense": E5, "setting": "A: fixed annotated candidates, full pool reranked",
         "train_meta": json.load(open(train_meta)) if train_meta.exists() else None},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(
        {"split": args.split, "num_queries": n, "methods": results, "paired": comps, "latency_ms": lat,
         "escalation": {"never": base, "always": full, "ideal": ideal, "num_helped": sum(pos),
                        "gate_auc": aucs, "curves": curves, "random": rand}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
