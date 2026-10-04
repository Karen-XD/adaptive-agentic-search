"""商品搜索 Setting A：在**固定候选商品集**上比较各种排序方法（Day 13 主评测）。

每个查询的候选就是官方已标注的那批 Query–Product 对（平均约 20 个），所有方法用**完全相同**的候选池，
所以这里衡量的是"排序好坏"，不是全库召回。四种方法：
  random     随机排序（下界；种子固定，可复现）
  oracle     按 ESCI 标签完美排序（上界，只作诊断，不参与和方法的比较）
  bm25       商品文本的词匹配打分
  dense      e5 向量相似度
  hybrid     两路分数归一化后各半融合（RRF 的简化版：用 min-max 归一化后加权，分数口径不同不能直接相加）
  hybrid_rerank  hybrid 取前 R 条候选 → Cross-Encoder 重排

标签只在评测器里用：`oracle` 那组是明确标注的诊断上限；其余方法打分时看不到标签。

用法：
  python -m experiments.day13_commerce_setting_a --split validation
  python -m experiments.day13_commerce_setting_a --split test --final      # 正式跑 test 必须加 --final
输出：outputs/runs/<时间>-esci-setting-a-<split>/metrics.json + per_query.jsonl
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

import bm25s

from evaluation.commerce_metrics import GAIN, GAIN_NO_COMPLEMENT, aggregate
from retrieval.bm25 import _tokenize
from retrieval.dense import E5Encoder, PASSAGE_PREFIX, QUERY_PREFIX
from retrieval.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/root/autodl-tmp/hf_models/e5-base-v2"
RERANKER = "/root/autodl-tmp/hf_models/bge-reranker-base"


def build_index() -> tuple[dict[str, dict], dict[str, int], "bm25s.BM25"]:
    """商品文本 + doc_id 在语料里的行号 + 语料库级 BM25 索引（Setting A 只给候选里的商品打分）。"""
    out, position = {}, {}
    with open(ROOT / "data/esci/v1/corpus.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            out[d["doc_id"]] = d
            position[d["doc_id"]] = i
    bm25 = bm25s.BM25.load(ROOT / "indexes/esci_v1_bm25")
    return out, position, bm25


def bm25_for_query(bm25, position: dict[str, int], query: str, doc_ids: list[str]) -> list[float]:
    """整库 BM25 分数里取出这些候选的分数（IDF / 平均长度都来自整个商品库，不在候选内重新统计）。"""
    tokens = _tokenize([query])
    if not tokens.ids or not tokens.ids[0]:
        return [0.0] * len(doc_ids)
    # _tokenize 返回的 id 是这一次分词的局部编号，和索引的词表不是一回事；要先换回词串，
    # 让索引用自己的词表查（不在词表里的词会被丢掉）。直接传局部 id 会给错的词打分，BM25 退化成随机
    inv = {i: w for w, i in tokens.vocab.items()}
    words = [inv[i] for i in tokens.ids[0]]
    if not bm25.get_tokens_ids(words):
        return [0.0] * len(doc_ids)
    scores = bm25.get_scores(words)
    return [float(scores[position[d]]) if d in position else 0.0 for d in doc_ids]


def minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    return [0.5] * len(scores) if hi - lo < 1e-12 else [(s - lo) / (hi - lo) for s in scores]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--final", action="store_true", help="跑 test 必须加：test 只在最后评测时跑")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rerank-pool", type=int, default=20, help="hybrid 取前多少条进重排")
    ap.add_argument("--out", default="outputs/runs")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("test 只在最后评测时跑，要跑请加 --final")

    labels = [json.loads(l) for l in open(ROOT / f"data/esci/v1/labels/{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        labels = labels[:args.limit]
    corpus, position, bm25 = build_index()
    missing = {c["doc_id"] for l in labels for c in l["candidates"]} - set(corpus)
    if missing:
        raise SystemExit(f"{len(missing)} 个候选商品不在语料里")

    enc = E5Encoder(MODEL, device="cuda", max_length=256)
    reranker = CrossEncoderReranker(RERANKER, max_length=512, batch_size=32)

    records = {m: [] for m in ("random", "oracle", "bm25", "dense", "hybrid", "hybrid_rerank")}
    per_query = []
    t0 = time.time()
    for n, lab in enumerate(labels):
        cands = lab["candidates"]
        docs = [corpus[c["doc_id"]] for c in cands]
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        label_of = {c["doc_id"]: c["esci_label"] for c in cands}
        query = lab["question"]

        # random：固定种子，每题独立
        rng = random.Random(f"{args.split}-{lab['qid']}")
        order = list(range(len(docs)))
        rng.shuffle(order)

        # bm25：用**语料库级**打分（IDF 才有统计意义；只在 20 个候选里数词频会退化成"谁更长谁分高"）
        bm = np.array(bm25_for_query(bm25, position, query, [c["doc_id"] for c in cands]))

        # dense：查询和商品都过 e5
        qv = enc.encode([QUERY_PREFIX + query])[0]
        dv = enc.encode([PASSAGE_PREFIX + t for t in texts])
        de = dv @ qv

        hy = [0.5 * a + 0.5 * b for a, b in zip(minmax(bm.tolist()), minmax(de.tolist()))]
        top = sorted(range(len(docs)), key=lambda i: (-hy[i], docs[i]["doc_id"]))[:args.rerank_pool]
        rr_scores = reranker.score(query, [type("D", (), {"title": docs[i]["title"], "text": docs[i]["text"]})() for i in top])

        rankings = {
            "random": order,
            "oracle": sorted(range(len(docs)), key=lambda i: (-GAIN[label_of[docs[i]["doc_id"]]], docs[i]["doc_id"])),
            "bm25": sorted(range(len(docs)), key=lambda i: (-bm[i], docs[i]["doc_id"])),
            "dense": sorted(range(len(docs)), key=lambda i: (-de[i], docs[i]["doc_id"])),
            "hybrid": sorted(range(len(docs)), key=lambda i: (-hy[i], docs[i]["doc_id"])),
            "hybrid_rerank": top + sorted(set(range(len(docs))) - set(top),
                                          key=lambda i: (-hy[i], docs[i]["doc_id"])),
        }
        # hybrid_rerank 的顺序再按重排分数排一次（top 内部按重排分，池外按融合分跟在后面）
        rr_order = [top[j] for j in sorted(range(len(top)), key=lambda j: (-rr_scores[j], docs[top[j]]["doc_id"]))]
        rankings["hybrid_rerank"] = rr_order + rankings["hybrid_rerank"][len(top):]

        for m, idx in rankings.items():
            records[m].append({"ranked_labels": [label_of[docs[i]["doc_id"]] for i in idx],
                               "latency_ms": 0.0})
        per_query.append({"qid": lab["qid"], "question": query, "num_candidates": len(docs),
                          "top10": {m: [docs[i]["product_id"] for i in idx[:10]] for m, idx in rankings.items()},
                          "labels": {m: [label_of[docs[i]["doc_id"]] for i in idx[:10]] for m, idx in rankings.items()}})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(labels)}  {time.time() - t0:.0f}s", flush=True)

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-esci-setting-a-{args.split}" + ("-dirty" if dirty else "")
    out = ROOT / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "git_commit.txt").write_text(commit + ("\n(dirty working tree)" if dirty else "") + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(
        {"argv": sys.argv, "split": args.split, "limit": args.limit, "rerank_pool": args.rerank_pool,
         "dense_model": MODEL, "dense_max_length": 256, "reranker": RERANKER, "bm25_index": "indexes/esci_v1_bm25",
         "hybrid": "0.5 * minmax(bm25) + 0.5 * minmax(dense)"}, indent=2, ensure_ascii=False), encoding="utf-8")
    results = {}
    print(f"\n{'method':16s} {'nDCG@10':>8s} {'nDCG@10(no C)':>14s} {'Recall@10':>10s} {'MRR':>6s}")
    for m in records:
        main_gain = aggregate(records[m], GAIN)
        no_c = aggregate(records[m], GAIN_NO_COMPLEMENT)
        results[m] = {**main_gain, "ndcg@10_no_complement": no_c["ndcg@10"],
                      "diagnostic_only": m == "oracle"}
        print(f"{m:16s} {main_gain['ndcg@10']:8.4f} {no_c['ndcg@10']:14.4f} {main_gain['recall@10']:10.4f} {main_gain['mrr']:6.3f}")
    (out / "metrics.json").write_text(json.dumps(
        {"split": args.split, "num_queries": len(labels), "setting": "A: fixed annotated candidates",
         "corpus_version": "esci_us_small_v1", "gain_mapping": GAIN, "methods": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
