"""检索指纹：迁移前用旧索引存下一批查询的检索结果，新机器重建索引后对比，确认检索行为没有变。

为什么不直接比索引文件：BM25 重建后词表编号会变（Python 哈希随机化），字节不同但检索结果完全相同（2026-10-05 实测
300/300 一致）；向量索引在另一种显卡上用 fp16 重新编码，数值可能有细微差别，近似并列的段落可能换位置。
所以比"检索结果"而不是"文件字节"。

指纹内容：每个数据集 validation 的前 300 条查询（HotpotQA validation 只有 200 条），每条记录前 k 条的 doc_id 和分数。
  bm25 / dense：前 10 条；dense+rerank（= B3 实际用的检索栈：Dense 取 20 条候选 → bge 重排 → 前 3 条）：前 3 条

用法（在仓库根目录）：
  python scripts/migration/retrieval_fingerprint.py save  --out scripts/migration/retrieval_fingerprint.json
  python scripts/migration/retrieval_fingerprint.py check --ref scripts/migration/retrieval_fingerprint.json
check 的判定：BM25 必须 100% 一致（否则退出码 1）；dense / dense+rerank 报告一致率，低于 95% 时退出码 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RERANKER = "/root/autodl-tmp/hf_models/bge-reranker-base"
N_QUERIES = 300
SPECS = [  # (名字, 索引类型, 索引目录, 查询文件, k)
    ("hotpotqa_bm25", "bm25", "indexes/hotpot_pool_v1_bm25", "data/hotpotqa/v1/questions/validation.jsonl", 10),
    ("hotpotqa_dense", "dense", "indexes/hotpot_pool_v1_e5", "data/hotpotqa/v1/questions/validation.jsonl", 10),
    ("hotpotqa_dense_rerank", "dense_rerank", "indexes/hotpot_pool_v1_e5", "data/hotpotqa/v1/questions/validation.jsonl", 3),
    ("2wiki_bm25", "bm25", "indexes/2wiki_pool_v1_bm25", "data/2wiki/v1/questions/validation.jsonl", 10),
    ("2wiki_dense", "dense", "indexes/2wiki_pool_v1_e5", "data/2wiki/v1/questions/validation.jsonl", 10),
    ("2wiki_dense_rerank", "dense_rerank", "indexes/2wiki_pool_v1_e5", "data/2wiki/v1/questions/validation.jsonl", 3),
    ("esci_bm25", "bm25", "indexes/esci_v1_bm25", "data/esci/v1/questions/validation.jsonl", 10),
]


def run(spec_filter: set[str] | None = None) -> dict:
    from retrieval.bm25 import BM25SearchTool
    from retrieval.dense import DenseSearchTool
    from retrieval.rerank import CrossEncoderReranker, RerankedSearchTool

    out, dense_cache, reranker = {}, {}, None
    for name, kind, index, qfile, k in SPECS:
        if spec_filter and name not in spec_filter:
            continue
        queries = [json.loads(l)["question"] for l in open(ROOT / qfile, encoding="utf-8")][:N_QUERIES]
        if kind == "bm25":
            tool = BM25SearchTool(ROOT / index)
        else:
            if index not in dense_cache:
                dense_cache[index] = DenseSearchTool(ROOT / index)
            tool = dense_cache[index]
            if kind == "dense_rerank":
                reranker = reranker or CrossEncoderReranker(RERANKER)
                tool = RerankedSearchTool(tool, reranker, pool_size=20)
        rows = []
        for q in queries:
            docs = tool.search(q, k)
            rows.append({"q": q, "ids": [d.doc_id for d in docs], "scores": [round(float(d.score), 5) for d in docs]})
        out[name] = {"kind": kind, "index": index, "k": k, "rows": rows}
        print(f"{name}: {len(rows)} 条查询", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["save", "check"])
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--only", nargs="*", help="只跑这几个（名字见 SPECS）")
    args = ap.parse_args()
    if args.mode == "save":
        Path(args.out).write_text(json.dumps(run(set(args.only or [])), ensure_ascii=False), encoding="utf-8")
        print(f"-> {args.out}")
        return
    ref = json.load(open(args.ref, encoding="utf-8"))
    cur = run(set(args.only or ref))
    failed = False
    print(f"\n{'指纹':24s} {'前 k 条完全一致':>14s} {'平均重合':>8s} {'最大分数差':>10s}  判定")
    for name, r in ref.items():
        if name not in cur:
            continue
        a, b = r["rows"], cur[name]["rows"]
        assert [x["q"] for x in a] == [x["q"] for x in b], f"{name}: 查询列表不同（数据划分变了？）"
        same = sum(x["ids"] == y["ids"] for x, y in zip(a, b)) / len(a)
        # 两边都没有结果（查询全是停用词）算完全重合
        overlap = sum(1.0 if not x["ids"] and not y["ids"] else len(set(x["ids"]) & set(y["ids"])) / max(1, len(x["ids"]))
                      for x, y in zip(a, b)) / len(a)
        maxdiff = max((abs(s - t) for x, y in zip(a, b) for s, t in zip(x["scores"], y["scores"])), default=0.0)
        need = 1.0 if r["kind"] == "bm25" else 0.95
        ok = same >= need
        failed |= not ok
        print(f"{name:24s} {same:14.1%} {overlap:8.1%} {maxdiff:10.2e}  {'通过' if ok else '不通过（要求 ≥ %.0f%%）' % (need * 100)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
