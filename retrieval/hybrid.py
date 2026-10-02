"""混合检索（Hybrid）：BM25 和向量检索各取一个候选池，用 RRF 按名次融合。接口同样是 search(query, top_k) -> list[Doc]。

为什么按名次不按分数：BM25 分数没有上界（几到几十），归一化后的余弦相似度挤在 0.7～0.9，量纲不同，直接相加等于让 BM25 说了算；
要按分数加权得先做校准。RRF（Reciprocal Rank Fusion）只看每一路里的名次，不需要校准，是多路召回融合最常用的无参基线：
  RRF(d) = Σ_r 1 / (k + rank_r(d))，没被某一路召回的文档在那一路记 0。
k=60 沿用 Cormack et al. 2009：k 越大，头部名次之间的差距越小，越看重"几路都召回了"。
"""
from __future__ import annotations

from agent.schema import Doc
from retrieval.bm25 import BM25SearchTool
from retrieval.dense import DenseSearchTool


def rrf_fuse(rankings: list[list[Doc]], top_k: int, k: int = 60) -> list[Doc]:
    scores: dict[str, float] = {}
    first: dict[str, Doc] = {}
    for docs in rankings:
        for d in docs:
            scores[d.doc_id] = scores.get(d.doc_id, 0.0) + 1.0 / (k + d.rank)
            first.setdefault(d.doc_id, d)
    # 同分按 doc_id：doc_id 是标题哈希，相当于随机但固定的顺序，不系统性偏向哪一路
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [first[i].model_copy(update={"score": scores[i], "rank": r + 1, "source": "hybrid"})
            for r, i in enumerate(order[:top_k])]


class HybridSearchTool:
    source = "hybrid"

    def __init__(self, bm25: BM25SearchTool, dense: DenseSearchTool, pool_size: int = 20, k: int = 60):
        # 两路必须建在同一份语料、同一顺序上，按 doc_id 融合才有意义；建索引后语料变了这里会直接报错
        if [d["doc_id"] for d in bm25.docs] != [d["doc_id"] for d in dense.docs]:
            raise ValueError("bm25 and dense indexes were built on different corpora")
        self.bm25, self.dense, self.pool_size, self.k = bm25, dense, pool_size, k
        self.meta = {"bm25": bm25.meta, "dense": dense.meta, "fusion": "rrf", "rrf_k": k, "pool_size": pool_size}

    def search(self, query: str, top_k: int) -> list[Doc]:
        # 两路串行调用，耗时是两路之和；线上会并行，那时耗时接近较慢的一路
        pool = max(self.pool_size, top_k)
        return rrf_fuse([self.bm25.search(query, pool), self.dense.search(query, pool)], top_k, self.k)
