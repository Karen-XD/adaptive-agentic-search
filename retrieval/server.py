"""检索服务：把检索器包成独立进程，Agent 通过 HTTP 调用。

为什么单独起服务：索引只加载一次，多个实验共用；检索和 Agent 解耦，换检索器不影响 Agent；
记录的耗时是真实的接口耗时。和线上召回服务独立部署是同一个思路。

三种检索走同一个 /search，用 method 区分；不传 method 就是 bm25，V1 的配置原样可复现。
rerank=true 时先按 method 取候选池，再用 Cross-Encoder 重排；检索器和"要不要重排"是两个独立的选择。
启动（放 tmux 里）：
  只有 BM25：python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --port 8100
  三路都开：再加 --dense-index indexes/hotpot_pool_v1_e5（dense 和 hybrid 才可用）
  加重排：再加 --reranker /root/autodl-tmp/hf_models/bge-reranker-base（默认放 GPU，fp16 约 0.6GB 显存）
"""
from __future__ import annotations

import argparse
import time
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.schema import Doc
from retrieval.bm25 import BM25SearchTool

Method = Literal["bm25", "dense", "hybrid"]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(ge=1, le=100)
    method: Method = "bm25"
    rerank: bool = False


class SearchResponse(BaseModel):
    docs: list[Doc]
    method: str
    rerank: bool
    latency_ms: float  # 服务端检索耗时；客户端另记含网络的总耗时
    timing: Optional[dict] = None  # 重排时拆成 retrieve_ms / rerank_ms / num_candidates


def create_app(tools: dict, reranked: Optional[dict] = None) -> FastAPI:
    """tools: method -> 检索器，各自实现 search(query, top_k) 和 meta。
    reranked: method -> 包了重排的同一个检索器（retrieval/rerank.py）；没加载重排模型时为空。"""
    app = FastAPI()
    reranked = reranked or {}

    @app.get("/health")
    def health() -> dict:
        # 返回每种检索的索引信息：跑实验时写进 config.yaml，确认连的是哪个索引、哪个重排模型
        reranker = next(iter(reranked.values())).meta["reranker"] if reranked else None
        return {"status": "ok", "methods": {m: t.meta for m, t in tools.items()}, "reranker": reranker,
                "rerank_pool_size": next(iter(reranked.values())).pool_size if reranked else None}

    @app.post("/search")
    def search(req: SearchRequest) -> SearchResponse:
        if req.method not in tools:  # 没加载就明确报错，不悄悄退回 BM25，否则实验对比会失真
            raise HTTPException(400, f"method {req.method!r} not loaded; available: {sorted(tools)}")
        if req.rerank and req.method not in reranked:  # 同理，不悄悄跳过重排
            raise HTTPException(400, "reranker not loaded")
        t0 = time.perf_counter()
        timing = None
        if req.rerank:
            docs, timing = reranked[req.method].search_timed(req.query, req.top_k)
        else:
            docs = tools[req.method].search(req.query, req.top_k)
        return SearchResponse(docs=docs, method=req.method, rerank=req.rerank,
                              latency_ms=(time.perf_counter() - t0) * 1000, timing=timing)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="BM25 索引目录")
    ap.add_argument("--dense-index", help="向量索引目录；给了才提供 dense 和 hybrid")
    # 查询编码默认放 CPU：单条约 10ms，和 BM25 同一量级；不占显存，不和 vLLM 抢 GPU
    ap.add_argument("--dense-device", default="cpu")
    ap.add_argument("--pool-size", type=int, default=20, help="hybrid 每一路取的候选数")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--reranker", help="Cross-Encoder 重排模型目录；给了才支持 rerank=true")
    # 重排放 GPU：20 个候选过一遍 base 模型，CPU 上要几百毫秒。Agent 是单并发串行，重排和 vLLM 生成不会同时跑
    ap.add_argument("--rerank-device", default="cuda")
    ap.add_argument("--rerank-pool-size", type=int, default=20, help="重排前从检索器取的候选数")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    tools: dict = {"bm25": BM25SearchTool(args.index)}
    if args.dense_index:
        from retrieval.dense import DenseSearchTool  # 只开 BM25 时不加载 torch / faiss
        from retrieval.hybrid import HybridSearchTool
        tools["dense"] = DenseSearchTool(args.dense_index, device=args.dense_device)
        tools["hybrid"] = HybridSearchTool(tools["bm25"], tools["dense"], args.pool_size, args.rrf_k)
    reranked: dict = {}
    if args.reranker:
        from retrieval.rerank import CrossEncoderReranker, RerankedSearchTool
        reranker = CrossEncoderReranker(args.reranker, device=args.rerank_device)
        reranked = {m: RerankedSearchTool(t, reranker, args.rerank_pool_size) for m, t in tools.items()}
    # 单进程即可：检索在毫秒级，瓶颈在模型生成
    uvicorn.run(create_app(tools, reranked), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
