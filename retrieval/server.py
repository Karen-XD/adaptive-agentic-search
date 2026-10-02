"""检索服务：把检索器包成独立进程，Agent 通过 HTTP 调用。

为什么单独起服务：索引只加载一次，多个实验共用；检索和 Agent 解耦，换检索器不影响 Agent；
记录的耗时是真实的接口耗时。和线上召回服务独立部署是同一个思路。

三种检索走同一个 /search，用 method 区分；不传 method 就是 bm25，V1 的配置原样可复现。
启动（放 tmux 里）：
  只有 BM25：python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --port 8100
  三路都开：再加 --dense-index indexes/hotpot_pool_v1_e5（dense 和 hybrid 才可用）
"""
from __future__ import annotations

import argparse
import time
from typing import Literal

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


class SearchResponse(BaseModel):
    docs: list[Doc]
    method: str
    latency_ms: float  # 服务端检索耗时；客户端另记含网络的总耗时


def create_app(tools: dict) -> FastAPI:
    """tools: method -> 检索器，各自实现 search(query, top_k) 和 meta。"""
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        # 返回每种检索的索引信息：跑实验时写进 config.yaml，确认连的是哪个索引
        return {"status": "ok", "methods": {m: t.meta for m, t in tools.items()}}

    @app.post("/search")
    def search(req: SearchRequest) -> SearchResponse:
        if req.method not in tools:  # 没加载就明确报错，不悄悄退回 BM25，否则实验对比会失真
            raise HTTPException(400, f"method {req.method!r} not loaded; available: {sorted(tools)}")
        t0 = time.perf_counter()
        docs = tools[req.method].search(req.query, req.top_k)
        return SearchResponse(docs=docs, method=req.method, latency_ms=(time.perf_counter() - t0) * 1000)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="BM25 索引目录")
    ap.add_argument("--dense-index", help="向量索引目录；给了才提供 dense 和 hybrid")
    # 查询编码默认放 CPU：单条约 10ms，和 BM25 同一量级；不占显存，不和 vLLM 抢 GPU
    ap.add_argument("--dense-device", default="cpu")
    ap.add_argument("--pool-size", type=int, default=20, help="hybrid 每一路取的候选数")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    tools: dict = {"bm25": BM25SearchTool(args.index)}
    if args.dense_index:
        from retrieval.dense import DenseSearchTool  # 只开 BM25 时不加载 torch / faiss
        from retrieval.hybrid import HybridSearchTool
        tools["dense"] = DenseSearchTool(args.dense_index, device=args.dense_device)
        tools["hybrid"] = HybridSearchTool(tools["bm25"], tools["dense"], args.pool_size, args.rrf_k)
    # 单进程即可：检索在毫秒级，瓶颈在模型生成
    uvicorn.run(create_app(tools), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
