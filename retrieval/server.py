"""检索服务：把检索器包成独立进程，Agent 通过 HTTP 调用。

为什么单独起服务：索引只加载一次，多个实验共用；检索和 Agent 解耦，之后换向量检索（要占 GPU）不影响 Agent；
记录的耗时是真实的接口耗时。和线上召回服务独立部署是同一个思路。

启动（放 tmux 里）：python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --port 8100
"""
from __future__ import annotations

import argparse
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.schema import Doc
from retrieval.bm25 import BM25SearchTool


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(ge=1, le=100)


class SearchResponse(BaseModel):
    docs: list[Doc]
    latency_ms: float  # 服务端检索耗时；客户端另记含网络的总耗时


def create_app(tool: BM25SearchTool) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        # 返回索引信息：跑实验时写进 config.yaml，确认连的是哪个索引
        return {"status": "ok", "source": tool.source, "index": tool.meta}

    @app.post("/search")
    def search(req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        docs = tool.search(req.query, req.top_k)
        return SearchResponse(docs=docs, latency_ms=(time.perf_counter() - t0) * 1000)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    # 单进程即可：BM25 查询在毫秒级，瓶颈在模型生成
    uvicorn.run(create_app(BM25SearchTool(args.index)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
