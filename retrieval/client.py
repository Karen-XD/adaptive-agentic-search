"""检索服务的客户端：接口与本地检索器一致，search(query, top_k) -> list[Doc]。

超时、连接失败、服务端报错都直接抛异常，由 Agent 循环记成 tool_error（执行层），不在这里吞掉。
"""
from __future__ import annotations

import httpx

from agent.schema import Doc


class HttpSearchTool:
    def __init__(self, url: str, timeout_s: float = 5.0):
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def health(self) -> dict:
        r = self.client.get(f"{self.url}/health")
        r.raise_for_status()
        return r.json()

    def search(self, query: str, top_k: int) -> list[Doc]:
        r = self.client.post(f"{self.url}/search", json={"query": query, "top_k": top_k})
        r.raise_for_status()
        return [Doc(**d) for d in r.json()["docs"]]
