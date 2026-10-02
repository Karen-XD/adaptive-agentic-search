"""向量检索、RRF 融合、服务按 method 分发。

RRF 和服务分发用假检索器测，不依赖模型；向量检索用真模型在小语料上现场建索引（CPU），模型不在就跳过。
运行：pytest tests/test_dense_hybrid.py
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.schema import Doc
from retrieval.hybrid import HybridSearchTool, rrf_fuse
from retrieval.server import create_app

E5_PATH = Path("/root/autodl-tmp/hf_models/e5-base-v2")


def _docs(ids: list[str], source: str) -> list[Doc]:
    return [Doc(doc_id=i, title=i, text=i, score=1.0 / r, rank=r, source=source) for r, i in enumerate(ids, 1)]


class FakeTool:
    def __init__(self, ranking: list[str], source: str, corpus: list[str]):
        self.ranking, self.source = ranking, source
        self.docs = [{"doc_id": i} for i in corpus]
        self.meta = {"source": source}
        self.calls: list[int] = []

    def search(self, query: str, top_k: int) -> list[Doc]:
        self.calls.append(top_k)
        return _docs(self.ranking[:top_k], self.source)


def test_rrf_rewards_docs_found_by_both():
    # b 在两路都是第 2 名，a 只在 BM25 排第 1：1/62 + 1/62 > 1/61，两路都召回的应该排前面
    out = rrf_fuse([_docs(["a", "b"], "bm25"), _docs(["c", "b"], "dense")], top_k=3, k=60)
    assert [d.doc_id for d in out] == ["b", "a", "c"]
    assert out[0].score == pytest.approx(2 / 62)
    assert [d.rank for d in out] == [1, 2, 3] and all(d.source == "hybrid" for d in out)


def test_rrf_ties_broken_by_doc_id():
    # a 和 c 都只在一路排第 1，同分：按 doc_id 排，不偏向哪一路
    out = rrf_fuse([_docs(["c"], "bm25"), _docs(["a"], "dense")], top_k=2)
    assert [d.doc_id for d in out] == ["a", "c"]


def test_rrf_handles_one_empty_ranking():
    # BM25 对纯停用词查询返回空，hybrid 应该退化成只用向量那一路
    out = rrf_fuse([[], _docs(["x", "y"], "dense")], top_k=5)
    assert [d.doc_id for d in out] == ["x", "y"]


def test_hybrid_uses_pool_size_per_retriever():
    corpus = list("abcdefgh")
    bm25, dense = FakeTool(list("abcdefgh"), "bm25", corpus), FakeTool(list("hgfedcba"), "dense", corpus)
    out = HybridSearchTool(bm25, dense, pool_size=6).search("q", 3)
    assert bm25.calls == [6] and dense.calls == [6] and len(out) == 3
    HybridSearchTool(bm25, dense, pool_size=2).search("q", 5)  # top_k 比候选池大时，候选池至少取 top_k
    assert bm25.calls[-1] == 5


def test_hybrid_rejects_different_corpora():
    with pytest.raises(ValueError):
        HybridSearchTool(FakeTool([], "bm25", ["a", "b"]), FakeTool([], "dense", ["b", "a"]))


def test_service_dispatches_by_method():
    corpus = ["a", "b", "c"]
    bm25, dense = FakeTool(["a", "b"], "bm25", corpus), FakeTool(["c", "b"], "dense", corpus)
    client = TestClient(create_app({"bm25": bm25, "dense": dense, "hybrid": HybridSearchTool(bm25, dense)}))
    assert set(client.get("/health").json()["methods"]) == {"bm25", "dense", "hybrid"}
    for method, top1 in [("bm25", "a"), ("dense", "c"), ("hybrid", "b")]:
        r = client.post("/search", json={"query": "q", "top_k": 2, "method": method}).json()
        assert r["method"] == method and r["docs"][0]["doc_id"] == top1
    assert client.post("/search", json={"query": "q", "top_k": 2}).json()["method"] == "bm25"  # V1 配置不传 method


def test_service_rejects_unloaded_or_unknown_method():
    client = TestClient(create_app({"bm25": FakeTool(["a"], "bm25", ["a"])}))
    assert client.post("/search", json={"query": "q", "top_k": 1, "method": "dense"}).status_code == 400
    assert client.post("/search", json={"query": "q", "top_k": 1, "method": "splade"}).status_code == 422


@pytest.fixture(scope="module")
def dense_tool(tmp_path_factory):
    if not E5_PATH.exists():
        pytest.skip("e5-base-v2 not downloaded")
    from retrieval.dense import DenseSearchTool, build_index
    corpus = [
        {"doc_id": "d1", "title": "Tessa Marrow", "text": "Tessa Marrow is an engineer who was born in Port Edvik."},
        {"doc_id": "d2", "title": "Luminara Labs", "text": "Luminara Labs is a company that makes telescopes."},
        {"doc_id": "d3", "title": "Port Edvik", "text": "Port Edvik is a harbor town with a lighthouse."},
    ]
    d = tmp_path_factory.mktemp("dense")
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(x) for x in corpus), encoding="utf-8")
    build_index(d / "corpus.jsonl", d / "index", E5_PATH, device="cpu")
    return DenseSearchTool(d / "index", device="cpu")


def test_dense_matches_paraphrase(dense_tool):
    # 查询和段落没有共同的实词（maker / optical instruments vs makes telescopes）：BM25 搜不到，向量检索应该能
    docs = dense_tool.search("Which firm is a maker of optical instruments for astronomy?", 3)
    assert docs[0].doc_id == "d2" and [d.rank for d in docs] == [1, 2, 3]
    assert all(d.source == "dense" for d in docs) and docs[0].score > docs[1].score


def test_dense_same_query_same_results(dense_tool):
    assert dense_tool.search("lighthouse town", 3) == dense_tool.search("lighthouse town", 3)


@pytest.mark.parametrize("query", ["", "   "])
def test_dense_empty_query_returns_empty(dense_tool, query):
    assert dense_tool.search(query, 3) == []
