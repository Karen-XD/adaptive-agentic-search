"""BM25 检索器、检索服务和数据体检。用小语料现场建索引，不依赖大索引文件。

运行：pytest tests/ 或 python -m tests.test_retriever
"""
import json

import pytest
from fastapi.testclient import TestClient

from data_prep.prepare_hotpot import build_corpus, doc_id_of
from data_prep.validate_splits import validate
from retrieval.bm25 import BM25SearchTool, build_index
from retrieval.server import create_app

CORPUS = [
    {"doc_id": "d1", "title": "Tessa Marrow", "text": "Tessa Marrow is an engineer who was born in Port Edvik."},
    {"doc_id": "d2", "title": "Luminara Labs", "text": "Luminara Labs was founded by Tessa Marrow."},
    {"doc_id": "d3", "title": "Port Edvik", "text": "Port Edvik is a harbor town with a lighthouse."},
    {"doc_id": "d4", "title": "Running club", "text": "The club organizes weekly runs."},
    # d5 和 d6 长度相同、与查询匹配的词相同，BM25 同分：用来检查同分时顺序固定
    # （BM25 会惩罚长段落，两者长度不同就不会同分）
    {"doc_id": "d6", "title": "Glass bridge one", "text": "A bridge made of glass."},
    {"doc_id": "d5", "title": "Glass bridge two", "text": "A bridge made of glass."},
]


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx")
    corpus = d / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(x) for x in CORPUS), encoding="utf-8")
    build_index(corpus, d / "index")
    return BM25SearchTool(d / "index")


def test_finds_matching_doc_first(tool):
    docs = tool.search("Where was Tessa Marrow born?", 3)
    assert docs[0].doc_id == "d1" and [d.rank for d in docs] == list(range(1, len(docs) + 1))
    assert all(d.source == "bm25" for d in docs)


def test_same_query_same_results(tool):
    assert tool.search("Tessa Marrow", 3) == tool.search("Tessa Marrow", 3)


def test_ties_broken_by_doc_id(tool):
    ids = [d.doc_id for d in tool.search("bridge made of glass", 2)]
    assert ids == ["d5", "d6"]


def test_stemming_matches_word_variants(tool):
    assert tool.search("running", 1)[0].doc_id == "d4"  # running ~ runs


@pytest.mark.parametrize("query", ["", "   ", "the of and", "???"])
def test_empty_or_stopword_query_returns_empty(tool, query):
    assert tool.search(query, 3) == []


def test_no_match_returns_empty(tool):
    assert tool.search("qwertyuiop", 3) == []


def test_service_matches_local_tool(tool):
    client = TestClient(create_app(tool))
    assert client.get("/health").json()["index"]["num_docs"] == len(CORPUS)
    r = client.post("/search", json={"query": "Luminara Labs founder", "top_k": 2}).json()
    assert [d["doc_id"] for d in r["docs"]] == [d.doc_id for d in tool.search("Luminara Labs founder", 2)]
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422  # 参数非法


def test_corpus_dedup_same_title_keeps_most_frequent():
    import pandas as pd
    ctx = lambda texts: {"title": ["A"] * len(texts), "sentences": [[t] for t in texts]}
    df = pd.DataFrame({"context": [ctx(["old"]), ctx(["new"]), ctx(["new"])]})
    corpus = build_corpus([df])
    assert corpus == [{"doc_id": doc_id_of("A"), "title": "A", "text": "new"}]
    assert doc_id_of("A") == doc_id_of("A") and doc_id_of("A") != doc_id_of("B")  # ID 只由标题决定，重建不变


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_validate_splits_catches_problems(tmp_path):
    _write(tmp_path / "corpus.jsonl", [{"doc_id": "d1"}, {"doc_id": "d2"}])
    _write(tmp_path / "questions" / "a.jsonl", [{"qid": "q1", "question": "?"}])
    _write(tmp_path / "labels" / "a.jsonl", [{"qid": "q1", "gold_doc_ids": ["d1"]}])
    assert validate(tmp_path) == []
    # 题目文件混进答案、两个划分有重叠、金标段落不在语料里：都要被抓到
    _write(tmp_path / "questions" / "b.jsonl", [{"qid": "q1", "question": "?", "answer": "x"}])
    _write(tmp_path / "labels" / "b.jsonl", [{"qid": "q1", "gold_doc_ids": ["d9"]}])
    problems = " | ".join(validate(tmp_path))
    assert "多余字段" in problems and "重叠" in problems and "不在语料里" in problems


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
