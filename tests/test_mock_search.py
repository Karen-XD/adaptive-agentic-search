"""Mock 检索的确定性：同一个查询永远返回同样的结果和顺序。

运行：pytest tests/ 或 python -m tests.test_mock_search
"""
import pytest

from retrieval.mock import MockSearchTool


def test_same_query_same_results():
    tool = MockSearchTool()
    assert tool.search("Tessa Marrow born", 3) == tool.search("Tessa Marrow born", 3)


def test_ties_broken_by_doc_id():
    # mock-001 和 mock-004 都只命中 "luminara" 一个词，同分按 doc_id 排
    docs = MockSearchTool().search("Luminara", 5)
    assert [(d.doc_id, d.rank) for d in docs] == [("mock-001", 1), ("mock-004", 2)]


def test_top_k_respected():
    assert len(MockSearchTool().search("Tessa Marrow", 1)) == 1


def test_no_overlap_returns_empty():
    assert MockSearchTool().search("qwerty zzz", 3) == []
    assert MockSearchTool().search("who is the", 3) == []  # 只有停用词


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
