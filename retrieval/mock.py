"""确定性的假检索：固定小语料 + 词重叠打分。只用来测 Agent 循环能不能跑通，不衡量检索效果。

接口和 Day 2 的 BM25 服务保持一致：search(query, top_k) -> list[Doc]。
输入只有 query，拿不到题目 ID 和标准答案——防泄漏靠接口签名保证，不靠自觉。
"""
from __future__ import annotations

import re
from typing import Optional

from agent.schema import Doc

# 全是虚构实体：接真模型后它没法凭记忆作答，答对就说明确实用了检索结果。
# 两跳问题 "Where was the founder of Luminara Labs born?" 需要 001 → 002；004、005 是只沾边的干扰段落
MOCK_CORPUS = [
    {"doc_id": "mock-001", "title": "Luminara Labs",
     "text": "Luminara Labs is a robotics company founded in 2011 by the engineer Tessa Marrow."},
    {"doc_id": "mock-002", "title": "Tessa Marrow",
     "text": "Tessa Marrow is an engineer who was born in Port Edvik in 1975."},
    {"doc_id": "mock-003", "title": "Port Edvik",
     "text": "Port Edvik is a small harbor town known for its lighthouse."},
    {"doc_id": "mock-004", "title": "Luminara (novel)",
     "text": "Luminara is a 1998 fantasy novel about a city of lights."},
    {"doc_id": "mock-005", "title": "Marrow Industries",
     "text": "Marrow Industries is a steel company unrelated to Tessa Marrow."},
]

# 停用词在 BM25 里 IDF 很低、几乎不贡献分数；这里直接去掉，行为更接近真实检索
_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "and",
              "is", "was", "who", "which", "what"}


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS}


class MockSearchTool:
    source = "mock"

    def __init__(self, corpus: Optional[list[dict]] = None):
        self.corpus = MOCK_CORPUS if corpus is None else corpus
        self._doc_terms = [_terms(d["title"] + " " + d["text"]) for d in self.corpus]

    def search(self, query: str, top_k: int) -> list[Doc]:
        q = _terms(query)
        scored = [(len(q & terms), d) for d, terms in zip(self.corpus, self._doc_terms)]
        # 分数相同按 doc_id 排：同一个 query 的结果顺序永远一样；零分不返回，才会出现"空结果"
        hits = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]["doc_id"]))
        return [Doc(doc_id=d["doc_id"], title=d["title"], text=d["text"], score=float(s),
                    rank=i + 1, source=self.source)
                for i, (s, d) in enumerate(hits[:top_k])]
