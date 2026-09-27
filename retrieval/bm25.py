"""BM25 检索器：接口与 retrieval/mock.py 一致，search(query, top_k) -> list[Doc]。

建索引：python -m retrieval.bm25 build --corpus data/hotpotqa/v1/corpus.jsonl --index indexes/hotpot_pool_v1_bm25
BM25 是词匹配打分：查询词在段落里出现得越多、越稀有，分数越高。擅长人名、地名等精确匹配，不懂同义改写。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bm25s
import Stemmer

from agent.schema import Doc

# 词干化：born / borns、running / run 归成同一个词根，Pyserini 等常见实现默认也这样做
_STEMMER = Stemmer.Stemmer("english")
_TIE_MARGIN = 10  # 多取几条再按 (分数, doc_id) 排序，同分时顺序固定


def _tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)


def build_index(corpus_path: Path, index_dir: Path) -> None:
    with open(corpus_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    retriever = bm25s.BM25()  # 默认参数 k1=1.5, b=0.75
    # 标题也参与匹配：HotpotQA 里标题就是实体名，是最强的检索信号
    retriever.index(_tokenize([f"{d['title']}\n{d['text']}" for d in docs]), show_progress=False)
    retriever.save(index_dir)
    with open(index_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta = {"corpus_path": str(corpus_path), "num_docs": len(docs), "k1": 1.5, "b": 0.75,
            "tokenizer": "bm25s en stopwords + snowball english stemmer", "build_seconds": round(time.time() - t0, 1)}
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


class BM25SearchTool:
    source = "bm25"

    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        self.retriever = bm25s.BM25.load(index_dir)
        with open(index_dir / "docs.jsonl", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int) -> list[Doc]:
        tokens = _tokenize([query])
        if not tokens.vocab:  # 全是停用词或符号：没有可匹配的词，返回空结果而不是报错
            return []
        k = min(top_k + _TIE_MARGIN, len(self.docs))
        idx, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        hits = [(float(s), int(i)) for i, s in zip(idx[0], scores[0]) if s > 0]
        hits.sort(key=lambda x: (-x[0], self.docs[x[1]]["doc_id"]))
        return [Doc(doc_id=self.docs[i]["doc_id"], title=self.docs[i]["title"], text=self.docs[i]["text"],
                    score=s, rank=r + 1, source=self.source)
                for r, (s, i) in enumerate(hits[:top_k])]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", required=True)
    b.add_argument("--index", required=True)
    args = ap.parse_args()
    Path(args.index).mkdir(parents=True, exist_ok=True)
    build_index(Path(args.corpus), Path(args.index))


if __name__ == "__main__":
    main()
