"""向量检索器（Dense）：接口与 retrieval/bm25.py 一致，search(query, top_k) -> list[Doc]。

建索引（要 GPU，约 50 万段几分钟；vLLM 占着显存时先停掉）：
  python -m retrieval.dense build --corpus data/hotpotqa/v1/corpus.jsonl --index indexes/hotpot_pool_v1_e5 \
      --model /root/autodl-tmp/hf_models/e5-base-v2
向量检索把查询和段落都编码成向量，按余弦相似度排序。擅长同义改写、语义描述；人名、型号这类要逐字匹配的词不如 BM25。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from agent.schema import Doc

# e5 的约定（见模型卡）：查询和段落要加不同前缀，不加效果会明显下降
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
_TIE_MARGIN = 10  # 和 BM25 一样多取几条再按 (分数, doc_id) 排序，同分时顺序固定
_INDEX_FILE = "faiss_flat_ip.index"


def passage_text(doc: dict) -> str:
    # 和 BM25 一样把标题拼进去：HotpotQA 里标题就是实体名
    return f"{PASSAGE_PREFIX}{doc['title']}\n{doc['text']}"


class E5Encoder:
    """mean pooling（所有 token 向量取平均，padding 不算）+ L2 归一化：归一化后内积就是余弦相似度。"""

    def __init__(self, model_path: str | Path, device: str | None = None, fp16: bool = False,
                 max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
        hidden = self.model(**batch).last_hidden_state.float()  # 转 fp32 再求和，fp16 下长段落求和有溢出风险
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(corpus_path: Path, index_dir: Path, model_path: Path, batch_size: int = 256,
                max_length: int = 512, device: str = "cuda") -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    texts = [passage_text(d) for d in docs]
    t0 = time.time()
    # 段落编码只做一次，用 GPU + fp16；查询编码在服务里用 fp32（见 DenseSearchTool）。device=cpu 只给测试用
    enc = E5Encoder(model_path, device=device, fp16=device != "cpu", max_length=max_length)
    # 按长度排序再分批：同一批长度相近，padding 少；结果按原顺序写回，第 i 行向量对应第 i 个段落
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    emb = np.empty((len(texts), enc.dim), dtype=np.float32)
    for n, s in enumerate(range(0, len(texts), batch_size)):
        idx = order[s:s + batch_size]
        emb[idx] = enc.encode([texts[i] for i in idx])
        if n % 200 == 0:
            print(f"encoded {s + len(idx)}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    encode_seconds = time.time() - t0
    index = faiss.IndexFlatIP(enc.dim)  # 精确内积检索：50 万段 CPU 暴力搜也只要几十毫秒，没有近似带来的召回损失
    index.add(emb)
    faiss.write_index(index, str(index_dir / _INDEX_FILE))
    with open(index_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta = {"corpus_path": str(corpus_path), "corpus_sha256": _sha256(corpus_path), "num_docs": len(docs),
            "model_path": str(model_path), "model_sha256": _sha256(Path(model_path) / "model.safetensors"),
            "dim": enc.dim, "max_length": max_length, "pooling": "mean", "normalize": True,
            "prefixes": {"query": QUERY_PREFIX, "passage": PASSAGE_PREFIX},
            "passage_dtype": "fp16" if device != "cpu" else "fp32",
            "faiss": "IndexFlatIP", "encode_seconds": round(encode_seconds, 1),
            "build_seconds": round(time.time() - t0, 1)}
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


class DenseSearchTool:
    source = "dense"

    def __init__(self, index_dir: str | Path, device: str | None = None):
        index_dir = Path(index_dir)
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / _INDEX_FILE))
        with open(index_dir / "docs.jsonl", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        if self.index.ntotal != len(self.docs):  # 向量行号就是段落下标，数量对不上说明映射已经错了
            raise ValueError(f"index has {self.index.ntotal} vectors but {len(self.docs)} docs")
        # 查询用 fp32：单条查询很快，fp32 结果稳定，换机器重跑能逐位复现
        self.encoder = E5Encoder(self.meta["model_path"], device=device, max_length=self.meta["max_length"])

    def search(self, query: str, top_k: int) -> list[Doc]:
        if not query.strip():  # 向量检索对空串也会返回结果，和 BM25 保持一致：空查询返回空
            return []
        q = self.encoder.encode([QUERY_PREFIX + query])
        k = min(top_k + _TIE_MARGIN, len(self.docs))
        scores, idx = self.index.search(q, k)
        hits = [(float(s), int(i)) for s, i in zip(scores[0], idx[0]) if i >= 0]
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
    b.add_argument("--model", required=True)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    Path(args.index).mkdir(parents=True, exist_ok=True)
    build_index(Path(args.corpus), Path(args.index), Path(args.model), args.batch_size, args.max_length)


if __name__ == "__main__":
    main()
