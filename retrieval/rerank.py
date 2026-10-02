"""重排（Rerank）：先用便宜的检索器取一个候选池（默认 20 条），再用 Cross-Encoder 逐条打分，按新分数取前 top_k。

Cross-Encoder 和向量检索（双塔）的区别：双塔把查询和段落分开编码，段落向量能离线算好，所以能在 50 万段里检索；
Cross-Encoder 把（查询, 段落）拼成一条输入一起过模型，每个词都能看到对方，判断更准，但每个候选都要过一遍模型，
只能用在几十条候选上。搜广推里的召回 → 精排是同一个分工：召回求全、精排求准。

RerankedSearchTool 包住任意一个检索器，接口不变：search(query, top_k) -> list[Doc]。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agent.schema import Doc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CrossEncoderReranker:
    """bge-reranker 系列：输入（查询, 段落）对，输出一个相关性 logit，越大越相关。"""

    def __init__(self, model_path: str | Path, device: str | None = None, fp16: bool | None = None,
                 max_length: int = 512, batch_size: int = 32):
        model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        fp16 = self.device != "cpu" if fp16 is None else fp16  # GPU 默认 fp16，CPU 上 fp16 反而慢
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.float16 if fp16 else torch.float32).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.meta = {"model_path": str(model_path), "model_sha256": _sha256(model_path / "model.safetensors"),
                     "device": self.device, "dtype": "fp16" if fp16 else "fp32", "max_length": max_length}

    @torch.inference_mode()
    def score(self, query: str, docs: list[Doc]) -> list[float]:
        # 段落和 BM25 / 向量检索一样带标题；超长时只截段落，查询保持完整
        pairs = [(query, f"{d.title}\n{d.text}") for d in docs]
        scores: list[float] = []
        for s in range(0, len(pairs), self.batch_size):
            batch = self.tokenizer(pairs[s:s + self.batch_size], padding=True, truncation="only_second",
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            scores += self.model(**batch).logits[:, 0].float().cpu().tolist()
        return scores


class RerankedSearchTool:
    def __init__(self, base, reranker: CrossEncoderReranker, pool_size: int = 20):
        self.base, self.reranker, self.pool_size = base, reranker, pool_size
        self.source = f"{base.source}+rerank"
        self.meta = {"base": base.meta, "reranker": reranker.meta, "pool_size": pool_size}

    def search_timed(self, query: str, top_k: int) -> tuple[list[Doc], dict]:
        """返回结果和耗时拆分：检索和重排的成本要分开记（重排是 V2 策略要决定"做不做"的那一步）。"""
        t0 = time.perf_counter()
        pool = self.base.search(query, max(self.pool_size, top_k))
        t1 = time.perf_counter()
        scores = self.reranker.score(query, pool) if pool else []
        t2 = time.perf_counter()
        # 同分按 doc_id：fp16 下分数精度有限，几乎重复的段落可能同分
        ranked = sorted(zip(scores, pool), key=lambda x: (-x[0], x[1].doc_id))
        docs = [d.model_copy(update={"score": s, "rank": r + 1, "source": self.source})
                for r, (s, d) in enumerate(ranked[:top_k])]
        return docs, {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000, "num_candidates": len(pool)}

    def search(self, query: str, top_k: int) -> list[Doc]:
        return self.search_timed(query, top_k)[0]
