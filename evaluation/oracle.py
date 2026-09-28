"""Oracle context 诊断：答题前读答案文件里的金标段落，交给 agent/methods.py 放进提示词。

这是防泄漏规则的唯一例外（CLAUDE.md，用户 2026-09-28 同意）：
- 目的：衡量"证据完美时模型能答对多少"，把检索损失和阅读损失拆开；它不是一个方法，不进方法对比
- 代码放在 evaluation/（评测侧），agent/ 里的代码仍然读不到答案文件
- test 集直接拒绝：测试集的标签只在最后评测时读一次
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.schema import Doc

ALLOWED_SPLITS = ("validation", "debug")


def load_gold_docs(data_dir: Path, split: str) -> dict[str, list[Doc]]:
    """qid → 金标段落。按 doc_id 排序：标签里的顺序是 HotpotQA 标注顺序（常是第一跳在前），会泄漏推理路径。"""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"oracle context is a diagnostic; only allowed on {ALLOWED_SPLITS}, got {split!r}")
    labels = [json.loads(line) for line in (data_dir / "labels" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    wanted = {doc_id for label in labels for doc_id in label["gold_doc_ids"]}
    corpus: dict[str, dict] = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["doc_id"] in wanted:
                corpus[doc["doc_id"]] = doc
    missing = wanted - corpus.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold doc ids not in corpus, e.g. {sorted(missing)[:3]}")
    return {label["qid"]: [Doc(doc_id=d, title=corpus[d]["title"], text=corpus[d]["text"], score=0.0, rank=i,
                               source="oracle")
                           for i, d in enumerate(sorted(label["gold_doc_ids"]), 1)]
            for label in labels}
