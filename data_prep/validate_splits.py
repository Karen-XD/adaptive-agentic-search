"""数据体检：划分之间题目不重叠、题目文件里没有答案、金标段落都在语料里、doc_id 唯一。

用法：python -m data_prep.validate_splits --data_dir data/hotpotqa/v1
任何一项不通过就以非零状态退出，不要带着有问题的数据去跑实验。
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(data_dir: Path) -> list[str]:
    problems = []
    corpus_ids = [d["doc_id"] for d in read_jsonl(data_dir / "corpus.jsonl")]
    if len(corpus_ids) != len(set(corpus_ids)):
        problems.append("corpus: doc_id 有重复")
    corpus_ids = set(corpus_ids)

    split_qids = {}
    for qfile in sorted((data_dir / "questions").glob("*.jsonl")):
        name = qfile.stem
        questions = read_jsonl(qfile)
        labels = read_jsonl(data_dir / "labels" / qfile.name)
        # 防泄漏：Agent 读的题目文件只能有这两个字段
        extra = {k for q in questions for k in q} - {"qid", "question"}
        if extra:
            problems.append(f"{name}: 题目文件含有多余字段 {sorted(extra)}")
        qids = [q["qid"] for q in questions]
        if len(qids) != len(set(qids)):
            problems.append(f"{name}: qid 有重复")
        if qids != [l["qid"] for l in labels]:
            problems.append(f"{name}: 题目文件和答案文件的 qid 对不上")
        missing = [l["qid"] for l in labels if not set(l["gold_doc_ids"]) <= corpus_ids]
        if missing:
            problems.append(f"{name}: {len(missing)} 道题的金标段落不在语料里，如 {missing[:3]}")
        split_qids[name] = set(qids)

    for a, b in combinations(sorted(split_qids), 2):
        overlap = split_qids[a] & split_qids[b]
        if overlap:
            problems.append(f"{a} 和 {b} 有 {len(overlap)} 道题重叠")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/hotpotqa/v1")
    problems = validate(Path(ap.parse_args().data_dir))
    for p in problems:
        print("❌", p)
    if problems:
        sys.exit(1)
    print("✅ 数据体检通过")


if __name__ == "__main__":
    main()
