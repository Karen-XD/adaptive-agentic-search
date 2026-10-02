"""从 2WikiMultihopQA 原始数据构造：语料池 + 分析集 + 数据清单。和 prepare_hotpot 同一套规则，两个数据集的检索结果才能直接比。

语料池 = train 与官方 dev 全部题目自带的上下文段落（每题 10 段），按标题去重；同一标题多个版本留出现最多的。
目前只做 analysis 划分：从 train 按题型分层抽样（每类同样多），用于离线检索分析。
2Wiki 的 inference 题在 train 里只占 2.6%，按自然分布抽几乎抽不到；分层后按题型报告，不报一个混合平均。
以后如果把 2Wiki 作为正式评测集，validation / test 要另抽，并且和 analysis 不重叠。
标签里额外保存 evidences（实体, 关系, 值）三元组：只给评测侧分析用（构造理想子查询），不进 prompt。

用法：python -m data_prep.prepare_2wiki
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_prep.prepare_hotpot import sha256, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {"train": "train.parquet", "dev": "dev.parquet"}  # hf-mirror: xanhho/2WikiMultihopQA


def doc_id_of(title: str) -> str:
    return "2w-" + hashlib.md5(title.encode("utf-8")).hexdigest()[:12]


def build_corpus(dfs: list[pd.DataFrame]) -> list[dict]:
    versions: dict[str, Counter] = defaultdict(Counter)
    for df in dfs:
        for ctx in df.context:
            for title, sents in json.loads(ctx):
                # 2Wiki 的句子不带前导空格（HotpotQA 带），所以用空格拼
                versions[title][" ".join(s.strip() for s in sents).strip()] += 1
    corpus = []
    for title in sorted(versions):
        text = min(versions[title].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        corpus.append({"doc_id": doc_id_of(title), "title": title, "text": text})
    return corpus


def to_split(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    questions, labels = [], []
    for ex in df.itertuples():
        gold_titles = sorted({t for t, _ in json.loads(ex.supporting_facts)})
        questions.append({"qid": ex._1, "question": ex.question})
        labels.append({"qid": ex._1, "answer": ex.answer, "type": ex.type, "gold_titles": gold_titles,
                       "gold_doc_ids": [doc_id_of(t) for t in gold_titles],
                       "evidences": json.loads(ex.evidences)})
    return questions, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/2wiki")
    ap.add_argument("--out", default="data/2wiki/v1")
    ap.add_argument("--n_per_type", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    raw, out = ROOT / args.raw, ROOT / args.out
    train = pd.read_parquet(raw / RAW_FILES["train"])
    dev = pd.read_parquet(raw / RAW_FILES["dev"])
    corpus = build_corpus([train, dev])
    write_jsonl(out / "corpus.jsonl", corpus)

    rng = random.Random(args.seed)
    picked = []
    for t in sorted(train.type.unique()):
        picked += rng.sample(sorted(train.loc[train.type == t, "_id"]), args.n_per_type)
    sub = train.set_index("_id").loc[picked].reset_index()
    questions, labels = to_split(sub)
    known = {d["doc_id"] for d in corpus}
    missing = sum(g not in known for l in labels for g in l["gold_doc_ids"])
    if missing:
        raise SystemExit(f"{missing} gold doc ids not in corpus")
    write_jsonl(out / "questions" / "analysis.jsonl", questions)
    write_jsonl(out / "labels" / "analysis.jsonl", labels)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "dataset": "2wikimultihopqa", "corpus_version": "2wiki_pool_v1", "retrieval_setting": "pool",
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "seed": args.seed,
        "source": {"hf_dataset": "xanhho/2WikiMultihopQA",
                   "files": {f: sha256(raw / f) for f in RAW_FILES.values()}},
        "corpus": {"num_docs": len(corpus), "dedup": "by title, keep most frequent version",
                   "built_from": f"all context paragraphs of train ({len(train)}) + dev ({len(dev)})"},
        "splits": {"analysis": {"num_questions": len(labels), "source": "train", "sampling": "stratified by type",
                                "type": dict(Counter(l["type"] for l in labels)),
                                "num_gold": dict(Counter(len(l["gold_doc_ids"]) for l in labels))}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
