"""从 2WikiMultihopQA 原始数据构造：语料池 + 分析集 + 数据清单。和 prepare_hotpot 同一套规则，两个数据集的检索结果才能直接比。

语料池 = train 与官方 dev 全部题目自带的上下文段落（每题 10 段），按标题去重；同一标题多个版本留出现最多的。
划分都从 train 按题型分层抽样（每类同样多），三者互不重叠：
  analysis    每类 500，离线检索分析（Day 8～9）
  validation  每类 200，端到端对比（Day 10.4 起）；提示词不在这里调
  debug       每类 25，调提示词、看输出
2Wiki 的 inference 题在 train 里只占 2.6%，按自然分布抽几乎抽不到；分层后按题型报告，不报一个混合平均。
  test        每类 200，从官方 dev 抽（Day 12 主实验，只跑一次，带 --final）
抽样顺序固定（先 analysis，再 validation、debug，最后 test），每个后加的划分用独立的随机数生成器，加新划分不改变已有划分。
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
    ap.add_argument("--n_validation_per_type", type=int, default=200)
    ap.add_argument("--n_debug_per_type", type=int, default=25)
    ap.add_argument("--n_test_per_type", type=int, default=200)
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
    # 后加的划分：从 analysis 没抽到的题里抽，用独立的随机数生成器，不影响 analysis 的抽样结果
    rest = train[~train["_id"].isin(set(picked))]
    rng2 = random.Random(args.seed + 1)
    extra = {"validation": [], "debug": []}
    for t in sorted(train.type.unique()):
        ids = rng2.sample(sorted(rest.loc[rest.type == t, "_id"]), args.n_validation_per_type + args.n_debug_per_type)
        extra["validation"] += ids[:args.n_validation_per_type]
        extra["debug"] += ids[args.n_validation_per_type:]
    # test 从官方 dev 抽（和 train 天然不重叠），同样按题型分层
    rng3 = random.Random(args.seed + 2)
    test_ids = []
    for t in sorted(dev.type.unique()):
        test_ids += rng3.sample(sorted(dev.loc[dev.type == t, "_id"]), args.n_test_per_type)
    known = {d["doc_id"] for d in corpus}
    by_id = {"train": train.set_index("_id"), "dev": dev.set_index("_id")}
    stats = {}
    for name, ids, src in (("analysis", picked, "train"), *((k, v, "train") for k, v in extra.items()),
                           ("test", test_ids, "dev")):
        questions, labels = to_split(by_id[src].loc[ids].reset_index())
        missing = sum(g not in known for l in labels for g in l["gold_doc_ids"])
        if missing:
            raise SystemExit(f"{name}: {missing} gold doc ids not in corpus")
        write_jsonl(out / "questions" / f"{name}.jsonl", questions)
        write_jsonl(out / "labels" / f"{name}.jsonl", labels)
        stats[name] = {"num_questions": len(labels), "source": src, "sampling": "stratified by type",
                       "type": dict(Counter(l["type"] for l in labels)),
                       "num_gold": dict(Counter(len(l["gold_doc_ids"]) for l in labels))}
    split_ids = {name: set(ids) for name, ids in (("analysis", picked), *extra.items(), ("test", test_ids))}
    assert not (split_ids["analysis"] & split_ids["validation"] or split_ids["analysis"] & split_ids["debug"]
                or split_ids["validation"] & split_ids["debug"]), "splits overlap"
    assert not split_ids["test"] & set(train["_id"]), "test overlaps train"

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "dataset": "2wikimultihopqa", "corpus_version": "2wiki_pool_v1", "retrieval_setting": "pool",
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "seed": args.seed,
        "source": {"hf_dataset": "xanhho/2WikiMultihopQA",
                   "files": {f: sha256(raw / f) for f in RAW_FILES.values()}},
        "corpus": {"num_docs": len(corpus), "dedup": "by title, keep most frequent version",
                   "built_from": f"all context paragraphs of train ({len(train)}) + dev ({len(dev)})"},
        "splits": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
