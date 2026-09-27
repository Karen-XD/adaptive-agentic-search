"""从 HotpotQA distractor 原始数据构造：语料池 + 题目划分 + 数据清单。

语料池 = train 与官方 dev 全部题目自带的上下文段落（每题 2 段金标 + 8 段干扰），按标题去重。
它由整个数据集决定，不依赖抽到哪些题；比全量维基（约 520 万段）小一个量级，结果表必须标注 retrieval_setting。

划分：
  test       官方 dev（全是 hard 题）固定抽样，最后才跑
  validation 从 train 的 hard 题抽，分布与 test 一致；提示词和阈值只在这里调
  debug      从 train 的 hard 题抽，与 validation 不重叠；随便看、随便试
每个划分拆成两个文件：questions/ 只有 qid + question（给 Agent），labels/ 放答案和金标段落（只给评测器）。

用法：python -m data_prep.prepare_hotpot
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

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
             "dev": ["validation-00000-of-00001.parquet"]}


def doc_id_of(title: str) -> str:
    # 标题在语料池里唯一，用标题哈希做 ID：重建语料时 ID 不变
    return "hp-" + hashlib.md5(title.encode("utf-8")).hexdigest()[:12]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_corpus(dfs: list[pd.DataFrame]) -> list[dict]:
    versions: dict[str, Counter] = defaultdict(Counter)
    for df in dfs:
        for ctx in df.context:
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                versions[title]["".join(sents).strip()] += 1
    corpus = []
    for title in sorted(versions):
        # 同一标题有多个版本时（维基不同时间的小修订，相似度中位数 0.998）留出现最多的；并列取字典序最小，保证可复现
        text = min(versions[title].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        corpus.append({"doc_id": doc_id_of(title), "title": title, "text": text})
    return corpus


def to_split(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    questions, labels = [], []
    for ex in df.itertuples():
        gold_titles = sorted(set(ex.supporting_facts["title"]))
        questions.append({"qid": ex.id, "question": ex.question})
        labels.append({"qid": ex.id, "answer": ex.answer, "type": ex.type, "level": ex.level,
                       "gold_titles": gold_titles, "gold_doc_ids": [doc_id_of(t) for t in gold_titles]})
    return questions, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/hotpotqa_distractor")
    ap.add_argument("--out_dir", default="data/hotpotqa/v1")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_validation", type=int, default=200)
    ap.add_argument("--n_debug", type=int, default=50)
    args = ap.parse_args()
    raw, out = ROOT / args.raw_dir, ROOT / args.out_dir

    train = pd.concat([pd.read_parquet(raw / f) for f in RAW_FILES["train"]], ignore_index=True)
    dev = pd.read_parquet(raw / RAW_FILES["dev"][0])

    corpus = build_corpus([train, dev])
    write_jsonl(out / "corpus.jsonl", corpus)

    # 先按 qid 排序再抽样：抽样结果只取决于 seed，不受原始文件行顺序影响
    rng = random.Random(args.seed)
    test_ids = rng.sample(sorted(dev.id), args.n_test)
    hard_train = sorted(train.loc[train.level == "hard", "id"])
    picked = rng.sample(hard_train, args.n_validation + args.n_debug)
    splits = {"test": (dev, test_ids), "validation": (train, picked[:args.n_validation]),
              "debug": (train, picked[args.n_validation:])}

    stats = {}
    for name, (df, ids) in splits.items():
        sub = df.set_index("id").loc[ids].reset_index()  # 保持抽样顺序
        questions, labels = to_split(sub)
        write_jsonl(out / "questions" / f"{name}.jsonl", questions)
        write_jsonl(out / "labels" / f"{name}.jsonl", labels)
        stats[name] = {"num_questions": len(ids), "source": "dev" if df is dev else "train",
                       "type": dict(Counter(sub.type)), "level": dict(Counter(sub.level))}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "dataset": "hotpotqa", "corpus_version": "hotpot_distractor_pool_v1",
        "retrieval_setting": "pool",  # 不是 fullwiki，结果表标题里必须写明
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
        "seed": args.seed,
        "source": {"hf_dataset": "hotpotqa/hotpot_qa", "config": "distractor",
                   "files": {f: sha256(raw / f) for fs in RAW_FILES.values() for f in fs}},
        "corpus": {"num_docs": len(corpus), "dedup": "by title, keep most frequent version",
                   "built_from": f"all context paragraphs of train ({len(train)}) + dev ({len(dev)})"},
        "splits": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
