"""从 Amazon ESCI（Shopping Queries Dataset）构造：商品语料 + 查询划分 + 数据清单。

只取 product_locale=us、small_version=1 的英文子集（官方 `large_version` 是超大池，本项目不用）。
划分：official split 原样保留，validation / debug 从官方 train 里抽样，互不重叠：
  test        官方 test 抽 400 条（最后才跑）
  validation  官方 train 抽 400 条；提示词和门槛只在这里调
  debug       官方 train 抽 25 条，和 validation 不重叠
每个查询拆成两个文件：questions/ 只有 qid + 查询词（给 Agent / 评测流程），
labels/ 放候选商品和 ESCI 标签（只给评测器，绝不进 prompt 或检索工具）。

两个设定（见计划 4.2）：
  Setting A（主评测）：每个查询用官方已标注的那批商品当**固定候选池**，比较各种排序方法；
                        这时"检索"退化成对候选池重排，不能声称是全库召回。
  Setting B（扩展）：用 corpus.jsonl 建全库索引，测已标注正例的覆盖率。未标注商品不能当负例。

增益映射 E=3, S=2, C=1, I=0 是**本项目人为设定**，不是官方业务效用；Complement（互补品）的权重
要做敏感性分析。

用法：python -m data_prep.prepare_esci
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {"examples": "shopping_queries_dataset_examples.parquet",
             "products": "shopping_queries_dataset_products.parquet"}


def doc_id_of(product_id: str) -> str:
    # 商品 ID 唯一，用哈希做 doc_id：重建语料时 ID 不变
    return "es-" + hashlib.md5(product_id.encode("utf-8")).hexdigest()[:12]


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


def product_text(row: pd.Series) -> str:
    """商品文本 = 品牌 + 要点 + 描述（标题单独放在 title 字段，和 HotpotQA 语料同一套结构）。"""
    parts = []
    if isinstance(row.get("product_brand"), str) and row["product_brand"].strip():
        parts.append(f"Brand: {row['product_brand'].strip()}")
    if isinstance(row.get("product_bullet_point"), str) and row["product_bullet_point"].strip():
        parts.append(row["product_bullet_point"].strip())
    if isinstance(row.get("product_description"), str) and row["product_description"].strip():
        parts.append(row["product_description"].strip())
    return "\n".join(parts) or "(no description)"


def to_split(df: pd.DataFrame, qids: list[str]) -> tuple[list[dict], list[dict]]:
    sub = df[df.query_id.isin(qids)]
    questions, labels = [], []
    for qid, g in sub.groupby("query_id", sort=False):
        query_text = g["query"].iloc[0]
        questions.append({"qid": qid, "question": query_text})
        cands = [{"product_id": p, "esci_label": l, "doc_id": doc_id_of(p)}
                 for p, l in zip(g.product_id, g.esci_label)]
        labels.append({"qid": qid, "question": query_text, "candidates": cands,
                       "num_candidates": len(cands),
                       "label_counts": dict(Counter(c["esci_label"] for c in cands)),
                       "gold_doc_ids": [c["doc_id"] for c in cands if c["esci_label"] == "E"]})
    return questions, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/esci")
    ap.add_argument("--out", default="data/esci/v1")
    ap.add_argument("--n_validation", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--n_debug", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    raw, out = ROOT / args.raw, ROOT / args.out
    examples = pd.read_parquet(raw / RAW_FILES["examples"])
    products = pd.read_parquet(raw / RAW_FILES["products"])

    us = examples[(examples.product_locale == "us") & (examples.small_version == 1)].copy()
    train_qids = sorted(us.loc[us.split == "train", "query_id"].unique())
    test_qids = sorted(us.loc[us.split == "test", "query_id"].unique())

    rng = random.Random(args.seed)
    val_qids = rng.sample(train_qids, args.n_validation)
    rest = [q for q in train_qids if q not in set(val_qids)]
    debug_qids = rng.sample(rest, args.n_debug)
    test_qids = rng.sample(test_qids, args.n_test)
    assert not set(val_qids) & set(debug_qids), "splits overlap"

    # 语料 = 这些查询用到的所有商品（Setting A 的候选池就是它的子集；Setting B 用它建索引）
    used = set(us[us.split.isin(["train", "test"])].product_id)
    prod = products[products.product_id.isin(used)].drop_duplicates("product_id").set_index("product_id")
    missing = used - set(prod.index)
    if missing:
        raise SystemExit(f"{len(missing)} 个商品在 products 表里找不到")
    corpus = [{"doc_id": doc_id_of(pid), "product_id": pid,
               "title": str(row.product_title), "text": product_text(row)}
              for pid, row in prod.iterrows()]
    corpus.sort(key=lambda d: d["doc_id"])
    write_jsonl(out / "corpus.jsonl", corpus)

    stats = {}
    for name, qids, src in (("validation", val_qids, "train"), ("debug", debug_qids, "train"),
                            ("test", test_qids, "test")):
        questions, labels = to_split(us, qids)
        write_jsonl(out / "questions" / f"{name}.jsonl", questions)
        write_jsonl(out / "labels" / f"{name}.jsonl", labels)
        stats[name] = {
            "num_queries": len(labels), "source": f"official {src} split", "sampling": "random",
            "mean_candidates": round(sum(l["num_candidates"] for l in labels) / len(labels), 2),
            "esci_labels": dict(Counter(c["esci_label"] for l in labels for c in l["candidates"])),
            "queries_without_exact": sum(not l["gold_doc_ids"] for l in labels)}
    # 全库检索时的"已标注正例覆盖率"分母：每个查询的 E 商品
    stats["num_train_queries"] = len(train_qids)
    stats["num_official_test_queries"] = len(test_qids)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "dataset": "amazon_esci", "corpus_version": "esci_us_small_v1", "retrieval_setting": "fixed_candidates",
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "seed": args.seed,
        "source": {"url": "https://github.com/amazon-science/esci-data",
                   "files": {k: sha256(raw / v) for k, v in RAW_FILES.items()}},
        "filter": {"product_locale": "us", "small_version": 1},
        "gain_mapping": {"E": 3, "S": 2, "C": 1, "I": 0,
                         "note": "本项目人为设定，不是官方业务效用；C 的权重需做敏感性分析"},
        "corpus": {"num_docs": len(corpus), "text_fields": ["product_title", "product_brand",
                                                            "product_bullet_point", "product_description"]},
        "splits": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
