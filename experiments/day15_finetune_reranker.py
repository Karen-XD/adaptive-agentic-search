"""Day 15：用 ESCI 官方 train 的 E/S/C/I 标签微调商品重排模型（从 bge-reranker-base 出发）。

为什么要微调：Day 13 里通用的向量模型、重排模型、3B 大模型在商品上全部打平（nDCG@10 0.84～0.85），
它们都没见过商品数据。商品相关性难在区分"精确匹配"和"替代品"，这需要领域标签来教。

训练目标（组内排序，ListNet）：同一个查询的候选商品放在一组，模型给每个（查询, 商品）打一个分；
目标分布 = softmax(增益)，增益沿用评测的 E=3 / S=2 / C=1 / I=0；损失 = 目标分布和 softmax(模型分数) 的交叉熵。
它和 nDCG 一样只看组内相对顺序（给所有分数加同一个常数，损失不变），所以训练目标和评测指标对得上。
全是同一种标签的组没有排序信号，跳过。

防泄漏：
  - 训练查询 = 官方 train 的 us + small 查询，**剔除** data/esci/v1 里 validation 和 debug 的 425 条（它们也是从官方 train 抽的）
  - 官方 test 的查询本来就不在官方 train 里（query_id 零重叠，Day 13 已查），这里再断言一遍
  - 选检查点只看 validation，test 不参与

用法（先停掉 vLLM 腾显存）：
  python -m experiments.day15_finetune_reranker --out /root/autodl-tmp/checkpoints/esci_reranker_v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_prep.prepare_esci import doc_id_of
from evaluation.commerce_metrics import GAIN, ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
BASE = "/root/autodl-tmp/hf_models/bge-reranker-base"


def load_corpus_text() -> dict[str, str]:
    """doc_id -> 商品文本（标题 + 品牌 + 要点 + 描述），和 Setting A 评测时给重排模型的文本完全相同。"""
    out = {}
    with open(ROOT / "data/esci/v1/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = f"{d['title']}\n{d['text']}"
    return out


def load_train_groups(exclude_qids: set) -> list[dict]:
    ex = pd.read_parquet(ROOT / "data/raw/esci/shopping_queries_dataset_examples.parquet",
                         columns=["query_id", "query", "product_id", "product_locale", "esci_label",
                                  "small_version", "split"])
    tr = ex[(ex.product_locale == "us") & (ex.small_version == 1) & (ex.split == "train")]
    test_qids = set(ex[(ex.product_locale == "us") & (ex.small_version == 1) & (ex.split == "test")].query_id)
    assert not set(tr.query_id) & test_qids, "官方 train 和 test 的查询重叠"
    tr = tr[~tr.query_id.isin(exclude_qids)]
    assert not set(tr.query_id) & exclude_qids, "训练集里混进了 validation / debug 查询"
    groups = []
    for qid, g in tr.groupby("query_id", sort=True):
        labels = g.esci_label.tolist()
        if len(set(labels)) < 2:  # 全是同一种标签：没有排序信号
            continue
        groups.append({"qid": int(qid), "query": g["query"].iloc[0],
                       "doc_ids": [doc_id_of(p) for p in g.product_id], "labels": labels})
    return groups


def load_eval_split(name: str) -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / f"data/esci/v1/labels/{name}.jsonl", encoding="utf-8")]
    return [{"qid": r["qid"], "query": r["question"], "doc_ids": [c["doc_id"] for c in r["candidates"]],
             "labels": [c["esci_label"] for c in r["candidates"]]} for r in rows]


@torch.inference_mode()
def score_groups(model, tok, groups: list[dict], text: dict[str, str], max_length: int, device: str,
                 batch_size: int = 128) -> list[list[float]]:
    model.eval()
    pairs, owner = [], []
    for gi, g in enumerate(groups):
        for d in g["doc_ids"]:
            pairs.append((g["query"], text[d]))
            owner.append(gi)
    scores = []
    for s in range(0, len(pairs), batch_size):
        batch = tok(pairs[s:s + batch_size], padding=True, truncation="only_second", max_length=max_length,
                    return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores += model(**batch).logits[:, 0].float().cpu().tolist()
    out = [[] for _ in groups]
    for gi, sc in zip(owner, scores):
        out[gi].append(sc)
    return out


def eval_ndcg(groups: list[dict], scores: list[list[float]]) -> float:
    vals = []
    for g, sc in zip(groups, scores):
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], g["doc_ids"][i]))  # 同分按 doc_id，和其他方法一致
        vals.append(ndcg_at_k([g["labels"][i] for i in order], 10, GAIN))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="检查点目录（数据盘，不进 git）")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--group-size", type=int, default=16, help="每个查询每步最多取多少个候选（多了随机抽）")
    ap.add_argument("--queries-per-batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=1000, help="每多少步在 validation 上评一次")
    ap.add_argument("--limit-queries", type=int, help="调试用：只用前 N 个训练查询")
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val = load_eval_split("validation")
    debug = load_eval_split("debug")
    exclude = {g["qid"] for g in val} | {g["qid"] for g in debug}
    groups = load_train_groups(exclude)
    if args.limit_queries:
        groups = groups[:args.limit_queries]
    text = load_corpus_text()
    missing = {d for g in groups + val for d in g["doc_ids"]} - set(text)
    assert not missing, f"{len(missing)} 个商品不在语料里"
    n_pairs = sum(len(g["doc_ids"]) for g in groups)
    print(f"训练查询 {len(groups)}（剔除 validation/debug {len(exclude)} 条、全同标签的查询）  候选对 {n_pairs}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE).to(device)
    steps_per_epoch = math.ceil(len(groups) / args.queries_per_batch)
    total_steps = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)

    # 起点：通用模型在 validation 上的 nDCG@10（同样的 max_length，对比才公平）
    base_ndcg = eval_ndcg(val, score_groups(model, tok, val, text, args.max_length, device))
    print(f"step 0  val nDCG@10 {base_ndcg:.4f}（通用 bge-reranker-base，max_length {args.max_length}）", flush=True)
    log = [{"step": 0, "val_ndcg": base_ndcg}]
    best = (base_ndcg, 0)

    gain = {k: float(v) for k, v in GAIN.items()}
    step, t0 = 0, time.time()
    order = []
    while step < total_steps:
        if not order:
            order = list(range(len(groups)))
            random.shuffle(order)
        batch_groups = [groups[order.pop()] for _ in range(min(args.queries_per_batch, len(order)))]
        pairs, sizes, targets = [], [], []
        for g in batch_groups:
            idx = list(range(len(g["doc_ids"])))
            if len(idx) > args.group_size:
                idx = random.sample(idx, args.group_size)
            labels = [g["labels"][i] for i in idx]
            if len(set(labels)) < 2:  # 抽样后可能只剩一种标签
                continue
            pairs += [(g["query"], text[g["doc_ids"][i]]) for i in idx]
            sizes.append(len(idx))
            targets.append(torch.softmax(torch.tensor([gain[l] for l in labels]), dim=0))
        if not sizes:
            continue
        model.train()
        batch = tok(pairs, padding=True, truncation="only_second", max_length=args.max_length,
                    return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits[:, 0].float()
        loss, start = 0.0, 0
        for size, tgt in zip(sizes, targets):
            s = logits[start:start + size]
            loss = loss - (tgt.to(device) * torch.log_softmax(s, dim=0)).sum()
            start += size
        loss = loss / len(sizes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total_steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}"
                  f"  {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total_steps:
            nd = eval_ndcg(val, score_groups(model, tok, val, text, args.max_length, device))
            log.append({"step": step, "val_ndcg": nd, "loss": loss.item(), "seconds": time.time() - t0})
            print(f"step {step}  val nDCG@10 {nd:.4f}  (best {best[0]:.4f} @ {best[1]})", flush=True)
            if nd > best[0]:
                best = (nd, step)
                model.save_pretrained(out / "best")
                tok.save_pretrained(out / "best")
    model.save_pretrained(out / "last")
    tok.save_pretrained(out / "last")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"argv": sys.argv, "git_commit": commit, "base_model": BASE, "train_queries": len(groups),
            "train_pairs": n_pairs, "excluded_qids": len(exclude), "loss": "ListNet, target softmax(gain)",
            "gain": GAIN, "args": vars(args), "steps": total_steps, "best_step": best[1],
            "best_val_ndcg": best[0], "base_val_ndcg": base_ndcg, "log": log,
            "train_seconds": time.time() - t0}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "log"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
