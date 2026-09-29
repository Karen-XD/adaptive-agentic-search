"""Day 5 错题分类：每道答错的题只归一个主因，按"改哪一处最可能让它变对"（最小修复）判断。

用法（Agent 的"查询写得差"要用原问题检索一次，先起检索服务）：
  python -m experiments.day5_error_taxonomy outputs/runs/<运行>              # 分类统计，写 <运行>/error_taxonomy.csv
  python -m experiments.day5_error_taxonomy outputs/runs/<运行> --show QID   # 逐轮打印一道题的轨迹
离线分析，读答案文件里的金标是评测侧行为；不改变任何实验结果。

按下面的顺序判断，命中就停。越靠前的修复越便宜（改答案规范 < 改停止策略 < 改查询 < 换检索器），
先把便宜的原因排除，剩下的才算检索器的能力问题：
  格式非法 → 标签问题 → 答案形式 → 证据够了仍读错
  → 证据不够：过早停止 → 查询写得差 → 无效重复搜索 → 缺第二跳 / 缺一个实体 → 检索未命中
"证据够了" = 金标段落全找到，或（桥接题）答案原文已经出现在检索到的文字里：这时停下没错，该修的是阅读。
只能作答的方法（Direct / Static RAG / Oracle）没有搜索行为，跳过"过早停止"到"无效重复搜索"三类。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from agent.schema import ErrorCode
from evaluation.qa_metrics import normalize_answer

CATEGORIES = ["格式非法", "标签问题", "答案形式", "证据够了仍读错", "过早停止", "查询写得差", "无效重复搜索",
              "缺第二跳", "缺一个实体", "检索未命中", "未检索"]
YES_NO = {"yes", "no"}
LONG_GOLD_WORDS = 12  # 金标答案超过这么多词基本是标注错误（4.3 只见到 1 题：金标是一整句人物简介）


def searches(traj: dict) -> list[dict]:
    """实际执行成功的搜索（ok=True）。检索出错不算：它和"搜到了但没新东西"是两回事。"""
    return [s for s in traj["steps"]
            if s["action"] and s["action"]["name"] == "search" and s["observation"] and s["observation"]["ok"]]


def docs_seen(traj: dict) -> list[dict]:
    docs = [d for s in searches(traj) for d in s["observation"]["docs"]]
    if traj["context"] and traj["context"]["observation"]["ok"]:
        docs += traj["context"]["observation"]["docs"]
    return docs


def answer_form_issue(pred: str, gold: str) -> bool:
    """内容基本对、形式不对：一方的词全在另一方里（"River Calder" vs "Calder"，"Michele Bachmann" vs
    "Michele Marie Bachmann"），或问是否却答了别的（"French" vs "yes"）。按词集合而不是子串判断：金标常带中间名。"""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if g in YES_NO:
        return p not in YES_NO
    pw, gw = set(p.split()), set(g.split())
    return bool(pw) and (pw <= gw or gw <= pw)


def evidence_enough(traj: dict, label: dict, seen: set[str]) -> bool:
    if set(label["gold_doc_ids"]) <= seen:
        return True
    if label["type"] != "bridge":
        return False  # 比较题的答案是问题里的实体名，出现在检索文本里不说明证据够
    g = normalize_answer(label["answer"])
    if g.isdigit() or g in normalize_answer(traj["question"]):
        return False  # 纯数字最容易碰巧出现在别的段落里（抽查：金标 "17" 被一段无关文字命中）
    return any(g in normalize_answer(f"{d['title']} {d['text']}") for d in docs_seen(traj))


def classify(traj: dict, label: dict, budget: dict, original_hits: set[str] | None) -> str:
    pred, gold = traj["final_answer"], label["answer"]
    if pred is None:
        return "格式非法"
    if len(gold.split()) > LONG_GOLD_WORDS:
        return "标签问题"
    if answer_form_issue(pred, gold):
        return "答案形式"
    if traj["method"] == "direct":
        return "未检索"
    gold_ids = set(label["gold_doc_ids"])
    seen = {d["doc_id"] for d in docs_seen(traj)}
    if evidence_enough(traj, label, seen):
        return "证据够了仍读错"

    if traj["method"] == "agent":
        used = traj["budget_state"]["search_calls_used"]
        if traj["stop_reason"] == "answered" and used < budget["max_search_calls"]:
            return "过早停止"
        if original_hits and (original_hits & gold_ids) - seen:
            return "查询写得差"
        wasted = any(s["num_new_docs"] == 0 for s in searches(traj)) or any(
            s["observation"] and s["observation"].get("error_code") == ErrorCode.DUPLICATE_QUERY.value
            for s in traj["steps"])
        if wasted and used >= budget["max_search_calls"]:
            return "无效重复搜索"
    if gold_ids & seen:
        return "缺第二跳" if label["type"] == "bridge" else "缺一个实体"
    return "检索未命中"


def show(traj: dict, label: dict) -> None:
    gold_ids = set(label["gold_doc_ids"])
    print(f"{traj['qid']} [{label['type']}] {traj['method']} stop={traj['stop_reason']}")
    print(f"Q: {traj['question']}\ngold: {label['answer']!r}  模型: {traj['final_answer']!r}  金标段落: {label['gold_titles']}")
    if traj["context"]:
        print(f"  [证据] {[(d['title'], '★' if d['doc_id'] in gold_ids else '') for d in traj['context']['observation']['docs']]}")
    for s in traj["steps"]:
        a, o = s["action"], s["observation"]
        head = f"  t{s['turn']}{'[强制]' if s['forced'] else ''} "
        if a is None:
            print(head + f"格式错 {o['error_code']}: {' '.join(s['generated'].split())[:200]}")
        elif a["name"] == "final_answer":
            print(head + f"作答 {a['arguments']['answer']!r}")
        elif o["ok"]:
            docs = [(d["title"], "★" if d["doc_id"] in gold_ids else "") for d in o["docs"]]
            print(head + f"搜 {a['arguments']['query']!r} → {docs} 新文档={s['num_new_docs']}")
        else:
            print(head + f"搜 {a['arguments']['query']!r} → {o['error_code']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--show", metavar="QID")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    labels_path = Path(cfg["data"]["dir"]) / "labels" / f"{cfg['data']['split']}.jsonl"
    labels = {l["qid"]: l for l in map(json.loads, open(labels_path, encoding="utf-8"))}
    trajs = [json.loads(line) for line in open(run / "trajectories.jsonl", encoding="utf-8")]

    if args.show:
        traj = next(t for t in trajs if t["qid"] == args.show)
        show(traj, labels[traj["qid"]])
        return

    tool = None
    if cfg["method"] == "agent":
        from retrieval.client import HttpSearchTool
        tool = HttpSearchTool(cfg["retrieval"]["url"])
    rows = []
    for t in trajs:
        if t["eval"]["em"]:
            continue
        label = labels[t["qid"]]
        # 原问题搜一次（和 Static RAG 同一个查询、同一个 top_k），用来判断 Agent 自己写的查询是不是更差
        hits = {d.doc_id for d in tool.search(t["question"], cfg["budget"]["top_k"])} if tool else None
        rows.append({"qid": t["qid"], "type": label["type"], "category": classify(t, label, cfg["budget"], hits),
                     "stop_reason": t["stop_reason"], "prediction": t["final_answer"], "gold": label["answer"],
                     "f1": round(t["eval"]["f1"], 3), "evidence_recall": t["eval"]["evidence_recall"],
                     "search_calls": t["budget_state"]["search_calls_used"]})

    with open(run / "error_taxonomy.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(trajs)
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["type"]][r["category"]] += 1
    types = sorted(by_type)
    print(f"{run.name}：{n} 题，答错 {len(rows)} 题（每题只归一个主因）")
    print(f"  {'主因':10s} {'题数':>4s} {'占全部':>6s}  " + "  ".join(f"{t:>10s}" for t in types))
    total = Counter(r["category"] for r in rows)
    for c in CATEGORIES:
        if total[c]:
            print(f"  {c:10s} {total[c]:4d} {total[c] / n:6.1%}  " + "  ".join(f"{by_type[t][c]:10d}" for t in types))
    print(f"逐题结果：{run / 'error_taxonomy.csv'}")


if __name__ == "__main__":
    main()
