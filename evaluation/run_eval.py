"""实验入口：读配置 → 锁定代码版本 → 逐题跑 Agent → 评测 → 写 outputs/runs/<run_id>/。

用法：python -m evaluation.run_eval --config configs/mock_v1.yaml [--allow-dirty]
题目文件（只有 qid + question）交给 Agent；答案文件只在评测这一步读取，不传给 Agent 和检索工具。
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from agent.llm import RetryingLLM, SearchThenTitleLLM
from agent.loop import run_episode
from agent.schema import Budget, ErrorCode, StopReason
from evaluation.qa_metrics import aggregate, exact_match
from retrieval.bm25 import BM25SearchTool
from retrieval.client import HttpSearchTool
from retrieval.mock import MockSearchTool

ROOT = Path(__file__).resolve().parents[1]
MAX_FAILURE_RATE = 0.02  # 模型服务失败率超过它，这次运行不能写进结论
FORMAT_ERRORS = {ErrorCode.NO_ACTION, ErrorCode.INVALID_JSON, ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGS}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def build_llm(cfg: dict):
    if cfg["type"] == "search_then_title":
        return SearchThenTitleLLM()
    raise ValueError(f"unknown llm type: {cfg['type']}")


def build_tool(cfg: dict):
    if cfg["type"] == "mock":
        return MockSearchTool(), {"source": "mock"}
    if cfg["type"] == "bm25":  # 进程内加载索引，适合调试
        tool = BM25SearchTool(ROOT / cfg["index"])
        return tool, {"source": tool.source, "index": tool.meta}
    if cfg["type"] == "http":  # 连检索服务，正式实验用这个；先问 /health，确认连的是哪个索引
        tool = HttpSearchTool(cfg["url"], cfg.get("timeout_s", 5.0))
        return tool, tool.health()
    raise ValueError(f"unknown retrieval type: {cfg['type']}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="调试用：允许在有未提交改动时运行，结果目录带 -dirty 后缀")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # 代码版本：有未提交改动时 commit hash 代表不了实际跑的代码，默认拒绝运行
    status = _git("status", "--porcelain")
    dirty = bool(status.strip())
    if dirty and not args.allow_dirty:
        sys.exit("工作区有未提交的改动，commit hash 无法代表本次代码。请先提交，或加 --allow-dirty 调试运行。")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}" + ("-dirty" if dirty else "")
    out = ROOT / "outputs" / "runs" / run_id
    out.mkdir(parents=True)
    commit = _git("rev-parse", "HEAD").strip()
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    if dirty:
        # 未跟踪的新文件不在 diff 里，所以把 status 一起存下
        (out / "git_diff.patch").write_text(f"# git status\n{status}\n# git diff HEAD\n{_git('diff', 'HEAD')}",
                                            encoding="utf-8")

    budget = Budget(**cfg["budget"])
    llm = RetryingLLM(build_llm(cfg["llm"]), **cfg["llm"].get("retry", {}))
    tool, retrieval_info = build_tool(cfg["retrieval"])
    data_dir = ROOT / cfg["data"]["dir"]
    split = cfg["data"]["split"]
    questions = read_jsonl(data_dir / "questions" / f"{split}.jsonl")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    # 结果由 代码 + 配置 + 数据 + 环境 共同决定，四样都记下
    random.seed(cfg["seed"])
    cfg["run"] = {
        "run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "dirty": dirty, "argv": sys.argv,
        "data_manifest": manifest, "retrieval": retrieval_info,
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                **{pkg: version(pkg) for pkg in ("pydantic", "pyyaml", "bm25s")}},
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    trajectories = [run_episode(q["qid"], q["question"], llm, tool, budget) for q in questions]

    # ---- 以下是评测：这时才读答案文件 ----
    labels = {l["qid"]: l for l in read_jsonl(data_dir / "labels" / f"{split}.jsonl")}
    records = []
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for traj in trajectories:
            label = labels[traj.qid]
            correct = exact_match(traj.final_answer, label["answer"])
            state = traj.budget_state
            # 证据召回：整条轨迹检索到的段落里，覆盖了几个金标段落。把"搜得好不好"和"答得好不好"分开
            retrieved = {d.doc_id for s in traj.steps if s.observation is not None for d in s.observation.docs}
            gold_ids = set(label.get("gold_doc_ids", []))
            records.append({
                "qid": traj.qid, "correct": correct, "error": traj.stop_reason == StopReason.ERROR,
                "stop_reason": traj.stop_reason.value, "prediction": traj.final_answer, "gold": label["answer"],
                "type": label.get("type"),
                "evidence_recall": len(gold_ids & retrieved) / len(gold_ids) if gold_ids else None,
                "turns": state.turns_used, "search_calls": state.search_calls_used,
                "search_attempts": state.search_attempts,
                "new_docs": sum(s.num_new_docs for s in traj.steps),
                "format_errors": sum(1 for s in traj.steps
                                     if s.observation is not None and s.observation.error_code in FORMAT_ERRORS),
                "error_message": traj.error,
            })
            f.write(json.dumps({**traj.model_dump(mode="json"),
                                "eval": {"gold": label["answer"], "em": correct,
                                         "gold_doc_ids": sorted(gold_ids), "evidence_recall": records[-1]["evidence_recall"]}},
                               ensure_ascii=False) + "\n")

    metrics = aggregate(records)
    metrics["valid"] = metrics["failure_rate"] <= MAX_FAILURE_RATE
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # errors.csv：没答对的题都进来，按类别分开，方便 badcase 分析
    fields = ["qid", "type", "category", "stop_reason", "prediction", "gold", "evidence_recall", "turns",
              "search_calls", "format_errors", "error_message"]
    with open(out / "errors.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            if r["correct"]:
                continue
            category = ("model_error" if r["error"] else
                        "no_answer" if r["stop_reason"] == StopReason.NO_ANSWER.value else "wrong_answer")
            w.writerow({**r, "category": category})

    print(f"run_id: {run_id}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not metrics["valid"]:
        print(f"⚠️ 失败率 {metrics['failure_rate']:.1%} 超过 {MAX_FAILURE_RATE:.0%}，本次结果无效")


if __name__ == "__main__":
    main()
