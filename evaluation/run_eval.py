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

from agent.llm import RetryingLLM, SearchThenTitleLLM, VLLMClient
from agent.methods import NEEDS_RETRIEVAL, check_budget, run_method
from agent.prompts import AGENT_PROMPTS, ANSWER_ONLY_PROMPTS
from agent.schema import Budget, ErrorCode, StopReason
from evaluation.oracle import ALLOWED_SPLITS as ORACLE_SPLITS, load_gold_docs
from evaluation.qa_metrics import aggregate, exact_match, f1_score
from retrieval.bm25 import BM25SearchTool
from retrieval.client import HttpSearchTool
from retrieval.mock import MockSearchTool

ROOT = Path(__file__).resolve().parents[1]
MAX_FAILURE_RATE = 0.02  # 模型服务失败率超过它，这次运行不能写进结论
MAX_CONSECUTIVE_ERRORS = 5  # 连续这么多题模型服务都失败，多半是服务挂了，提前中止，别把剩下的题全记成 error
FORMAT_ERRORS = {ErrorCode.NO_ACTION, ErrorCode.INVALID_JSON, ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGS}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _package_versions(python: str, packages: tuple[str, ...]) -> dict:
    """模型服务跑在另一个 conda 环境里（verl_env）：用那个环境的解释器读包版本。
    vLLM 0.6.3 的 /version 接口因为缺版本文件只返回 "dev"，靠不住。只读元数据，不 import vllm。"""
    code = ("import importlib.metadata as m, json; "
            f"print(json.dumps({{p: m.version(p) for p in {list(packages)!r}}}))")
    return json.loads(subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout)


def build_llm(cfg: dict):
    """返回 (模型, 要写进 config.yaml 的服务端信息)。"""
    if cfg["type"] == "search_then_title":
        return SearchThenTitleLLM(), {"type": "rule"}
    if cfg["type"] == "vllm":
        llm = VLLMClient(cfg["base_url"], cfg["model"], timeout_s=cfg.get("timeout_s", 60.0), **cfg["sampling"])
        # 先问服务端实际加载了哪个模型：配置里的名字只是别名，root 才是模型目录
        info = llm.server_info()
        if cfg.get("server_python"):
            info["server_env"] = _package_versions(cfg["server_python"], ("vllm", "torch", "transformers"))
        info["warmup_ms"] = round(llm.warmup(), 1)
        return llm, info
    raise ValueError(f"unknown llm type: {cfg['type']}")


def build_tool(cfg: dict | None):
    if cfg is None:  # direct / oracle 不检索
        return None, None
    if cfg["type"] == "mock":
        return MockSearchTool(), {"source": "mock"}
    if cfg["type"] == "bm25":  # 进程内加载索引，适合调试
        tool = BM25SearchTool(ROOT / cfg["index"])
        return tool, {"source": tool.source, "index": tool.meta}
    if cfg["type"] == "http":  # 连检索服务，正式实验用这个；先问 /health，确认连的是哪个索引
        tool = HttpSearchTool(cfg["url"], cfg.get("timeout_s", 5.0), cfg.get("method", "bm25"))
        info = tool.health()
        if tool.method not in info["methods"]:  # 服务没加载这一路就别开跑，否则每次搜索都是 tool_error
            sys.exit(f"retrieval method {tool.method!r} not loaded by server: {sorted(info['methods'])}")
        return tool, {"method": tool.method, **info}
    raise ValueError(f"unknown retrieval type: {cfg['type']}")


def load_config(path: Path) -> dict:
    """支持 extends：同一组对比实验继承同一份基础配置，只覆盖不同的字段。
    解码参数、数据、预算只写一处，才能保证"统一解码参数"不是靠人抄对的。"""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        cfg = _deep_merge(load_config(path.parent / cfg["extends"]), cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        merged[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return merged


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="调试用：允许在有未提交改动时运行，结果目录带 -dirty 后缀")
    ap.add_argument("--limit", type=int, help="调试用：只跑前 N 题（写进 config.yaml 的 argv，结果不能当正式结论）")
    ap.add_argument("--split", help="覆盖配置里的 data.split（同一份方法配置跑 debug / validation / test）")
    ap.add_argument("--final", action="store_true", help="跑 test 集必须加：test 只在最后评测时跑，平时误跑会让人忍不住按它调参")
    # 看解码波动用（Day 6）：主结果是贪心解码，换 seed 不改变输出；要看"换一次采样结论还在不在"，得开温度采样
    ap.add_argument("--temperature", type=float, help="覆盖 llm.sampling.temperature；run_id 带 -t<值>")
    ap.add_argument("--seed", type=int, help="覆盖 llm.sampling.seed（vLLM 每个请求的采样种子）；run_id 带 -s<值>")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if "name" not in cfg:
        sys.exit("配置里没有 name：基础配置不能直接运行，请运行继承它的方法配置")
    if args.split:
        cfg["data"]["split"] = args.split
    sampling_tag = ""
    if args.temperature is not None:
        cfg["llm"]["sampling"]["temperature"] = args.temperature
        sampling_tag += f"-t{args.temperature:g}"
    if args.seed is not None:
        cfg["llm"]["sampling"]["seed"] = args.seed
        sampling_tag += f"-s{args.seed}"
    if cfg["data"]["split"] == "test" and not args.final:
        sys.exit("test 集只在最后评测时跑一次；确认要跑请加 --final。调参请用 --split validation。")
    if cfg.get("method") == "oracle" and cfg["data"]["split"] not in ORACLE_SPLITS:
        sys.exit(f"Oracle 是读了标签的诊断，只许在 {ORACLE_SPLITS} 上跑")  # 建输出目录前就拦下
    method = cfg.setdefault("method", "agent")
    budget = Budget(**cfg["budget"])
    check_budget(method, budget)  # 在建输出目录之前就检查，配错了不留半成品
    if method in NEEDS_RETRIEVAL and not cfg.get("retrieval"):
        sys.exit(f"{method} 需要 retrieval 配置")

    # 代码版本：有未提交改动时 commit hash 代表不了实际跑的代码，默认拒绝运行
    status = _git("status", "--porcelain")
    dirty = bool(status.strip())
    if dirty and not args.allow_dirty:
        sys.exit("工作区有未提交的改动，commit hash 无法代表本次代码。请先提交，或加 --allow-dirty 调试运行。")
    run_id = (f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}-{cfg['data']['split']}{sampling_tag}"
              + ("-dirty" if dirty else ""))
    out = ROOT / "outputs" / "runs" / run_id
    out.mkdir(parents=True)
    commit = _git("rev-parse", "HEAD").strip()
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    if dirty:
        # 未跟踪的新文件不在 diff 里，所以把 status 一起存下
        (out / "git_diff.patch").write_text(f"# git status\n{status}\n# git diff HEAD\n{_git('diff', 'HEAD')}",
                                            encoding="utf-8")

    base_llm, llm_info = build_llm(cfg["llm"])
    llm = RetryingLLM(base_llm, **cfg["llm"].get("retry", {}))
    tool, retrieval_info = build_tool(cfg.get("retrieval") if method in NEEDS_RETRIEVAL else None)
    data_dir = ROOT / cfg["data"]["dir"]
    split = cfg["data"]["split"]
    questions = read_jsonl(data_dir / "questions" / f"{split}.jsonl")[:args.limit]
    # ⚠️ Oracle 诊断：防泄漏规则的唯一例外，答题前读金标段落（只许 validation / debug，见 evaluation/oracle.py）
    gold_docs = load_gold_docs(data_dir, split) if method == "oracle" else {}
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    # 结果由 代码 + 配置 + 数据 + 环境 共同决定，四样都记下
    random.seed(cfg["seed"])
    cfg["run"] = {
        "run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "dirty": dirty, "argv": sys.argv,
        "data_manifest": manifest, "retrieval": retrieval_info, "llm_server": llm_info,
        # 模型看到的原文；改提示词不用翻 git 就能对比两次运行
        "system_prompt": (AGENT_PROMPTS if method == "agent" else ANSWER_ONLY_PROMPTS).system,
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                **{pkg: version(pkg) for pkg in ("pydantic", "pyyaml", "bm25s", "openai", "transformers")}},
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    trajectories, consecutive_errors = [], 0
    for i, q in enumerate(questions, 1):
        traj = run_method(method, q["qid"], q["question"], llm, tool, budget, gold_docs.get(q["qid"]))
        trajectories.append(traj)
        consecutive_errors = consecutive_errors + 1 if traj.stop_reason == StopReason.ERROR else 0
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            sys.exit(f"连续 {consecutive_errors} 题模型服务失败（最近一次：{traj.error}），中止运行。"
                     f"已跑 {i}/{len(questions)} 题，输出目录 {out} 不完整，不要用。")
        if i % 20 == 0:
            print(f"[{i}/{len(questions)}]", flush=True)

    # ---- 以下是评测：这时才读答案文件 ----
    labels = {l["qid"]: l for l in read_jsonl(data_dir / "labels" / f"{split}.jsonl")}
    records = []
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for traj in trajectories:
            label = labels[traj.qid]
            correct = exact_match(traj.final_answer, label["answer"])
            f1 = f1_score(traj.final_answer, label["answer"])
            state = traj.budget_state
            # 证据召回：整条轨迹里模型看到的段落（含答题前给的证据），覆盖了几个金标段落。把"搜得好不好"和"答得好不好"分开
            retrieved = {d.doc_id for s in traj.steps if s.observation is not None for d in s.observation.docs}
            if traj.context is not None:
                retrieved |= {d.doc_id for d in traj.context.observation.docs}
            gold_ids = set(label.get("gold_doc_ids", []))
            # 成本：假模型没有 token 数（None），这时整题记 None 而不是 0
            prompt_tokens = [s.prompt_tokens for s in traj.steps]
            completion_tokens = [s.completion_tokens for s in traj.steps]
            has_tokens = None not in prompt_tokens and None not in completion_tokens
            context_ms = traj.context.latency_ms if traj.context is not None else 0.0
            records.append({
                "qid": traj.qid, "correct": correct, "f1": f1, "error": traj.stop_reason == StopReason.ERROR,
                "stop_reason": traj.stop_reason.value, "prediction": traj.final_answer, "gold": label["answer"],
                "type": label.get("type"),
                "evidence_recall": len(gold_ids & retrieved) / len(gold_ids) if gold_ids else None,
                "turns": state.turns_used, "search_calls": state.search_calls_used,
                "search_attempts": state.search_attempts,
                "new_docs": sum(s.num_new_docs for s in traj.steps),
                "format_errors": sum(1 for s in traj.steps
                                     if s.observation is not None and s.observation.error_code in FORMAT_ERRORS),
                "unclosed_calls": sum(s.unclosed_tool_call for s in traj.steps),
                "ignored_suffixes": sum(bool(s.ignored_suffix) for s in traj.steps),
                "repaired_quotes": sum(s.repaired_quotes for s in traj.steps),
                "error_message": traj.error,
                "prompt_tokens": sum(prompt_tokens) if has_tokens else None,
                "completion_tokens": sum(completion_tokens) if has_tokens else None,
                "latency_ms": context_ms + sum(s.llm_latency_ms + s.tool_latency_ms for s in traj.steps),
                "llm_call_ms": [s.llm_latency_ms for s in traj.steps],
            })
            f.write(json.dumps({**traj.model_dump(mode="json"),
                                "eval": {"gold": label["answer"], "em": correct, "f1": f1,
                                         "gold_doc_ids": sorted(gold_ids), "evidence_recall": records[-1]["evidence_recall"]}},
                               ensure_ascii=False) + "\n")

    metrics = {"method": method, "diagnostic_only": method == "oracle", **aggregate(records)}
    metrics["valid"] = metrics["failure_rate"] <= MAX_FAILURE_RATE
    metrics["llm_retries"] = llm.num_retries
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # errors.csv：没答对的题都进来，按类别分开，方便 badcase 分析
    fields = ["qid", "type", "category", "stop_reason", "prediction", "gold", "f1", "evidence_recall", "turns",
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
