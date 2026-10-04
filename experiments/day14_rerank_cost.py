"""Day 14 单次检索的成本拆分：回放 test 轨迹里的真实查询，测"向量召回"和"重排"各花多少时间、重排处理多少 token。

为什么要回放：轨迹里每次检索只记了总耗时（tool_latency_ms），没拆召回和重排；重排模型处理的 token 也没记。
检索服务在 rerank=true 时会返回 retrieve_ms / rerank_ms / num_candidates（retrieval/server.py），
重排的输入 token 数用同一个分词器、同样的截断规则（只截段落、max_length 512）对候选池离线算出来。

用法（先起对应数据集的检索服务）：
  python -m experiments.day14_rerank_cost --url http://127.0.0.1:8100 --runs outputs/runs/<test 运行> ... --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import httpx
from transformers import AutoTokenizer

RERANKER = "/root/autodl-tmp/hf_models/bge-reranker-base"


def collect_queries(runs: list[Path]) -> list[str]:
    """轨迹里所有实际执行过的检索查询：流程替模型搜的（context.searches / context.query）+ Agent 自己发的。"""
    out = []
    for run in runs:
        for t in map(json.loads, open(run / "trajectories.jsonl", encoding="utf-8")):
            ctx = t.get("context") or {}
            if ctx.get("searches"):
                out += [s["query"] for s in ctx["searches"] if s.get("query")]
            elif ctx.get("query"):
                out.append(ctx["query"])
            for s in t["steps"]:
                a = s.get("action") or {}
                if a.get("name") == "search" and (s.get("observation") or {}).get("ok"):
                    out.append(a["arguments"]["query"])
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--method", default="dense")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    queries = collect_queries([Path(r) for r in args.runs])
    sample = random.Random(0).sample(queries, min(args.n, len(queries)))
    tok = AutoTokenizer.from_pretrained(RERANKER)
    client = httpx.Client(timeout=30)
    pool_size = client.get(f"{args.url}/health").json().get("rerank_pool_size") or 20
    # 先热身：第一次请求要初始化 CUDA，不算进统计
    client.post(f"{args.url}/search", json={"query": "warmup", "top_k": 3, "method": args.method, "rerank": True})
    retrieve_ms, rerank_ms, total_ms, rerank_tokens, n_cands = [], [], [], [], []
    for q in sample:
        r = client.post(f"{args.url}/search", json={"query": q, "top_k": 3, "method": args.method, "rerank": True}).json()
        tm = r["timing"]
        retrieve_ms.append(tm["retrieve_ms"])
        rerank_ms.append(tm["rerank_ms"])
        total_ms.append(tm["retrieve_ms"] + tm["rerank_ms"])
        n_cands.append(tm["num_candidates"])
        pool = client.post(f"{args.url}/search", json={"query": q, "top_k": pool_size, "method": args.method,
                                                       "rerank": False}).json()["docs"]
        enc = tok([(q, f"{d['title']}\n{d['text']}") for d in pool], truncation="only_second", max_length=512)
        rerank_tokens.append(sum(len(ids) for ids in enc["input_ids"]))
    summary = {
        "url": args.url, "method": args.method, "num_queries_in_runs": len(queries), "num_sampled": len(sample),
        "pool_size": pool_size, "mean_candidates": statistics.mean(n_cands),
        "retrieve_ms": {"mean": statistics.mean(retrieve_ms), "p50": pct(retrieve_ms, .5), "p95": pct(retrieve_ms, .95)},
        "rerank_ms": {"mean": statistics.mean(rerank_ms), "p50": pct(rerank_ms, .5), "p95": pct(rerank_ms, .95)},
        "server_total_ms": {"mean": statistics.mean(total_ms), "p50": pct(total_ms, .5)},
        "rerank_input_tokens_per_search": {"mean": statistics.mean(rerank_tokens), "p50": pct(rerank_tokens, .5)},
        "rerank_share_of_server_time": sum(rerank_ms) / sum(total_ms),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
