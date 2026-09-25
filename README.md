# Adaptive Agentic Search

Adaptive Agentic Search with Cost-aware Retrieval Policy: a single agent with typed tools that dynamically decides
query rewrite, retriever (BM25 / Dense / Hybrid), whether to rerank, and when to stop — evaluated for
quality–cost trade-offs against strong fixed search funnels on HotpotQA, and transferred to Amazon ESCI product search.

## Layout

```text
configs/       experiment / budget / policy configs
agent/         state, policy, action schema, loop, stop rules, prompts
retrieval/     bm25, dense, hybrid (RRF), reranker, search gateway
data_prep/     dataset preparation and split validation (NQ, HotpotQA, ESCI)
evaluation/    QA / e-commerce / cost metrics, bootstrap
experiments/   baselines, ablations, trajectory analysis, plots
tests/
outputs/       runs/<run_id>/ (not committed), figures/
third_party/   Search-R1 (pinned submodule) + local patches
```

## Setup

```bash
git clone --recursive git@github.com:Karen-XD/adaptive-agentic-search.git
```
