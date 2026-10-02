#!/usr/bin/env bash
# Day 9：B3 强固定基线的两个候选（Dense + 重排、Hybrid + 重排），Static RAG 和 Agent 各跑一遍。
#   tmux new -d -s b3 "bash experiments/run_b3_candidates.sh validation 2>&1 | tee /root/autodl-tmp/logs/b3_candidates.log"
# 检索服务要带 --dense-index 和 --reranker；要求代码已提交、vLLM 在跑。
set -euo pipefail
split="${1:-validation}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for cfg in static_rag_dense_rerank static_rag_hybrid_rerank agent_dense_rerank agent_hybrid_rerank; do
  python -m evaluation.run_eval --config "configs/qwen3b_${cfg}.yaml" --split "$split"
done
