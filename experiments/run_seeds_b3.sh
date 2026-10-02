#!/usr/bin/env bash
# Day 9 B3 稳健性：Dense / 重排相关的配置各跑 3 个采样 seed（temperature 0.7），按 V2 口径"多数 seed 显著才算显著"。
# BM25 版 Static RAG / Agent 的 seed 运行 Day 6 已有（run_seeds.sh），这里不重跑。
#   tmux new -d -s seeds_b3 "bash experiments/run_seeds_b3.sh validation 2>&1 | tee /root/autodl-tmp/logs/seeds_b3.log"
# 检索服务要带 --dense-index 和 --reranker；运行期间不要改仓库里的文件（runner 发现工作区脏了会拒绝后面的运行）
set -euo pipefail
split="${1:-validation}"
temperature="${2:-0.7}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for seed in 1 2 3; do
  for cfg in static_rag_dense static_rag_dense_rerank static_rag_hybrid_rerank agent_dense_rerank agent_hybrid_rerank; do
    python -m evaluation.run_eval --config "configs/qwen3b_${cfg}.yaml" --split "$split" \
      --temperature "$temperature" --seed "$seed"
  done
done
