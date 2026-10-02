#!/usr/bin/env bash
# Day 8 收尾：Static RAG 和 Agent 换用 Dense 检索，和 BM25 版本在 validation 上对比。
# 解压在 tmux 里跑：
#   tmux new -d -s dense_baselines "bash experiments/run_baselines_dense.sh validation 2>&1 | tee /root/autodl-tmp/logs/baselines_dense.log"
# 日志放仓库外：写在仓库里会让工作区变脏，runner 会拒绝运行。要求代码已提交、两个服务在跑。
set -euo pipefail
split="${1:-validation}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for method in static_rag_dense agent_dense; do
  python -m evaluation.run_eval --config "configs/qwen3b_${method}.yaml" --split "$split"
done
