#!/usr/bin/env bash
# Day 10 改写对照：检索栈都是 Dense + 重排（和 B3 相同）；作答提示词都和 B3 一字不差。
#   B3（original，1 次检索，前 3 条）已有：20261002-133518-qwen3b-static-rag-dense-rerank-validation
#   rewrite_rag                 1 次检索：静态改写代替原问题
#   static_rag_dense_rerank_top6 1 次检索，前 6 条：和两跳的组看到的段落数相同
#   two_hop_static              2 次检索：原问题 → 静态改写（不看证据）
#   two_hop_evidence            2 次检索：原问题 → 证据条件改写
#   tmux new -d -s rewrite "bash experiments/run_rewrite.sh validation 2>&1 | tee /root/autodl-tmp/logs/rewrite.log"
# 运行期间不要改仓库里的文件（runner 发现工作区脏了会拒绝后面的运行）
set -euo pipefail
split="${1:-validation}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for cfg in rewrite_rag static_rag_dense_rerank_top6 two_hop_static two_hop_evidence; do
  python -m evaluation.run_eval --config "configs/qwen3b_${cfg}.yaml" --split "$split"
done
