#!/usr/bin/env bash
# Day 12 主实验：test 上每组只跑一次（贪心，配置和 validation 一字不改）。组和判定规则见 docs/DAY12_PREREG.md。
#   bash experiments/run_day12_test.sh 2wiki      检索服务 8101
#   bash experiments/run_day12_test.sh hotpotqa   检索服务 8100
# 先起对应的检索服务和 vLLM；运行期间不要改仓库里的文件
set -euo pipefail
dataset="${1:?usage: run_day12_test.sh hotpotqa|2wiki}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
case "$dataset" in
  hotpotqa) p="" ;;
  2wiki) p="2wiki_" ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
for cfg in static_rag_dense_rerank cascade_gap cascade_always_agent cascade_always agent_dense_rerank; do
  python -m evaluation.run_eval --config "configs/qwen3b_${p}${cfg}.yaml" --split test --final
done
