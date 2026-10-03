#!/usr/bin/env bash
# Day 11 多 seed：分差门控和 agent 探测各跑 3 个采样 seed（temperature 0.7），和同一 seed 的 B3 配对，按"多数 seed 显著才算显著"。
# 采样下离线推算不再精确，必须在线跑。HotpotQA 的 B3 seed 运行 Day 9 已有（run_seeds_b3.sh），2Wiki 的 B3 这里补跑。
#   bash experiments/run_seeds_cascade.sh hotpotqa   检索服务 8100
#   bash experiments/run_seeds_cascade.sh 2wiki      检索服务 8101
#   tmux new -d -s seeds_cascade "bash experiments/run_seeds_cascade.sh hotpotqa 2>&1 | tee /root/autodl-tmp/logs/seeds_cascade_hotpotqa.log"
# 先起对应的检索服务和 vLLM；运行期间不要改仓库里的文件
set -euo pipefail
dataset="${1:?usage: run_seeds_cascade.sh hotpotqa|2wiki}"
temperature="${2:-0.7}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
case "$dataset" in
  hotpotqa) cfgs="cascade_gap cascade_always_agent" ;;
  2wiki) cfgs="2wiki_static_rag_dense_rerank 2wiki_cascade_gap 2wiki_cascade_always_agent" ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
for seed in 1 2 3; do
  for cfg in $cfgs; do
    python -m evaluation.run_eval --config "configs/qwen3b_${cfg}.yaml" --split validation \
      --temperature "$temperature" --seed "$seed"
  done
done
