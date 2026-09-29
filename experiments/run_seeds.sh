#!/usr/bin/env bash
# Day 6 解码波动：Static RAG 和 Agent 各用温度采样跑 3 个 seed，看"Agent − Static RAG"换一次采样还在不在。
# 主结果仍是贪心解码（temperature=0）；这里只测稳健性，不参与方法比较的主表。
# 用法：bash experiments/run_seeds.sh [split] [temperature]，默认 validation、0.7。先起检索服务和 vLLM。
# 运行期间不要改仓库里的文件：runner 发现工作区脏了会拒绝后面的方法（Day 6 踩过一次）
set -euo pipefail
split="${1:-validation}"
temperature="${2:-0.7}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for seed in 1 2 3; do
  for method in static_rag agent; do
    python -m evaluation.run_eval --config "configs/qwen3b_${method}.yaml" --split "$split" \
      --temperature "$temperature" --seed "$seed"
  done
done
