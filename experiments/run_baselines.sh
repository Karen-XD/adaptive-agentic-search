#!/usr/bin/env bash
# 四种方法在同一个划分上依次跑完（默认 validation）。放 tmux 里跑，防断线：
#   tmux new -d -s baselines "bash experiments/run_baselines.sh validation 2>&1 | tee outputs/baselines.log"
# 先起检索服务（8100）和 vLLM（8000）；要求代码已提交（runner 会检查）。
set -euo pipefail
split="${1:-validation}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
for method in direct static_rag oracle agent; do
  python -m evaluation.run_eval --config "configs/qwen3b_${method}.yaml" --split "$split"
done
