#!/usr/bin/env bash
# Day 11 按需升级的在线验证（validation）。两个数据集的检索服务不同，分开跑：
#   bash experiments/run_cascade.sh hotpotqa   检索服务 8100（HotpotQA 索引）：always（验证离线推算）、gap、always_agent
#   bash experiments/run_cascade.sh 2wiki      检索服务 8101（2Wiki 索引）：gap、always_agent
#   tmux new -d -s cascade "bash experiments/run_cascade.sh hotpotqa 2>&1 | tee /root/autodl-tmp/logs/cascade_hotpotqa.log"
# 先起对应的检索服务和 vLLM；运行期间不要改仓库里的文件
set -euo pipefail
dataset="${1:?usage: run_cascade.sh hotpotqa|2wiki}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
case "$dataset" in
  hotpotqa) cfgs="cascade_always cascade_gap cascade_always_agent" ;;
  2wiki) cfgs="2wiki_cascade_gap 2wiki_cascade_always_agent" ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
for cfg in $cfgs; do
  python -m evaluation.run_eval --config "configs/qwen3b_${cfg}.yaml" --split validation
done
