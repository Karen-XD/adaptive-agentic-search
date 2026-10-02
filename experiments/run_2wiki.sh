#!/usr/bin/env bash
# Day 10.4：2Wiki validation（每类题 200，共 800）上的端到端对比。分两批：
#   bash experiments/run_2wiki.sh b3       两个 B3 候选（Dense + 重排 / Hybrid + 重排），选出 2Wiki 的检索栈
#   bash experiments/run_2wiki.sh rewrite  在选出的检索栈上（qwen3b_2wiki_base.yaml 的 retrieval.method）跑：
#                                          原问题取前 6 条、两跳证据改写、两跳静态拆解
#   tmux new -d -s wiki "bash experiments/run_2wiki.sh b3 2>&1 | tee /root/autodl-tmp/logs/2wiki_b3.log"
# 先起 2Wiki 的检索服务（端口 8101，见 configs/qwen3b_2wiki_base.yaml）和 vLLM；运行期间不要改仓库里的文件
set -euo pipefail
stage="${1:?usage: run_2wiki.sh b3|rewrite}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
case "$stage" in
  b3) cfgs="static_rag_dense_rerank static_rag_hybrid_rerank" ;;
  rewrite) cfgs="static_rag_top6 two_hop_evidence two_hop_decompose" ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
for cfg in $cfgs; do
  python -m evaluation.run_eval --config "configs/qwen3b_2wiki_${cfg}.yaml" --split validation
done
