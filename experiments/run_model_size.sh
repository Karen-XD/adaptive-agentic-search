#!/usr/bin/env bash
# 模型尺寸对照（3B vs 7B）：同一台机器、同一份代码，在 validation 上跑 V2 的 5 组，只换模型。
#   bash experiments/run_model_size.sh 3b hotpotqa    vLLM 起 3B（served name qwen2.5-3b-instruct）+ 检索服务 8100
#   bash experiments/run_model_size.sh 7b 2wiki       vLLM 起 7B（served name qwen2.5-7b-instruct）+ 检索服务 8101
# 3B 也重跑：换卡后同一配置的 EM 能差 1.5 个点（docs/PROGRESS.md 迁移验收），只和同机的运行比。
# 只跑 validation（Oracle 本来就拒绝 test）；test 要先预先登记。汇总：python -m experiments.model_size_summary
set -euo pipefail
size="${1:?usage: run_model_size.sh 3b|7b hotpotqa|2wiki}"
dataset="${2:?usage: run_model_size.sh 3b|7b hotpotqa|2wiki}"
case "$size" in 3b|7b) ;; *) echo "unknown size $size"; exit 1 ;; esac
case "$dataset" in
  hotpotqa) p="" ;;
  2wiki) p="2wiki_" ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
# 先跑不检索的两组：服务端模型不对时，第一个请求就会报模型不存在，不浪费检索
for cfg in direct oracle static_rag_dense_rerank cascade_gap cascade_always_agent; do
  python -m evaluation.run_eval --config "configs/qwen${size}_${p}${cfg}.yaml" --split validation
done
