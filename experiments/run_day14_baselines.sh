#!/usr/bin/env bash
# Day 14：test 上补跑 B0 / B1 / B2（预先登记补充见 docs/DAY12_PREREG.md），并回放真实查询测单次检索里召回 vs 重排的成本。
#   tmux new -d -s day14 "bash experiments/run_day14_baselines.sh 2>&1 | tee /root/autodl-tmp/logs/day14.log"
# 先起 vLLM（8000）；本脚本自己起停检索服务（先 8100 HotpotQA，再 8101 2Wiki）。运行期间不要改仓库里的文件
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsr1
cd "$(dirname "$0")/.."
start_retriever() {  # $1 = 数据集前缀（hotpot_pool / 2wiki_pool），$2 = 端口
  tmux kill-session -t "retriever_$2" 2>/dev/null || true
  tmux new -d -s "retriever_$2" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 && cd $(pwd) && \
    python -m retrieval.server --index indexes/$1_v1_bm25 --dense-index indexes/$1_v1_e5 \
    --reranker /root/autodl-tmp/hf_models/bge-reranker-base --port $2"
  until curl -s -m 2 "localhost:$2/health" | grep -q ok; do sleep 5; done
}
for ds in hotpotqa 2wiki; do
  if [ "$ds" = hotpotqa ]; then idx=hotpot_pool; port=8100; p=""; else idx=2wiki_pool; port=8101; p="2wiki_"; fi
  start_retriever "$idx" "$port"
  # 回放：在没有其他负载时测，用 Day 12 test 运行里实际执行过的检索查询
  runs=$(ls -d outputs/runs/*-qwen3b-${p//_/-}static-rag-dense-rerank-test outputs/runs/*-qwen3b-${p//_/-}cascade-always-agent-test)
  mkdir -p "outputs/runs/day14-rerank-cost-$ds"
  python -m experiments.day14_rerank_cost --url "http://127.0.0.1:$port" --runs $runs --n 300 \
    --out "outputs/runs/day14-rerank-cost-$ds/rerank_cost.json"
  for cfg in direct static_rag agent; do
    python -m evaluation.run_eval --config "configs/qwen3b_${p}${cfg}.yaml" --split test --final
  done
  tmux kill-session -t "retriever_$port"
done
