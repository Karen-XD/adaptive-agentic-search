#!/usr/bin/env bash
# 新机器上重建数据盘（2026-10-05 从 RTX 4090 迁到 A800 80GB 时写；交接说明见 docs/MIGRATION.md）。
# 幂等：已存在且校验通过的东西直接跳过，中途断了重跑即可。任何严格校验不通过都会停下并说明原因，不带着错数据往下走。
#
# 前提：仓库已经 clone 到 /root/autodl-tmp/adaptive-agentic-search（见 docs/MIGRATION.md 第 2.2 节）。
# 用法：
#   mkdir -p /root/autodl-tmp/logs
#   tmux new -d -s restore "bash /root/autodl-tmp/adaptive-agentic-search/scripts/migration/restore_data_disk.sh 2>&1 | tee /root/autodl-tmp/logs/restore.log"
#   只跑某几步：bash scripts/migration/restore_data_disk.sh models raw
# 步骤（默认按顺序全跑）：dirs stash models raw data indexes fingerprint tests
# 额外步骤：v3 = 下载 Qwen2.5-0.5B-Instruct（V3 调通训练链路用，默认不跑）
set -euo pipefail
DISK=/root/autodl-tmp
REPO=$DISK/adaptive-agentic-search
STASH=/root/migration_stash          # 关机前放在系统盘、随镜像带过来的东西
LOG=$DISK/logs
MIG=$REPO/scripts/migration
# ~/.bashrc 里也有这几个，但非交互 shell 不会加载，这里显式设置
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=$DISK/cache/huggingface PIP_CACHE_DIR=$DISK/cache/pip

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] 失败：$*" >&2; exit 1; }

fetch() {  # fetch <url> <目标文件>：aria2c 16 线程 + 断点续传，失败重试 3 次；已完整下载过就跳过（校验在每步最后统一做）
  local url=$1 dst=$2
  if [ -s "$dst" ] && [ ! -f "$dst.aria2" ]; then return 0; fi  # .aria2 是 aria2c 未完成下载的控制文件
  mkdir -p "$(dirname "$dst")"
  for i in 1 2 3; do
    if aria2c -x16 -s16 -c --console-log-level=warn --summary-interval=0 --allow-overwrite=true \
         -d "$(dirname "$dst")" -o "$(basename "$dst")" "$url" >/dev/null 2>&1; then
      say "  ok $(basename "$dst")  $(stat -c%s "$dst") bytes"
      return 0
    fi
    say "  下载失败，第 $i 次重试：$url"
    sleep 5
  done
  die "下载失败：$url"
}

conda_dsr1() {
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate dsr1
}

step_dirs() {
  mkdir -p "$DISK/.cache_root" "$DISK/cache/huggingface" "$DISK/cache/pip" "$LOG" "$DISK/hf_models" "$DISK/checkpoints"
  # 系统盘上的软链接随镜像带过来，目标在数据盘；/root/.cache 指向的目录不存在时很多工具会报错
  for pair in "/root/adaptive-agentic-search:$REPO" "/root/hf_models:$DISK/hf_models" \
              "/root/.cache:$DISK/.cache_root" "/root/Search-R1/hf_models:$DISK/hf_models"; do
    src=${pair%%:*}; dst=${pair#*:}
    if [ -L "$src" ]; then
      [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ] || say "  注意：$src 指向 $(readlink "$src")，预期 $dst"
    elif [ -e "$src" ]; then
      say "  注意：$src 存在但不是软链接，没有改动它"
    else
      ln -s "$dst" "$src" && say "  新建软链接 $src -> $dst"
    fi
  done
  say "  GPU：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
  df -h / "$DISK" | tail -2
}

step_stash() {
  if [ ! -d "$STASH" ]; then
    say "  没有 $STASH（镜像里没带过来？）：实验输出没法恢复；微调检查点要按 PROGRESS.md 重训（sha 会和预先登记的不同）"
    return 0
  fi
  # 1) 实验输出 outputs/runs：报告里所有数字的原始轨迹。新 clone 的仓库里只有 .gitkeep
  if [ -f "$STASH/outputs_runs.tar.gz" ] && [ "$(ls "$REPO/outputs/runs" | wc -l)" -eq 0 ]; then
    tar -xzf "$STASH/outputs_runs.tar.gz" -C "$REPO/outputs"
  fi
  say "  outputs/runs：$(ls "$REPO/outputs/runs" | wc -l) 个运行（迁移前 121 个）"
  # 2) Day 15 微调重排检查点 last：商品 test 预先登记的就是它，sha 必须一致
  if [ ! -f "$DISK/checkpoints/esci_reranker_v1/last/model.safetensors" ]; then
    mkdir -p "$DISK/checkpoints/esci_reranker_v1"
    cp -r "$STASH/esci_reranker_v1/last" "$STASH/esci_reranker_v1/train_meta.json" "$DISK/checkpoints/esci_reranker_v1/"
  fi
  grep "/checkpoints/" "$MIG/sha256_assets.txt" | sha256sum -c --quiet - || die "微调检查点校验不通过"
  say "  微调检查点校验通过"
}

QWEN3B="config.json generation_config.json merges.txt model-00001-of-00002.safetensors model-00002-of-00002.safetensors
        model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json"
E5="1_Pooling/config.json config.json model.safetensors modules.json sentence_bert_config.json special_tokens_map.json
    tokenizer.json tokenizer_config.json vocab.txt"
BGE="config.json model.safetensors sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json"

step_models() {
  # 来源和迁移前相同：Qwen / e5 走 ModelScope，bge 走 hf-mirror（2026-10-05 抽测重新下载，和本地逐字节一致）
  for f in $QWEN3B; do fetch "https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct/resolve/master/$f" "$DISK/hf_models/Qwen2.5-3B-Instruct/$f"; done
  for f in $E5; do fetch "https://modelscope.cn/models/intfloat/e5-base-v2/resolve/master/$f" "$DISK/hf_models/e5-base-v2/$f"; done
  for f in $BGE; do fetch "https://hf-mirror.com/BAAI/bge-reranker-base/resolve/main/$f" "$DISK/hf_models/bge-reranker-base/$f"; done
  grep "/hf_models/" "$MIG/sha256_assets.txt" | sha256sum -c --quiet - \
    || die "模型文件校验不通过：上面标 FAILED 的文件删掉后重跑本步（bash $0 models）"
  say "  模型校验通过（Qwen2.5-3B-Instruct / e5-base-v2 / bge-reranker-base）"
}

step_raw() {
  for f in train-00000-of-00002.parquet train-00001-of-00002.parquet validation-00000-of-00001.parquet; do
    fetch "https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/$f" "$REPO/data/raw/hotpotqa_distractor/$f"
  done
  for f in train.parquet dev.parquet; do
    fetch "https://hf-mirror.com/datasets/xanhho/2WikiMultihopQA/resolve/main/$f" "$REPO/data/raw/2wiki/$f"
  done
  # ESCI 的 products 有 1.1GB，迁移前在 4090 上下载时断过一次；aria2c -c 会续传
  for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet; do
    fetch "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/$f" "$REPO/data/raw/esci/$f"
  done
  (cd "$REPO" && grep " data/raw/" "$MIG/sha256_repo_data.txt" | sha256sum -c --quiet -) \
    || die "原始数据校验不通过：上面标 FAILED 的文件删掉后重跑本步（bash $0 raw）"
  say "  原始数据校验通过（HotpotQA / 2Wiki / ESCI）"
}

step_data() {
  conda_dsr1
  cd "$REPO"
  # manifest.json 是每个准备脚本最后写的文件，用它判断上次是否跑完
  [ -f data/hotpotqa/v1/manifest.json ] || { say "  准备 HotpotQA"; python -m data_prep.prepare_hotpot > "$LOG/prep_hotpot.log" 2>&1 || die "见 $LOG/prep_hotpot.log"; }
  [ -f data/2wiki/v1/manifest.json ] || { say "  准备 2Wiki"; python -m data_prep.prepare_2wiki > "$LOG/prep_2wiki.log" 2>&1 || die "见 $LOG/prep_2wiki.log"; }
  [ -f data/esci/v1/manifest.json ] || { say "  准备 ESCI"; python -m data_prep.prepare_esci > "$LOG/prep_esci.log" 2>&1 || die "见 $LOG/prep_esci.log"; }
  # 2026-10-05 在旧机器上实测：三个准备脚本重跑，输出和原来逐字节一致（种子固定、先排序再抽样）
  grep -v " data/raw/" "$MIG/sha256_repo_data.txt" | sha256sum -c --quiet - \
    || die "准备好的数据和迁移前不一致：划分变了，旧结果和新实验不可比。先查原因，不要往下做"
  say "  语料和划分校验通过（和迁移前逐字节一致）"
}

build_bm25() {  # build_bm25 <语料> <索引目录>
  [ -f "$2/index_meta.json" ] && return 0
  say "  建 BM25 $2"
  python -m retrieval.bm25 build --corpus "$1" --index "$2" > "$LOG/build_$(basename "$2").log" 2>&1 || die "见 $LOG/build_$(basename "$2").log"
}

build_dense() {  # build_dense <语料> <索引目录>；要 GPU，vLLM 占着显存时先停掉
  [ -f "$2/index_meta.json" ] && return 0
  say "  建 Dense $2（GPU，每个几分钟）"
  # --model 必须是这个绝对路径：服务加载索引时按 index_meta.json 里记的路径找查询编码器
  python -m retrieval.dense build --corpus "$1" --index "$2" --model "$DISK/hf_models/e5-base-v2" \
    > "$LOG/build_$(basename "$2").log" 2>&1 || die "见 $LOG/build_$(basename "$2").log"
}

step_indexes() {
  conda_dsr1
  cd "$REPO"
  build_bm25 data/hotpotqa/v1/corpus.jsonl indexes/hotpot_pool_v1_bm25
  build_bm25 data/2wiki/v1/corpus.jsonl indexes/2wiki_pool_v1_bm25
  build_bm25 data/esci/v1/corpus.jsonl indexes/esci_v1_bm25
  build_dense data/hotpotqa/v1/corpus.jsonl indexes/hotpot_pool_v1_e5
  build_dense data/2wiki/v1/corpus.jsonl indexes/2wiki_pool_v1_e5
  say "  索引就绪：$(ls indexes | tr '\n' ' ')"
}

step_fingerprint() {
  # 索引文件不比字节（BM25 词表编号每次重建都不同），比检索结果：用迁移前存下的指纹对比
  conda_dsr1
  cd "$REPO"
  if python scripts/migration/retrieval_fingerprint.py check --ref "$MIG/retrieval_fingerprint.json" 2>&1 \
       | grep -v -i warning | tee "$LOG/fingerprint_check.log"; then
    say "  检索指纹全部通过"
  else
    say "  检索指纹没有全部通过（见上表）：BM25 不通过必须查原因；Dense 略低于 95% 可能是换卡后 fp16 数值差异，记进 PROGRESS.md 再决定"
  fi
}

step_tests() {
  conda_dsr1
  cd "$REPO"
  python -m pytest tests/ -q 2>&1 | tail -3   # 迁移前：171 passed
}

step_v3() {
  # V3 先用 0.5B 调通 GRPO 训练链路（几分钟一轮），再上 3B；文件清单和 3B 相同，只是权重只有一个分片
  for f in config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json; do
    fetch "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/master/$f" "$DISK/hf_models/Qwen2.5-0.5B-Instruct/$f"
  done
  say "  Qwen2.5-0.5B-Instruct 就绪（没有迁移前的 sha 可比，是新下载的）"
}

[ -d "$REPO/.git" ] || die "$REPO 不是 git 仓库：先按 docs/MIGRATION.md 第 2.2 节 clone"
STEPS=${*:-dirs stash models raw data indexes fingerprint tests}
for s in $STEPS; do
  say "== $s"
  "step_$s"
done
say "全部完成：$STEPS"
