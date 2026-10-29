#!/usr/bin/env bash
# 从 ModelScope（国内源）下载一个模型仓库的全部文件，并按 ModelScope 接口给的 sha256 逐个校验。
#   bash scripts/download_model_modelscope.sh Qwen/Qwen2.5-7B-Instruct /root/autodl-tmp/hf_models/Qwen2.5-7B-Instruct
# 为什么不直接 huggingface-cli：hf-mirror 在 AutoDL 上只有约 0.5MB/s；ModelScope 单连接也限速，aria2c 多连接并行能快一个量级。
# -c 断点续传：中断后重跑同一条命令，已下完且校验通过的文件直接跳过。校验结果写进目标目录的 SHA256SUMS。
set -euo pipefail
repo="${1:?usage: download_model_modelscope.sh <org/name> <target_dir>}"
dir="${2:?usage: download_model_modelscope.sh <org/name> <target_dir>}"
api="https://modelscope.cn/api/v1/models/${repo}/repo/files?Revision=master&Recursive=true"
base="https://modelscope.cn/models/${repo}/resolve/master"
mkdir -p "$dir"

# 文件清单（路径 + sha256）来自 ModelScope 接口，不手写：换模型时分片数不同
manifest=$(curl -sS -m 30 "$api" | python3 -I -c '
import json, sys
for f in json.load(sys.stdin)["Data"]["Files"]:
    if f["Type"] == "blob" and f["Path"] != ".gitattributes":
        print(f["Path"], f["Sha256"])')
[ -n "$manifest" ] || { echo "empty file list from $api"; exit 1; }

: > "$dir/SHA256SUMS"
while read -r path sha; do
  if [ -f "$dir/$path" ] && [ ! -f "$dir/$path.aria2" ] && echo "$sha  $dir/$path" | sha256sum -c --status; then
    echo "  skip (verified): $path"
  else
    aria2c -x16 -s16 -c --console-log-level=warn --summary-interval=60 --allow-overwrite=true \
      -d "$dir" -o "$path" "$base/$path" < /dev/null  # 别让 aria2c 吃掉循环读的文件清单
    echo "$sha  $dir/$path" | sha256sum -c --status || { echo "SHA256 MISMATCH: $path"; exit 1; }
    echo "  ok: $path ($(stat -c%s "$dir/$path") bytes)"
  fi
  echo "$sha  $path" >> "$dir/SHA256SUMS"
done <<< "$manifest"
echo "ALL_FILES_VERIFIED: $repo -> $dir"
