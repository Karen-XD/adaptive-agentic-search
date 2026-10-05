# 迁移交接：RTX 4090 → A800 80GB（2026-10-05 写，迁移前在旧机器上演练通过）

> **✅ 2026-10-06 已在 A800 上执行完毕**，验收结果和换卡复现性检查见 `docs/PROGRESS.md`「当前位置」。和本文预期不同的两点：Dense 指纹前 10 名顺序一致率 90.5% / 94.0%（低于 95%，前 3 名 100% 一致，判可接受）；下载比预计慢很多（GitHub 约 30KB/s / 连接），总耗时约 2 小时 20 分钟。下面保留原文，供以后再迁移时参考。

> **给新机器上的 Claude**：先完整读完本文件再动手。
> 数据盘不进镜像，所以新机器上 `/root/adaptive-agentic-search` 是一个指向空位置的软链接，`CLAUDE.md`、`docs/PROGRESS.md` 都还不存在。
> 本文件在系统盘上有一份副本 `/root/migration_stash/README_FIRST.md`，仓库 clone 下来之前先读它。
> 按第 2 节做完、第 3 节验收全部通过之后，再按第 5 节接着做项目。

## 0. 背景（30 秒）

- 项目：自适应 Agentic Search，秋招搜广推算法岗的面试项目。用户是 Agent 新手，**全部用中文回答**，宏观优先；协作方式见 `CLAUDE.md`
- 进度：V1（Day 1～7）、V2 QA（Day 8～14，标签 `v2-adaptive-qa`）、商品搜索（Day 13～15）全部完成。报告：`docs/V1_REPORT.md`、`docs/V2_REPORT.md`、`docs/COMMERCE_REPORT.md`
- 为什么迁移：下一步是 V3（GRPO 强化学习训练）。Qwen2.5-3B 全参数训练的更新阶段峰值约 55GB，4090 的 24GB 放不下；A800 80GB 够用（估算见 `docs/PROGRESS.md`「V3（GRPO）资源评估」）
- 迁移时 GitHub 上的最新 commit：见第 2.2 节 clone 后的 `git log`；迁移交接本身的 commit 信息是 "Add migration handoff"

## 1. 什么跟着镜像过来了，什么没有

**系统盘（随镜像过来，不用动）**：

| 内容 | 位置 |
|---|---|
| conda 环境 `verl_env`（vLLM 服务 + V3 训练，**不要往里装包**）、`dsr1`（本项目代码） | `/root/miniconda3/envs/` |
| Search-R1 旧 checkout（`verl_env` 以可编辑方式装的 verl 就在这里）、官方 NQ + HotpotQA 训练数据、Qwen 下载脚本 | `/root/Search-R1/` |
| GitHub SSH 密钥（推送用，迁移前已验证可用） | `~/.ssh/id_ed25519` |
| 环境变量（HF 镜像、缓存目录） | `~/.bashrc`（非交互 shell 不会自动加载） |
| Claude 的记忆文件 | `~/.claude/projects/.../memory/` |
| **迁移暂存**：实验输出 `outputs_runs.tar.gz`（41MB，121 个运行）、Day 15 微调重排检查点 `esci_reranker_v1/last` + `train_meta.json`（1.1GB）、本文件副本 `README_FIRST.md` | `/root/migration_stash/` |
| 软链接 `/root/adaptive-agentic-search`、`/root/hf_models`、`/root/.cache`、`/root/Search-R1/hf_models` | 目标都在数据盘，恢复脚本会建好目标目录 |

**数据盘（丢失，由恢复脚本重建）**：

| 内容 | 大小 | 怎么回来 |
|---|---|---|
| 项目仓库 | — | `git clone`（第 2.2 节） |
| 原始数据：HotpotQA / 2Wiki / ESCI | 1.8GB | hf-mirror、GitHub 下载，sha256 校验 |
| 语料 + 划分（`data/*/v1/`） | 1GB | 重跑准备脚本，**和迁移前逐字节一致**（迁移前实测） |
| 模型：Qwen2.5-3B-Instruct、e5-base-v2、bge-reranker-base | 7.3GB | ModelScope / hf-mirror 下载，sha256 校验 |
| 索引：BM25 × 3、Dense × 2 | 4.7GB | 重建，再用检索指纹对比检索结果 |
| 实验输出 `outputs/runs` | 222MB | 从暂存解压 |
| 微调重排检查点 `last` | 1.1GB | 从暂存复制，sha256 校验（必须是 `8b7c60ef…`，商品 test 预先登记的就是它） |
| 不恢复：检查点 `best`、日志、缓存、空的 `search-agent` 环境 | — | 用不到 |

## 2. 新机器上的步骤

### 2.1 开机自检

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv   # 应是 A800 80GB（若只有 40GB，3B 全参放不下，先告诉用户）
df -h / /root/autodl-tmp                                  # 数据盘恢复完约占 18GB；V3 每个全参检查点约 12GB，空间不够先和用户商量扩容
free -g; nproc
ssh -T git@github.com                                     # 应显示 "Hi Karen-XD!"
ls /root/migration_stash                                  # 应有 README_FIRST.md、outputs_runs.tar.gz、esci_reranker_v1/
```

### 2.2 clone 仓库（含子模块）

```bash
mkdir -p /root/autodl-tmp/logs && cd /root/autodl-tmp
git clone --recurse-submodules git@github.com:Karen-XD/adaptive-agentic-search.git
cd /root/adaptive-agentic-search && git log --oneline -3     # 软链接现在应该能进去了
```

### 2.3 跑恢复脚本（约 30～60 分钟，主要花在下载和建 Dense 索引上）

```bash
tmux new -d -s restore "bash /root/autodl-tmp/adaptive-agentic-search/scripts/migration/restore_data_disk.sh 2>&1 | tee /root/autodl-tmp/logs/restore.log"
tail -f /root/autodl-tmp/logs/restore.log      # 最后一行是"全部完成：..."才算完
```

脚本 `scripts/migration/restore_data_disk.sh` 按顺序做 8 步，**幂等**：已完成且校验通过的直接跳过，断了就重跑；严格校验不通过会停下并说明原因。

| 步骤 | 做什么 | 校验 |
|---|---|---|
| dirs | 建数据盘目录、确认软链接 | 打印 GPU 和磁盘 |
| stash | 解压实验输出、复制微调检查点 | 检查点 sha256 |
| models | 下载 3 个模型（aria2c 16 线程、断点续传、失败重试 3 次） | sha256（`scripts/migration/sha256_assets.txt`） |
| raw | 下载 7 个原始数据文件 | sha256（`scripts/migration/sha256_repo_data.txt`） |
| data | 跑 3 个数据准备脚本 | 全部语料和划分 sha256，**必须和迁移前逐字节一致** |
| indexes | 建 BM25 × 3（CPU）、Dense × 2（GPU，每个几分钟） | — |
| fingerprint | 检索指纹对比（`scripts/migration/retrieval_fingerprint.py`） | BM25 必须 100% 一致；Dense ≥ 95% |
| tests | pytest | 迁移前 171 passed |

单独跑某几步：`bash scripts/migration/restore_data_disk.sh models raw`。额外步骤 `v3` 下载 Qwen2.5-0.5B-Instruct（第 5 节要用），默认不跑。

**迁移前演练（2026-10-05 15:22，旧机器）**：8 步全部通过；数据准备脚本在临时目录重跑，23 个语料 / 划分文件和原来逐字节一致；BM25 在临时目录重建，词表编号不同（Python 哈希随机化）但 300 条查询的检索结果和分数完全一致；抽测从 ModelScope / hf-mirror 重新下载的模型文件，和本地逐字节一致。

## 3. 验收标准（全部满足才算迁移完成）

1. 恢复脚本最后打印"全部完成"，退出码 0
2. 模型、原始数据、语料和划分、微调检查点的 sha256 全部通过
3. 检索指纹：BM25 三个都是 100%；Dense / Dense + 重排 ≥ 95%。换了显卡，Dense 用 fp16 重新编码，数值可能有极小差异，近似并列的段落可能换位，**实际一致率记进 PROGRESS.md**
4. `pytest tests/ -q` → 171 passed
5. 换卡复现性检查（第 4 节）做完并记录

## 4. 换卡复现性检查（约 10 分钟，必须做）

**目的**：同样的代码、数据和配置，在 A800 上贪心解码的结果和 4090 上差多少。不同显卡的矩阵运算实现不同，vLLM 的贪心输出不保证逐字一致。这决定了 V3 能不能直接拿旧运行当对照，还是必须在新机器上重跑基线。

```bash
# 1) 起服务（命令和恢复清单相同，见 docs/PROGRESS.md「服务器重启后的恢复清单」）
tmux new -d -s vllm "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl_env \
  && python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3b-instruct --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --seed 0 --disable-log-requests --guided-decoding-backend lm-format-enforcer \
  2>&1 | tee /root/autodl-tmp/logs/vllm.log"
tmux new -d -s retriever "source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 && cd /root/adaptive-agentic-search \
  && python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --dense-index indexes/hotpot_pool_v1_e5 \
     --reranker /root/autodl-tmp/hf_models/bge-reranker-base --port 8100"
# 2) 在新机器上重跑 HotpotQA validation 的 B3（贪心，200 题，约 2 分钟）
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 && cd /root/adaptive-agentic-search
python -m evaluation.run_eval --config configs/qwen3b_static_rag_dense_rerank.yaml --split validation
# 3) 和 4090 上的同一组运行逐题对比
python -m experiments.compare_runs outputs/runs/20261002-133518-qwen3b-static-rag-dense-rerank-validation outputs/runs/<新运行>
```

记录：EM（4090 上是 0.405）、逐题答案一致的题数、EM 配对差的区间。
- 如果 EM 一样、绝大多数题答案逐字一致：旧运行可以当参考，但 **V3 的正式对照仍在新机器上重跑**（同一台机器、同一批题，最干净）
- 如果差异明显：先查原因（检索指纹？vLLM 版本？），再决定

**注意**：A800 上 vLLM 的 `--gpu-memory-utilization 0.85` 会占约 68GB。V3 训练时 vLLM 由 veRL 自己在进程里起，要先把这个服务停掉。

## 5. 迁移完成后：更新文档，然后进入 V3

### 5.1 先更新三处文档并 commit + push

- `CLAUDE.md`「环境事实」：GPU 型号和显存、vCPU、系统盘 / 数据盘大小（以 `nvidia-smi`、`nproc`、`df` 为准）
- `docs/PROGRESS.md`「当前位置」：迁移完成、检索指纹一致率、换卡复现性检查结果
- 记忆文件：把"迁移进行中"的那条改成"迁移完成"

### 5.2 V3 要做什么（和用户确认后再开始）

用户已决定在 A800 上做 V3。V3 要回答的问题是：**用 GRPO 训练，让模型自己学会"什么时候搜、搜几次"；"成本感知奖励"能不能在准确率不掉的前提下少搜**。计划见 `.claude/agentic_search_autumn_recruitment_plan_v2.md` 第三周（Day 18～20、Stop Point 3），资源估算见 `docs/PROGRESS.md`「V3（GRPO）资源评估」。

建议的顺序（每一步做完都更新 PROGRESS.md，讲清楚在做什么、为什么）：

1. **读 Search-R1 训练链路**：`third_party/Search-R1/train_grpo.sh`（官方 8 卡配置，要缩到单卡）、`verl/trainer/main_ppo.py`、`search_r1/llm_agent/generation.py`（多轮生成，检索结果拼回上下文、训练时不算 loss）、`verl/utils/reward_score/qa_em.py`（奖励只看 EM）。对照本项目 Agent 的提示词和动作格式，列出差异
2. **检索适配层**：Search-R1 训练时批量调用 `POST {retriever.url}`，请求 `{"queries": [...], "topk": 3, "return_scores": true}`，期望返回 `{"result": [[{"document": {"contents": "标题\n正文"}, "score": ...}, ...], ...]}`；本项目服务是单条 `POST /search`。写一个薄适配层，接到本项目的 HotpotQA 语料池 + Dense + 重排（和 B3 同一个检索栈），端口避开 8000
3. **训练数据**：HotpotQA train 的题，**剔除** validation / debug / test 的 qid（`data/hotpotqa/v1/labels/`），转成 Search-R1 的 parquet 格式，写断言防泄漏。答案只进奖励函数，不进提示词
4. **两组奖励**：只看答对（EM，= Search-R1 原版）vs 成本感知（EM − λ × 检索次数）。λ 等超参只在 validation 上定，跑 test 前预先登记（参照 `docs/DAY12_PREREG.md` 的写法）
5. **冒烟**：先 `restore_data_disk.sh v3` 下载 0.5B，跑 5～10 步把整条链路调通（几分钟一轮）；再 3B 跑 20～50 步，看奖励、KL、显存峰值、格式有效率、检索次数分布、能否续训
6. **正式训练**：两组各约 200 步（每步 16 题 × 5 条轨迹，预计每组约 3 小时）；用本项目评测器在同一批 validation / test 上和 B3、cascade 比质量和成本
7. 不能写未经验证的 RL 提升；如果冒烟反复失败，按计划的风险表停止 RL 投入

**租卡在计费**：长任务一律放 tmux；不用 GPU 的时候提醒用户关机；每个阶段 commit + push（数据盘不进镜像）。

## 6. 已知坑

- 非交互 shell 不加载 `~/.bashrc`：手动跑命令前 `export HF_ENDPOINT=https://hf-mirror.com`（恢复脚本里已经设了）
- vLLM 0.6.3 必须加 `--guided-decoding-backend lm-format-enforcer`，否则每个请求都 500
- 建 Dense 索引、微调、训练之前先停掉 vLLM 服务腾显存
- `verl_env` 不要装新包（会破坏 RL 依赖）；新依赖放 `dsr1`，或在数据盘上新建环境（系统盘只剩约 6GB）
- `data_prep.prepare_hotpot` 的输出参数是 `--out_dir`（其他两个是 `--out`）
- ESCI 的 products 文件 1.1GB，从 GitHub 下载可能中途断；脚本用 aria2c 续传，重跑即可
- 跑实验时工作区要干净（`run_eval` 发现未提交的改动会拒绝运行），实验期间不要改仓库文件
- 写长的进度说明、给用户的回复：**全部中文**，包括工具调用之间的简短说明
