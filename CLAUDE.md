# adaptive-agentic-search 协作说明

执行计划：`.claude/agentic_search_autumn_recruitment_plan_v2.md`（以它为准，按 Day 推进）。

## 会话开始 / 结束

1. **开始时先读 `docs/PROGRESS.md`**，从"当前位置"和"下一步"接着做。
2. 每完成一个小步骤就更新 `docs/PROGRESS.md`（完成项、决策及原因、下一步）。
3. 讲过的概念、论文要点、面试问答写进 `docs/LEARNING_NOTES.md`。
4. 仓库在数据盘上，数据盘不进 AutoDL 镜像、实例释放即丢失 → 阶段性提醒用户 commit + push。

## 协作方式（用户是 Agent 新手，目标是秋招搜广推算法岗）

- **全部用中文回答**，英文术语首次出现时给中文解释。
- 目标是能在秋招面试里讲清这个项目。**宏观优先**：每一步讲清"在做什么 → 为什么 → 结论 → 在整个项目流程里的位置"，面试怎么讲；阶段性更新 `docs/PROJECT_OVERVIEW.md`（项目全景和当前运行流程）。
- 实现细节（边界条件、计数口径、异常分支等）由 Claude 自行决定，写进 `docs/PROGRESS.md` 决策记录，给用户一两句话的结论即可，不逐条提问。
- 思考题宁少勿滥（每步最多 1～2 道），只问有助于理解项目主线或应对面试的问题，不为了问而问。需要读论文时可以停下来布置阅读。
- 多用搜广推的类比（多路召回、融合、精排、算力分配）解释 Agent 概念。
- 不一次性堆大量代码；每段代码都要说明为什么这样设计。

## 环境事实（AutoDL，1× RTX 4090 24GB，16 vCPU）

- 系统盘 `/` 30G（剩余约 9G），**会**进镜像；数据盘 `/root/autodl-tmp` 50G，**不**进镜像。
- 项目本身放在数据盘（`/root/autodl-tmp` 下），`/root/adaptive-agentic-search` 是指向它的软链接；Claude 工具里显示的路径可能被映射成别的样子，以 `df` 结果为准。
- 缓存都指向数据盘：`HF_HOME=/root/autodl-tmp/cache/huggingface`、`PIP_CACHE_DIR=/root/autodl-tmp/cache/pip`、`HF_ENDPOINT=https://hf-mirror.com`（写在 `~/.bashrc`，非交互 shell 不会自动加载，需显式 export）。`/root/.cache` 是指向 `/root/autodl-tmp/.cache_root` 的软链接，新数据盘上要先 `mkdir -p` 目标目录。
- 模型下载：ModelScope + aria2c 多线程（参考 `/root/Search-R1/download_model_modelscope.sh`），放 `/root/autodl-tmp/hf_models/`。
- conda 环境分工：
  - `verl_env`（`/root/miniconda3/envs/verl_env`）：Search-R1 原版栈，torch 2.4.0+cu124、vLLM 0.6.3、flash-attn 2.6.3、pyserini（环境内自带 JDK 21）、faiss-gpu。用于 **vLLM 模型服务** 和 V3 的 RL。**不要往里装新包**，避免破坏 RL 依赖。其 `verl` 以可编辑方式装自 `/root/Search-R1`（另一份带本地修改的旧 checkout，不是本仓库子模块）。
  - `dsr1`（`/root/miniconda3/envs/dsr1`）：torch 2.6.0+cu124、transformers 4.51.3、bm25s、fastapi、openai、datasets、pydantic。用于 **本项目代码**（agent / 检索 / 评测 / 测试）。
  - `search-agent`（数据盘上，几乎为空）：暂不使用。
- 长任务（建索引、批量评测、训练）一律放 `tmux` 或 `nohup` 里跑，防止断线中断。

## 工程约定

- 数据、索引、模型、checkpoint、密钥不进 git（见 `.gitignore`）。
- `third_party/Search-R1` 子模块保持上游原样，本地修改放 `third_party/patches/`。
- 每次实验输出到 `outputs/runs/<run_id>/`：`config.yaml`、`git_commit.txt`、`metrics.json`、`trajectories.jsonl`、`errors.csv`。
- 防泄漏：gold answer / 测试集标签只由评测器读取，绝不进入 prompt 或检索工具；阈值和 prompt 只在 validation 上调。
