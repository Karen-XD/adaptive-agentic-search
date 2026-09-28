# 项目进度日志

> 每次会话开始先读本文件；每完成一个小步骤就更新。

## 当前位置

**Day 3 进行中（2026-09-28）。3.4 完成：四种方法（Direct / Static RAG / Agent / Oracle 诊断）共用一个入口，debug 集上各跑一次，出了第一张对比表。**

**下一步：3.5** —— 人工看约 30 条轨迹（重点：比较题 Agent 为什么不如 Static RAG、查询改写得失），修格式报错提示，然后在 validation（200 题）上批量跑四种方法，出第一张正式基线表。

> 2026-09-28 关机前：3.3 还没开始写代码，工作区已提交并 push。重启后按下面清单起**两个**服务（检索 + vLLM），再从 3.3 开始。
> 2026-09-28 恢复后：讲了 BM25（新增 "Who founded Apple" 词干化误合并的例子），用户回答了查询改写思考题，点评和实测排名见 LEARNING_NOTES「3.2 思考题」。结论：拆分方向对，但 `birth year` 式改写让两个实体的金标排名都变差，"实体名 + 类型词"才变好。

### 🔌 服务器重启后的恢复清单（2026-09-27 关机前写）

关机（非释放实例）后两个盘都在，代码、数据、索引、模型、记忆文件都还在；**只有 tmux 会话会消失**。

```bash
# 1. 确认资产都在（应输出 3 行都存在）
ls -d /root/autodl-tmp/adaptive-agentic-search/data/hotpotqa/v1 \
      /root/autodl-tmp/adaptive-agentic-search/indexes/hotpot_pool_v1_bm25 \
      /root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct

# 2. 重启检索服务（Day 3 跑实验前必须启动）
tmux new -d -s retriever "source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 \
  && cd /root/adaptive-agentic-search \
  && python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --port 8100"
curl -s http://127.0.0.1:8100/health   # 应返回 num_docs: 507494

# 3. 重启 vLLM 模型服务（Day 3 起，约 50s 就绪，显存占约 19GB）
#    --guided-decoding-backend 必须加：vLLM 0.6.3 默认后端 outlines 缺依赖，每个请求都会 500
mkdir -p /root/autodl-tmp/logs
tmux new -d -s vllm "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl_env \
  && python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3b-instruct --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --seed 0 --disable-log-requests --guided-decoding-backend lm-format-enforcer \
  2>&1 | tee /root/autodl-tmp/logs/vllm.log"
curl -s http://127.0.0.1:8000/v1/models   # 应列出 qwen2.5-3b-instruct

# 4. 自检
cd /root/adaptive-agentic-search && conda activate dsr1 && pytest tests/ -q   # 95 passed
```

若资产丢失（例如释放了实例），按本文件「数据与索引位置」一节的命令重建；模型用 `/root/Search-R1/download_model_modelscope.sh` 重新下载。

> 2026-09-27 用户反馈：讲解和提问要宏观优先（每步做什么 / 为什么 / 结论 / 全局位置），实现细节由 Claude 决定并记在决策表，不逐条提问。已写入 `CLAUDE.md` 和记忆；宏观全景见 `docs/PROJECT_OVERVIEW.md`。

## Day 3 子步骤

- [x] 3.1 vLLM 模型服务（`verl_env`，tmux `vllm`，端口 8000，OpenAI 兼容接口）。首个请求 6s 是预热，之后单次调用 130～440ms
- [x] 3.2 看原始输出（debug 前 8 题，temperature=0，不设停止词）
  - `role="tool"` 被模板渲染成 user 轮次 + `<tool_response>…</tool_response>`，连续多条合并进同一轮 → 现有拼回方式正确
  - 占位提示词：格式有效率 **0/8**，模型写成 `Search: Search({"query": ...})`（模仿提示词里的列表写法），但查询意图合理
  - 改用 Qwen 原生工具说明（`# Tools` + `<tools>` 函数签名，由对话模板生成）：**8/8**
  - 比较题里模型一轮写了 3 个调用，还没看到结果就作答（答错）→ 需要在 `</tool_call>` 截断
  - 模型每轮都自己结束，不会续写假的检索结果（和 Search-R1 用的 base 模型不同）
  - 复现脚本：`python -m experiments.day3_prompt_format_probe`（自包含，保留了占位提示词原文；复跑结果一致）
- [x] 3.3 真模型客户端 + 原生工具提示词 + 运行配置补全（2026-09-28）
  - `agent/llm.py`：`VLLMClient`（chat 接口、停止词、预热、错误分类：超时 / 连不上 / 5xx 可重试，4xx 不重试）；`generate` 改为返回 `Generation`（文字 + token 数 + 结束原因）
  - `agent/native_tools.py`：用代码生成原生工具说明；`tests/test_prompt_native.py` 用模型目录里的对话模板逐字比对
  - `Step` 新增 `prompt_tokens` / `completion_tokens` / `finish_reason` / `unclosed_tool_call`
  - runner：`type: vllm`、`--limit N`、连续 5 题模型服务失败就中止、`config.yaml` 记模型目录 / 服务端 vllm·torch·transformers 版本 / 系统提示词原文 / 预热耗时、`metrics` 记重试次数
  - `configs/qwen3b_agent_debug.yaml`；测试 53 → 72
  - debug 前 8 题（调试运行）：第一轮格式 8/8，但看到检索结果后 5/20 轮格式错，其中 4 个是 JSON 完整、漏了 `</tool_call>`（关停止词重放也一样，是模型自己输出了结束符）→ 解析器放宽这一种情况后格式错误率 18% → 4%，没作答 2 → 0
  - **正式运行 `20260928-212118-qwen3b-agent-debug`**（commit `bed5236`，debug 全 50 题，96s）：EM 0.28（宽松包含匹配 0.44）、证据召回 0.66、金标全部找齐 0.42、格式错误率 4.9%（补上结尾标签 8 次）、平均实际搜索 1.86 次；停止原因 answered 40 / forced 9 / no_answer 1；每题输入 token 中位数 2031、输出 162；每题端到端 P50 1.6s、P95 3.7s。**证据找齐的 21 题里仍答错 10 题** → 瓶颈不只在检索。debug 集只用于调试，数字不进结论
  - 贪心解码对输入极敏感：同一道题去掉数据里原有的末尾空格，第一轮查询就从 "Mary Gordon birth year" 变成两人合并的 "Mary Gordon birth year H. L. Mencken birth year" → Day 6 多 seed 看波动时要记住，单题结论不可靠
- [x] 3.4 B0 Direct / B1 Static RAG / Oracle 诊断；补 Token-F1、输入输出 token 数、P50/P95 延迟（2026-09-28）
  - `agent/methods.py`：四种方法共用循环、解析器、Budget；只能作答的三种方法提示词一字不差，只差证据
  - `evaluation/oracle.py`：评测侧读金标段落，拒绝 test；`CLAUDE.md` 记下这个例外（用户同意）
  - 配置改成基础配置 + 四个 `extends` 它的方法配置，解码参数、数据、预算只写一处（3.5 起改名为 `qwen3b_base.yaml` / `qwen3b_<方法>.yaml`）
  - 测试 72 → 95，三处改坏检查都能抓到（Static RAG 不计成本、Oracle 不排序/放行 test、Direct 用错提示词）
  - 正式运行（commit `cc0b7c1`，debug 50 题，只用于调试，不出结论）：

    | 方法 | EM | F1 | 证据召回 | 全找齐 | 搜索次数 | 输入 token | 输出 token | P50 / P95 延迟 |
    |---|---|---|---|---|---|---|---|---|
    | B0 Direct `220911` | 0.16 | 0.24 | 0 | 0 | 0 | 221 | 21 | 0.21s / 0.25s |
    | B1 Static RAG `220924` | 0.24 | 0.32 | 0.61 | 0.30 | 1 | 605 | 26 | 0.26s / 0.56s |
    | Agent `220956` | 0.28 | 0.36 | 0.66 | 0.42 | 1.86 | 2482 | 190 | 1.67s / 3.64s |
    | Oracle（诊断上限）`220941` | 0.52 | 0.70 | 1 | 1 | 0 | 442 | 22 | 0.21s / 0.27s |

  - 按题型：桥接题（36 题）EM Direct 0.06 → RAG 0.14 → Agent 0.22 → Oracle 0.53；比较题（14 题）Direct 0.43 → RAG 0.50 → **Agent 0.43** → Oracle 0.50
  - Agent vs Static RAG 逐题：Agent 对、RAG 错 7 题；RAG 对、Agent 错 5 题 → 50 题上差距在噪声范围内
  - Agent 与 3.3 的运行逐轮输出 50/50 完全一致 → 本次重构没有改变 Agent 行为
- [ ] 3.5 人工看约 30 条轨迹，再批量跑 validation

## Day 2 子步骤

- [x] 2.1 数据：从 HotpotQA distractor 原始数据重建语料池（507,494 段）+ test 500 / validation 200 / debug 50 + 数据清单；`validate_splits` 通过
- [x] 2.2 BM25 索引（bm25s + 英文词干化，建索引 59s、449MB）；检索体检：原问题搜一次，前 3 条找齐两个金标只有 28%（桥接题前 20 条也只有 50%）
- [x] 2.3 检索服务（FastAPI `/health` `/search`）+ HTTP 客户端；实验入口支持 mock / bm25 / http 三种检索、题目与答案分文件读取、证据召回指标
- [x] 2.4 端到端运行：检索服务 + `configs/bm25_debug.yaml` → `20260927-224225-bm25-debug`（commit `84c096e`）：证据召回 0.61，金标全部找齐 0.30，检索耗时中位数 10ms；准确率 0.02 无意义（规则假模型）
- 测试 53 个通过（新增 `tests/test_retriever.py`）

## 数据与索引位置（不进 git，实例释放会丢，可用脚本重建）

| 内容 | 路径 | 重建命令 |
|---|---|---|
| HotpotQA 原始数据（360MB） | `data/raw/hotpotqa_distractor/` | 从 hf-mirror 下载，sha256 见 manifest |
| 语料池 + 划分 + 清单（270MB） | `data/hotpotqa/v1/` | `python -m data_prep.prepare_hotpot` |
| BM25 索引（449MB） | `indexes/hotpot_pool_v1_bm25/` | `python -m retrieval.bm25 build --corpus data/hotpotqa/v1/corpus.jsonl --index indexes/hotpot_pool_v1_bm25` |
| Qwen2.5-3B-Instruct（6.17GB） | `/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct/` | `bash /root/Search-R1/download_model_modelscope.sh`（ModelScope + aria2c，约 5 分钟） |

## Day 1 子步骤

- [x] 1.0 锁定 Search-R1 上游 commit `598e61b`（记录在 `third_party/README.md`）
- [x] 1.0 盘点环境：GPU / conda 环境 / 磁盘 / 已有资产（2026-09-25）
- [x] 1.1 阅读 ReAct、Search-R1 论文和 `infer.py`，回答 4 个思考题（2026-09-27；第 2、3 题已纠正，要点见 LEARNING_NOTES）
- [x] 1.2 确认环境分工；在 `dsr1` 中补装 pytest（9.1.1，2026-09-27）
- [x] 1.3 定义 Action Schema、Budget 对象、异常返回格式（2026-09-27）
  - [x] `agent/schema.py`：Action（search / final_answer）、ErrorCode（四层）、Doc、Observation、Budget、BudgetState
  - [x] `agent/parser.py`：文字 → Action（语法层 + 结构层），11 种典型输入手动验证通过，1.5 转成正式测试
- [x] 1.4 Mock Search Tool + 最小循环：question → Action → Observation → final_answer（2026-09-27）
  - `retrieval/mock.py`（虚构语料、词重叠、同分按 doc_id 排）、`agent/llm.py`（LLM 接口 + ScriptedLLM）、`agent/prompts.py`（占位提示词）、`agent/loop.py`（策略层 + 执行层）
  - `schema.py` 补了 `search_attempts`、`StopReason`、`Step`、`Trajectory`
- [x] 1.5 冒烟测试：非法 JSON、未知工具、空结果、超过最大轮数（2026-09-27）
  - `tests/test_action_schema.py`（14）、`tests/test_mock_search.py`（4）、`tests/test_loop.py`（15），共 33 个通过；`python -m tests.test_action_schema` 可运行
  - 改坏检查：解析范围、查询归一化两处改坏都能被抓到；顺带修正了 `test_prompt_example_is_never_executed` 原先测不出"误执行"的问题
- [x] 1.6 产出第一条 JSONL 轨迹（记录 seed、模型、请求参数）（2026-09-27）
  - `evaluation/run_eval.py` + `configs/mock_v1.yaml` + `evaluation/qa_metrics.py` + `data_prep/mock/qa.jsonl`（5 道假题）
  - 首次运行 `20260927-210853-mock-v1-dirty`（调试运行），5 个文件齐全；准确率 0.2 无意义（规则假模型）
  - 正式运行 `20260927-211645-mock-v1`（commit `939e2aa`，工作区干净），结果与调试运行一致
- [x] Day 1 验收：冒烟测试 40 个通过、`python -m tests.test_action_schema` 可运行、JSONL 轨迹已产出、`nvidia-smi` 与 PyTorch CUDA 自检通过（RTX 4090，torch 2.6.0+cu124）

## 从 `infer.py` 发现的隐患（1.3～1.5 要逐条覆盖）

1. `while True` 无轮数上限（`cnt` 只计数）→ Budget 的 max_turns。
2. `get_query` 在包括 prompt 的整段文本上取最后一个 `<search>`，截断时会静默拿旧查询或示例词 "query" 去搜 → 只解析本轮新生成内容，解析失败返回明确错误。
3. `requests.post` 无超时、无异常处理 → 统一异常返回格式。
4. 空查询照搜 → 空查询 / 空结果测试。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-09-25 | 不用 wiki-18 全量语料，采用可控 HotpotQA 语料池 | ~~wiki-18 向量索引约 64GB，超过 50G 数据盘~~ → 2026-09-27 更新：数据盘可扩容，磁盘不再是理由；仍先用语料池，因为迭代快、符合 plan 分阶段设计；Day 20 左右扩容后用 HotpotQA fullwiki（约 520 万段）做鲁棒性验证 |
| 2026-09-25 | 环境分工（默认方案，用户可改）：`verl_env` 负责 vLLM 模型服务和 V3 RL；`dsr1` 跑项目代码；`search-agent` 暂不用 | 两个现成环境都能用 GPU；不往 RL 环境装新包以免破坏依赖；`dsr1` 在系统盘，会随镜像保存 |
| 2026-09-27 | 动作文字格式采用 Qwen2.5 原生 `<tool_call>{JSON}</tool_call>` | 标签负责定位和停止，JSON 负责多参数和明确报错；V1/V2 不训练模型，顺着模型已有习惯；Day 3 统计格式有效率再确认 |
| 2026-09-27 | 解析只看本轮新生成内容；训练、评测、推理共用同一个解析器 | `infer.py` 解析整段文本导致静默出错；训练和推理两套解析 = train-serve skew |
| 2026-09-27 | `final_answer` 作为一个动作；`top_k` 由 Budget 固定、不交给模型 | 每轮恰好一个动作，停止决策可记录可优化；和 Static RAG 证据量一致，比较公平 |
| 2026-09-27 | 校验分四层（语法 / 结构 / 策略 / 执行），各有错误码；"检索无结果"不算错误 | 便于 badcase 归因和 1.5 冒烟测试分层覆盖 |
| 2026-09-27 | 解析严格：没有 tool_call 的纯文字不当答案；所有方法（含 Direct / Static RAG）共用同一个解析器 | 宽松会把格式失误记成"决定停下"，污染停止统计；提取规则不同会让比较不公平。原始输出存轨迹，离线算宽松诊断指标 |
| 2026-09-27 | 轮数将尽时强制作答：占 `max_turns` 最后一轮，明确告诉模型只能作答；`stop_reason` 分 answered / forced_answer / no_answer | 保持 max_turns 为硬上限；不训练时规则只能写进上下文；拆开报告才能看出模型会不会自己停 |
| 2026-09-27 | 搜索次数分"尝试"和"实际执行"两个数，成本按实际执行算 | Search-R1 强制轮会把未执行的搜索计入统计 |
| 2026-09-27 | 检索工具接口只收 `query, top_k`；Mock 语料用虚构实体 | 防泄漏靠接口签名保证；虚构事实让真模型无法凭记忆作答 |
| 2026-09-27 | 重复查询：归一化（NFKC + 小写 + 去标点 + 合并空白）后完全相同才拦；拦下不扣搜索次数、扣轮数；另记 `num_new_docs` 只记录不拦截 | 检索器眼里一样才算重复；语义去重误判代价大；新文档数是 V2 边际收益特征 |
| 2026-09-27 | 强制作答只在"最后一轮"或"搜索次数用完后仍想搜"之后触发 | 让 `forced_answer` 只统计"还想搜但被截停" |
| 2026-09-27 | 检索出错算一次实际执行，但不记入"已搜过" | 请求已经发出、占了检索资源；允许模型原样重试 |
| 2026-09-27 | V1 不在观察里告诉模型剩余搜索次数；V2 作为消融项 / 零成本对照基线 | 会干扰"模型能否自己判断何时停"的测量，有锚定效应，且让预算曲线混入提示词差异；但它本身是预算平滑式策略，适合做对照 |
| 2026-09-27 | 关键测试要做改坏检查 | 第一次全绿不代表测试有效；已发现一个测不出目标问题的测试 |
| 2026-09-27 | 模型服务出错：只重试临时性错误（指数退避，最多 3 次）；仍失败记 `stop_reason=error`、计 0 分、留在分母里；另报去掉失败题的诊断指标；失败率 > 2% 判运行无效 | 跳过失败题会把难题移出分母，准确率虚高且方法间不可比 |
| 2026-09-27 | runner 在工作区有未提交改动时拒绝运行；`--allow-dirty` 调试运行会存 `git_diff.patch`、目录名带 `-dirty` | commit hash 要能代表实际代码；靠代码把关而不是靠人记 |
| 2026-09-27 | `errors.csv` 收录所有没答对的题，分 model_error / no_answer / wrong_answer；`trajectories.jsonl` 每行 = 轨迹 + `eval`（gold、em） | 错题分析的入口；gold 只在评测后写入输出文件，不进 prompt |
| 2026-09-27 | 数据处理目录沿用 `data_prep/`，不按 plan 改名为 `datasets/` | 仓库根目录下的 `datasets/` 会遮住 HuggingFace `datasets` 库的导入 |
| 2026-09-27 | 新增 `docs/PROJECT_OVERVIEW.md`（项目全景），每个阶段更新 | 用户要求宏观优先，便于面试准备 |
| 2026-09-27 | 旧 `hotpotqa_corpus.jsonl` 不复用，从原始数据重建 | 旧语料只收了 4500 道被选中题目的上下文（考哪些题决定库里有什么），且没保存金标段落，无法算证据召回 |
| 2026-09-27 | 语料池 = train + 官方 dev 全部题目的上下文段落，按标题去重（1821 个标题有多版本，相似度中位数 0.998，留出现最多的）；`doc_id` = 标题哈希 | 语料由整个数据集决定、不依赖抽到哪些题；同一段落的多个版本会重复占用 top_k 名额；标题唯一所以 ID 可复现 |
| 2026-09-27 | test = 官方 dev 抽 500；validation 200 / debug 50 从 train 的 hard 题抽 | 官方 dev 全是 hard 题，调参集分布要和测试集一致；官方 test 没有公开答案 |
| 2026-09-27 | 每个划分拆成 `questions/`（只有 qid + question）和 `labels/`（答案、类型、金标段落）两个文件 | 防泄漏落到文件结构上：Agent 只读题目文件，评测器才读答案文件；`validate_splits` 检查题目文件没有多余字段 |
| 2026-09-27 | 不做 NQ 调试集 | NQ 是单跳维基问答，HotpotQA 语料池里大多没有它的答案；调试改用 HotpotQA debug 集。偏离 plan §4.1 |
| 2026-09-27 | BM25 用 bm25s（默认 k1=1.5, b=0.75）+ 英文停用词 + 词干化；标题和正文一起建索引；同分按 doc_id 排序 | 标题就是实体名，是最强信号；词干化与 Pyserini 默认一致；同分排序保证结果可复现 |
| 2026-09-27 | 检索做成独立服务，正式实验走 HTTP；客户端超时 5s，异常交给循环记 tool_error | 索引只加载一次、多实验共用；之后换向量检索不影响 Agent；记录真实接口耗时 |
| 2026-09-27 | 新增证据召回指标：`evidence_recall`（平均覆盖金标比例）、`all_evidence_found`（金标全部找齐的比例） | 把"搜得好不好"和"答得好不好"分开，是 V2 比较路由策略的核心指标 |
| 2026-09-28 | vLLM 启动加 `--guided-decoding-backend lm-format-enforcer`，不往 `verl_env` 装 `pyairports` | vLLM 0.6.3 对每个 chat 请求都构造（空的）约束解码参数，默认后端 outlines 导入时缺 `pyairports` → 全部 500；lm-format-enforcer 已装好，遇到空参数直接返回"不约束"，行为和不约束一致 |
| 2026-09-28 | 模型服务走 chat 接口（服务端套对话模板），不在客户端自己拼模板 | 模板以模型目录里的 `tokenizer_config.json` 为准，只有一份；`usage` 直接给出输入输出 token 数 |
| 2026-09-28 | 系统提示词 = 任务说明 + Qwen 原生工具说明（`# Tools` / `<tools>` 函数签名 / `<tool_call>` 格式，和对话模板传 `tools` 时逐字一致）；文本放在我们的代码里，不靠服务端的 `tools` 参数生成 | 占位提示词 0/8 → 原生写法 8/8；提示词进 git、进轨迹，不随 vLLM 版本变 |
| 2026-09-28 | 生成时 `stop=["</tool_call>"]`，保留停止词本身 | 原生模板允许并行调用，提示词说"每轮一个"压不住；模型会在看到结果前就写 final_answer，这段文字留在上下文里会误导后续轮次。规则靠代码强制（同 Search-R1 在 `</search>` 截断） |
| 2026-09-28 | 每次运行正式计时前先发一个预热请求 | 首个请求 6s（CUDA graph 等初始化），不预热会拉高 P95 延迟 |
| 2026-09-28 | 原生工具说明在客户端用代码生成，不在请求里传 `tools` 让服务端套模板；测试里拿对话模板逐字比对 | 模型看到的每个字都在代码里、进 git 和 `config.yaml`；Direct 可以复用同一写法只给作答工具；换模型时测试会报不一致 |
| 2026-09-28 | 解析器唯一的放宽：缺 `</tool_call>`，但最后一个 `<tool_call>` 之后直到结尾恰好是完整 JSON 对象 → 照常解析，`Step.unclosed_tool_call=True`；后面还有文字、JSON 不完整（截断）都仍然报错 | 实测格式错误多数是这种，意图无歧义；判成"没作答"会把格式失误记成停止失败。放宽范围窄、有标记、所有方法共用，不影响"纯文字不当答案"的原则 |
| 2026-09-28 | 模型客户端关掉 openai 库自带的重试（`max_retries=0`），值为 None 的参数不发 | 库自带 2 次重试会和 `RetryingLLM` 叠加，重试次数说不清；库会把 None 发成 null（测试抓到的） |
| 2026-09-28 | 服务端软件版本用 `verl_env` 的解释器读包元数据，不用 vLLM 的 `/version` | vLLM 0.6.3 缺版本文件，`/version` 只返回 "dev" |
| 2026-09-28 | 假模型的 token 数记 None，不记 0 | 0 会被统计成"零成本" |
| 2026-09-28 | 观察的 token 数不单独用 tokenizer 算，离线用相邻两轮的 `prompt_tokens` 差减去上一轮 `completion_tokens` 得到（含模板包装的几个 token） | 服务端的 `usage` 就是真实计费口径；少一个 tokenizer 依赖，也不会和服务端的计数方式不一致 |
| 2026-09-28 | 连续 5 题模型服务失败就中止整次运行，输出目录标为不完整 | 服务挂了时继续跑只会把剩下的题全记成 error，浪费时间且结果无效 |
| 2026-09-28 | Direct / Static RAG / Oracle 共用一份"只能作答"的提示词（原生写法，只给 final_answer 工具），走同一个循环和解析器；证据放在用户消息里问题前面 | 三者只差"证据"一个变量；Direct 若沿用 Agent 提示词，模型想搜会浪费轮数并被记成 forced_answer |
| 2026-09-28 | Static RAG 的那次检索计入 `search_calls_used`（=1），`max_search_calls` 配成 1；Oracle 不计检索成本 | 成本和 Agent 放在同一把尺子上；配置和方法对不上时 `check_budget` 拒绝运行 |
| 2026-09-28 | 只能作答的方法也用 `max_turns=5` | 多出来的轮次只在格式出错时用来重试，和 Agent 的纠错机会一致 |
| 2026-09-28 | Oracle 段落按 doc_id 排序，不按标注顺序 | 标注顺序常是第一跳在前，会把推理路径透露给模型 |
| 2026-09-28 | Token-F1 照搬 HotpotQA 官方脚本（yes/no/noanswer 只认完全一致，不做词干化） | 和论文可比；"Latvia" vs "Latvian" 仍记 0 分是官方口径 |
| 2026-09-28 | 证据召回把答题前给的证据也算进去 | Static RAG 的证据来自流程检索，不算就会是 0，和 Agent 不可比 |
| 2026-09-28 | 配置支持 `extends` 继承；基础配置没有 `name`，不能直接运行 | 对比实验的解码参数、数据、预算必须完全一致，靠继承保证，不靠人抄对 |
| 2026-09-25 | 进度靠 `CLAUDE.md` + `docs/PROGRESS.md` + `docs/LEARNING_NOTES.md` 保存，并定期 push 到 GitHub | 对话记录会被压缩或清理；仓库在数据盘上，实例释放即丢失 |

## 已有资产（上一次 Search-R1 复现留下，位于系统盘 `/root/Search-R1/`）

- `data/nq_hotpotqa/{train,test}.parquet`（632M）：Search-R1 官方处理后的 NQ + HotpotQA 数据。
- `data/bm25_index/`（2.2G）：wiki-18 的 BM25（Lucene）索引；对应语料 `wiki-18.jsonl` 原先放在数据盘，已丢失。
- `eval/data/corpus/hotpotqa_corpus.jsonl`：41,897 个段落；`eval/data/eval/hotpotqa_dev.jsonl`：500 道题。
- `eval/src/`：上次写的 base / rag / agent 评测和 GRPO 代码。

## 待办 / 开放问题

- ~~观察结果用 `role="tool"` 拼回，要核对 Qwen2.5 对话模板的实际渲染~~ → 2026-09-28 已核对：包进 `<tool_response>`，方式正确。
- 3.5 看轨迹时专门看查询改写：每个 Agent 查询的金标排名 vs 原问题的金标排名，统计改写是得是失（例：`Mary Gordon birth year` 让金标 1 → 2，`H. L. Mencken birth year` 让 5 → 10）。
- 用了停止词后 `num_tool_calls` 永远 ≤ 1，"模型想并行调用"的信息丢了。需要时可在 debug 集上不设停止词单独统计。
- ~~服务整体挂掉时应提前中止~~ → 3.3 已加（连续 5 题）。
- ~~`config.yaml` 补模型名、解码参数、版本；真实客户端异常映射到可重试类型~~ → 3.3 已完成。
- 3.5 修：格式报错提示（`agent/parser.py` 的 `_FORMAT_HINT`）固定举例 search，但只能作答的方法没有 search 工具；Agent 里模型已经知道答案时也会被它带去重搜（`5ae3fd6d` 第 3→4 轮）。提示应按当前可用工具生成。Static RAG 唯一的格式错误题（`5abc3fe7`）第 2～5 轮一直写 `Call final_answer {...}` 不带标签。
- 3.5 看：比较题 Agent（0.43）不如 Static RAG（0.50）。原问题同时含两个实体，一次检索就能覆盖；Agent 的改写可能反而变差（同 Mary Gordon 的例子）→ 这是"按题型自适应"的直接证据，validation 上确认。
- 3.5 / validation 上调答案规范：Oracle 下比较题仍错的 7 题里，多数是答案形式问题（问"谁"却答了年份、答 "true" 而不是 "yes"、把标题 "Firehose (band)" 原样抄下来），真正推理错约 2～3 题。可以在提示词里加答案格式说明，只在 validation 上调。
- 错因拆分（`20260928-212118`，debug 50 题）：答错 36 题中 26 题证据没找齐、10 题证据齐了仍答错；这 10 题约 5 题是 EM 口径（意思对）、1 题标签问题、约 4 题真读错 → 当前瓶颈主要在检索。
- 候选诊断：**Oracle context**（直接给金标段落，衡量纯阅读能力上限）。它要把 labels 里的金标段落放进 prompt，和"标签只给评测器"的防泄漏规则冲突 → 只能作为明确标注的诊断上限、只在 validation/debug 上跑、不作为方法参与比较；**实现前先和用户确认这个例外**。
- 候选消融：模型尺寸（Qwen2.5-3B vs 7B，7B bf16 权重约 15GB，4090 能放下；14B 需要量化）。放在 Oracle 设定下比较才能测纯阅读能力；建议 Day 5～6 基线表定下来后再做。
- 答案常写成句子或带多余修饰（"Atlanta" vs "Atlanta, Georgia"、答比较题时写年份），EM 偏严 → 3.4 补 Token-F1。
- Direct 基线怎么配：`max_search_calls=0` 时提示词仍说可以搜，模型想搜会浪费一轮并被记成 forced_answer。Day 3 决定是用 `max_turns=1`，还是给 Direct 单独一份不带工具的提示词。
- ~~轨迹还没记录观察的 token 数~~ → 每轮记了服务端的 `prompt_tokens`，观察 token 由相邻两轮差值得到（见决策记录）。
- `/root/Search-R1`（`verl_env` 可编辑安装）与子模块 `third_party/Search-R1` 的关系，到 V3 再决定。
