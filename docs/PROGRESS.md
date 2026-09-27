# 项目进度日志

> 每次会话开始先读本文件；每完成一个小步骤就更新。

## 当前位置

**Day 1 ｜ 步骤 1.5 已完成（33 个测试通过，改坏检查通过），等待用户回答 2 道 1.6 思考题（未提交改动时 commit hash 还算不算数；模型调用出错怎么记）**

下一步：点评思考题 → 1.6 写 runner：对 mock 题跑一遍，产出 `outputs/runs/<run_id>/` 下的 `config.yaml`、`git_commit.txt`、`metrics.json`、`trajectories.jsonl`、`errors.csv`（记录 seed、模型、请求参数）→ Day 1 验收。

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
- [ ] 1.6 产出第一条 JSONL 轨迹（记录 seed、模型、请求参数）

## 从 `infer.py` 发现的隐患（1.3～1.5 要逐条覆盖）

1. `while True` 无轮数上限（`cnt` 只计数）→ Budget 的 max_turns。
2. `get_query` 在包括 prompt 的整段文本上取最后一个 `<search>`，截断时会静默拿旧查询或示例词 "query" 去搜 → 只解析本轮新生成内容，解析失败返回明确错误。
3. `requests.post` 无超时、无异常处理 → 统一异常返回格式。
4. 空查询照搜 → 空查询 / 空结果测试。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-09-25 | 不用 wiki-18 全量语料，采用可控 HotpotQA 语料池 | wiki-18 约 2100 万段落，e5 Flat 向量索引约 64GB，超过 50G 数据盘；与 plan §0.2 一致 |
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
| 2026-09-25 | 进度靠 `CLAUDE.md` + `docs/PROGRESS.md` + `docs/LEARNING_NOTES.md` 保存，并定期 push 到 GitHub | 对话记录会被压缩或清理；仓库在数据盘上，实例释放即丢失 |

## 已有资产（上一次 Search-R1 复现留下，位于系统盘 `/root/Search-R1/`）

- `data/nq_hotpotqa/{train,test}.parquet`（632M）：Search-R1 官方处理后的 NQ + HotpotQA 数据。
- `data/bm25_index/`（2.2G）：wiki-18 的 BM25（Lucene）索引；对应语料 `wiki-18.jsonl` 原先放在数据盘，已丢失。
- `eval/data/corpus/hotpotqa_corpus.jsonl`：41,897 个段落；`eval/data/eval/hotpotqa_dev.jsonl`：500 道题。
- `eval/src/`：上次写的 base / rag / agent 评测和 GRPO 代码。

## 待办 / 开放问题

- Qwen2.5-3B-Instruct 需要重新下载（原先在数据盘上，已丢失），最晚 Day 3 前完成。
- 旧的 `hotpotqa_corpus.jsonl` 是怎么构造的？复用前要核查是否只含 gold 段落（防泄漏，plan §4.1）。Day 2 处理。
- 观察结果用 `role="tool"` 拼回，Day 3 接真模型时要核对 Qwen2.5 对话模板的实际渲染（是否包进 `<tool_response>`）。
- 模型调用出错（`llm.generate` 抛异常）目前会直接中断循环，1.6 的 runner 要按题捕获并写进 `errors.csv`，处理方式看 1.6 思考题 2 的结论。
- Direct 基线怎么配：`max_search_calls=0` 时提示词仍说可以搜，模型想搜会浪费一轮并被记成 forced_answer。Day 3 决定是用 `max_turns=1`，还是给 Direct 单独一份不带工具的提示词。
- 轨迹还没记录观察的 token 数（plan 要求），需要真 tokenizer，Day 3 补。
- `/root/Search-R1`（`verl_env` 可编辑安装）与子模块 `third_party/Search-R1` 的关系，到 V3 再决定。
