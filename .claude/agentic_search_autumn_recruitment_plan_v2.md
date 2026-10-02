# Agentic Search 秋招项目执行计划｜2026 年 9 月修订版

> **项目名称（未做 RL 时）**：Adaptive Agentic Search with Cost-aware Retrieval Policy  
> **项目名称（实际完成 RL 后）**：Adaptive Agentic Search with Retrieval-Aware Reinforcement Learning  
> **周期**：约 21 个有效开发日；Day 7、Day 14、Day 17 设置明确的终止/转向点。  
> **主线**：Search-R1 的多轮搜索框架 → 证据驱动 Query Rewrite → 自适应召回与重排策略 → 固定强基线下的质量—成本评估 → 商品搜索迁移 → 资源允许时做 GRPO。  
> **执行优先级**：完整可信的 V2 实验结果 > Agent 功能数量；第三周的 GRPO 是条件分支，不是必达项。

---

# 0. 阅读指南：这次到底改了什么？

原计划已经具备 Search-R1、BM25/Dense/Hybrid、Query Rewrite、Reranker、GRPO 和多跳 QA 的基础路线。但它可能出现两个问题：一是功能不断累积，却缺少足够强的固定 Search Funnel 对照；二是只在知识问答上验证，难以向一般搜推面试官说明为什么这些能力能用于商品/内容场景。

本次修订不推翻 Search-R1，而是把核心问题收敛为：

> **Agent 能否根据问题、现有证据和预算，动态决定 Query、Retriever、是否 Rerank 和何时停止，并且在接近固定强基线质量的情况下减少实际工具成本？这套策略结构能否在商品搜索中复用？**

| 维度 | 原计划 | 本次修订 |
|---|---|---|
| 第一周 | 多轮检索 Demo + 基线 | **冻结可复现 QA 基线**：同数据/同语料/同预算，日志齐全 |
| 第二周 | 增加 Rewrite、Hybrid、Rerank 等工具 | 聚焦**动态路由和停止**，比较 `Always Hybrid + Rerank`、规则路由和 LLM 路由 |
| 场景 | NQ / HotpotQA / 2Wiki | HotpotQA 为主，NQ 调试，2Wiki 可选；增加 **Amazon ESCI 商品相关性** 小实验 |
| 第三周 | 默认尝试 GRPO | 商品迁移优先，GRPO 以 Stop Point 决定是否启动 |
| 评价 | EM/F1、调用次数 | 质量、证据召回、真实时延、tokens、各工具成本、路由分布、错误归因 |
| 推荐/广告 | 假设搜索技术自动泛化 | 区分**接口复用、策略结构复用、模型效果迁移**，不宣称商品搜索=个性化推荐 |

## 0.1 三个可交付版本

- **V1｜Day 7**：Direct LLM、Static RAG、Vanilla Agentic Search 都能运行；有固定数据拆分、完整轨迹与第一张实验表，能作为基础面试项目。
- **V2｜Day 14～17（真正的项目主体）**：具有可比较的 BM25/Dense/Hybrid + Reranker、Evidence-conditioned Rewrite、自适应 Router、Cost-aware Stop；对比强固定 Funnel 并完成至少一组商品搜索迁移结果。
- **V3｜Day 21（可选）**：完成 GRPO 策略训练和 Reward 消融，或者用同样时间完善商品/跨数据集实验。未做出训练前后对照，绝不在简历上声称 RL 提升。

## 0.2 明确不做的事情

1. 不做只有多个 Prompt 互相调用的 Multi-Agent；先采用 **Single Agent + Typed Tools**。
2. 不在第一周做 RL，不在第二周同时训练新 Embedding、Reranker 和 Agent。
3. 不从完整 Wikipedia GPU Flat Index 起步；初期使用 CPU BM25/CPU ANN 与可控语料。
4. Search-o1 用于理解推理范式，DeepResearcher 用作相关工作，不同时复现三套代码。
5. 不在三周内实现工业级推荐/广告系统。个性化推荐所需用户行为、协同关系和目标函数，广告所需 pCTR/pCVR、预算/竞价，都不是 ESCI 商品数据能证明的内容。

---

# 1. 岗位适配：项目到底要证明什么

前期搜集的腾讯、阿里、字节招聘 JD 表明，不同“Agent”岗位强调的能力并不相同。下面是本项目据此制定的**技术能力覆盖计划**，而非对这些公司全部团队的统一要求。

| 岗位问题 | 本项目的直接证据 | 暂不覆盖的部分 |
|---|---|---|
| AI Search / Agentic Search | 多轮推理、Tool Use、Evidence-conditioned Rewrite、受约束停止 | 真实互联网长周期 Deep Research |
| 搜索召排 | Sparse/Dense/Hybrid、Neural Rerank、效果—延迟权衡 | 亿级线上检索、实时 A/B |
| LLM 搜广推 | 策略提示词、动态工具选择、可选 GRPO | 只有跑通模型不等于后训练能力 |
| 推荐 Agent | 可复用 Tool Gateway、成本优化框架；如有独立数据实验可加兴趣状态 | 长期用户画像、行为序列、协同召回、推荐目标 |
| 广告算法 | 成本约束、多目标 Reward 的思路 | 广告预估、竞价、预算、商业效果验证 |

**与已有经历的分工**：已有工业 Query Rewrite/SFT/偏好学习经历负责说明模型后训练与业务数据经验，既有多目标排序项目负责回答 CTR/CTCVR、跷跷板和多目标建模；新项目重点补上 Agent、RAG、动态召回及可以量化的策略选择。不要为了增加“广推覆盖率”而在新项目中重做全部经典算法。

## 1.1 四个研究问题

- **RQ1**：多轮 Agent 相对 Static RAG 的效果来自哪里？是更多搜索、改写 Query，还是证据驱动的下一跳？
- **RQ2（主问题）**：动态选择 BM25/Dense/Hybrid、可选 Rerank 和 Stop，能否相较 `Always Hybrid + Rerank` 和规则 Router，在质量相近时降低成本，或在预算固定时提高质量？
- **RQ3**：这套策略的**软件接口/状态机**能否在商品相关性任务复用？是否能观察到真正的策略效果迁移？这两种结论分开报告。
- **RQ4（可选）**：GRPO 能否优化 Search Policy，特别是无意义 Search、Rerank 决策和停止行为？成本奖励会不会导致从不搜索？

---

# 2. 系统设计：Single Agent + Adaptive Search Funnel

```text
Question / Commerce Query
         |
         v
 State Builder + Budget Manager
 [request, query_history, evidence, action_history, remaining_budget, domain]
         |
         v
 Reasoning LLM / Policy
 [answer? rewrite? decompose? search with strategy? rerank? stop?]
         |
         v
 Action Validator (schema, allowed tools, token/turn budget)
         |
         v
 Search Gateway: search(query, method, top_k, rerank)
      /         |          \
     BM25     Dense       Hybrid (RRF)
       \        |          /
           Candidate Pool
                 |
        Optional Reranker
                 |
       Evidence / Product List
                 |
   Observation Formatter + Cost Logger ----> back to State Builder
                 |
      Final Answer / Product Ranking
                 |
     Evaluation + Trajectory Analyzer
```

核心设计是把**策略**与**检索工具实现**隔离：QA 与商品搜索共享 `ActionSchema`、`BudgetManager`、`SearchGateway`、日志/消融框架；但允许 Prompt、语料、Embedding 参数和指标不同。**工程复用不等于学得的模型参数零样本泛化。**

## 2.1 状态、动作和规则

时刻 \(t\) 的状态定义为：

\[
s_t=(x,\;q_{0:t},\;E_{0:t},\;A_{0:t-1},\;B_t,\;d)
\]

其中 `x` 为原请求，`q` 为 Query 历史，`E` 为已取回证据，`A` 为历史动作，`B` 为剩余预算，`d ∈ {qa, commerce}` 表示任务域。保留这些轨迹即为第一版 Memory；暂不实现复杂长期记忆。

动作控制在：`ANSWER(answer)`、`REWRITE(query)`、`DECOMPOSE(subqueries)`（非 P0）、`SEARCH(query, retriever, top_k)`、`RERANK(candidate_ids, top_k)`、`STOP(reason)`。执行层可把 Search 与可选 Rerank 打包到一次 Tool Gateway 请求，但日志与 Reward 中要分开计算两类成本。

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class SearchAction:
    query: str
    retriever: Literal["bm25", "dense", "hybrid"]
    top_k: int = 5             # 给 Agent 的观察窗口
    pool_size: int = 20        # Rerank 前候选数量
    rerank: bool = False

@dataclass
class SearchResult:
    item_id: str               # doc_id 或 product_id
    text: str
    score: float
    rank: int
    retriever: str
```

**协议要求**：

- 搜索结果必须带可稳定追踪的 `doc_id/product_id`、索引版本、rank、score、原始 query。
- 非法 JSON/未知 Retriever 返回 Schema Error；允许一次确定性修复，但计入格式错误统计。
- 重复 Query 先提示 Agent，无法解释的重复调用可拒绝执行；超预算返回明确 Observation，并让 Agent 回答或 Stop。
- Agent 的答案不能通过系统暗中注入测试集 gold answer；所有策略只能观察同一索引允许返回的内容。

## 2.2 起步配置（实验超参，非既有结论）

| 变量 | 起步值 | 原因 |
|---|---:|---|
| Max Search Turns | 3 | 第一版避免长轨迹失控，后测 1/2/3/5 |
| BM25/Dense 候选数 | 每种 20 | 让后续 Reranker 有选择空间 |
| Observation TopK | 5 | 限制 Context Explosion |
| Reranker 候选池 | 20 | 输出 Top5 给 Agent |
| 默认运行方式 | 单并发串行 | 保证早期延迟测量可比较 |
| API 异常 | 超时、至多一次重试、保留失败样本 | 不允许只统计成功样本 |

所有参数必须进入配置文件，固定后每个实验保存实际使用版本。若硬件限制较大，减小 corpus 与候选池，而不是把不同方法改成不同预算后仍直接对比。

## 2.3 推荐工程目录

```text
adaptive-agentic-search/
├── configs/{qa_v1,commerce_v1,budgets,policies}.yaml
├── agent/
│   ├── {state,policy,action_schema,loop,stop_rules,prompts}.py
├── retrieval/
│   ├── {schema,bm25,dense,hybrid,reranker,gateway,build_index}.py
├── datasets/
│   ├── {prepare_nq,prepare_hotpot,prepare_esci,validate_splits}.py
├── evaluation/
│   ├── {qa_metrics,ecommerce_metrics,cost_metrics,bootstrap,run_eval}.py
├── experiments/
│   ├── {run_baselines,run_ablation,analyze_trajectories,plot_pareto}.*
├── rl/                       # V3 才创建
│   ├── reward.py
│   ├── tool_adapter.py
│   └── configs/
├── tests/
│   ├── {test_action_schema,test_retriever,test_budget,test_no_leakage,test_eval}.py
├── outputs/
│   ├── runs/<run_id>/{config.yaml,metrics.json,trajectories.jsonl,errors.csv}
│   └── figures/
└── third_party/Search-R1/    # 记录 upstream commit，尽量不要大改原库
```

使用自己的 Agent/Tool Adapter 与上游 Search-R1 对接：第一周推理版本可以轻量实现，第 3 周再移植到锁定版本的 veRL Rollout。大语料、索引、Checkpoint 和密钥不提交 Git；每次评估必须保留模型版本、数据 Manifest、Seed、Git commit。

---

# 3. 环境与 GPU：前两周只为可靠推理付费

**首租建议**：1 × RTX 4090/4090D 24GB、约 16 vCPU、64GB RAM、200GB 持久化存储；实际只够起步子集，完整 Wiki 语料/多套索引需另外估算。BM25 与 CPU FAISS ANN 放在 CPU，Agent/Reranker 串行用 GPU；不推荐一开始用全库 GPU Flat Dense Index。

**镜像**：现成 PyTorch + CUDA + Ubuntu 镜像，项目依赖装进独立 Conda 环境。若沿用 Search-R1 原版训练，先核查其 README 给出的 Python 3.9、PyTorch 2.4.0、CUDA 12.1、vLLM 0.6.3 组合；独立新版 veRL/SGLang 使用其匹配的新依赖，**不可不经验证混装**。

```bash
conda create -n search-agent python=3.10 -y
conda create -n retriever python=3.10 -y
# 原版 Search-R1 RL 确实要复现时再创建：
conda create -n searchr1-legacy python=3.9 -y
nvidia-smi
```

**安装与自检顺序**：CPU Retriever → 一次 Agent 推理 → Agent/Tool 交互 → 完整 Baseline → V3 才配 RL。如果 FlashAttention/vLLM 版本冲突影响进度，可先用兼容的 Transformers 推理走通完整 V1，别让推理引擎兼容问题拖垮第一周。每个环境记录 `pip freeze` 和 PyTorch/CUDA/Driver 版本。

**第三周**：1 × 4090 不承诺能跑完整 3B 多轮全参 GRPO。极小模型/LoRA 可以试探，但是否成功取决于框架、轨迹长度、KV Cache 与并发；3B 多轮训练可在小样本 Smoke Test 后考虑多张 80GB GPU。原版 Search-R1 的 `train_grpo.sh` 示例默认 8 GPU、大 Batch，不能直接复制到单卡。

---

# 4. 数据与防泄漏协议

## 4.1 QA 数据：NQ 调试，HotpotQA 主评测

| 数据 | 作用 | 初始规模（建议） | 风险 |
|---|---|---:|---|
| NQ | 单跳调试和 Search-R1 Pipeline 冒烟 | 100～300 | 调试集不能当最终测试集 |
| HotpotQA | 主正式多跳 QA 与证据评价 | 200 验证，最终 500～1,000 固定样本 | `distractor` 和 `fullwiki` 不是同一检索设定 |
| 2WikiMultihopQA | 跨 QA 数据集泛化 | 可选 300～500 | 不得挤掉主实验时间 |

**严谨性重点**：HotpotQA distractor 给了特定候选上下文，适合早期调试但不等同开放式全库检索；若宣称开放式检索，所有方法必须在同一明确的独立语料池搜索，不得只给 Agent 注入每题的 gold supporting paragraphs。HotpotQA 官方 fullwiki dev 随包的检索候选不等于完整 Wikipedia 索引。代码中固定 `retrieval_setting`、`corpus_version`、`sample_ids` 并写在每张实验表标题下。

训练/调参只看 train 和独立 validation；正式固定样本最后运行。公开 test 若无参考答案则不编造 EM/F1。统一答案归一化、Yes/No 规则和 gold evidence 匹配规则。

## 4.2 商品迁移：Amazon ESCI（商品**搜索**，不是推荐）

官方数据含 `query_id, query, product_id, product_locale, esci_label, split`，另有商品标题、描述、品牌等属性。第一版只选 `product_locale=us` 的英文 `small_version` 子集，用官方 train/test 标记并从 train 再切 validation；必须保存筛选与采样 Seed。

有两个互不混淆的实验：

- **Setting A／主评测：固定已标注候选商品集上的排序。** 每个 Query 的官方已判定 Query–Product 对形成同一候选池。比较 BM25、Dense、Hybrid、Hybrid+Rerank、规则 Router、Adaptive Policy 在**完全相同候选商品**上的排序及工具成本。可以说明重排和策略选择，但不能称全库召回。
- **Setting B／扩展：真实商品池检索。** 给所有方法建立同一商品语料池，报告已标注正例覆盖率、候选命中、成本；ESCI 标注不完整，**未标注商品不得直接当负例**，不能把正例覆盖率当全库真 Recall。

`E/S/C/I` 分别表示 Exact / Substitute / Complement / Irrelevant。使用 nDCG 前**预先声明**增益映射，例如 `E=3,S=2,C=1,I=0` 是本项目人为设定而非官方业务效用；需对 Complement 调低/不计入做敏感性分析。商品文本可作为索引语料，但 **test 的 `esci_label` 只由 evaluator 读取**，绝不注入模型 Prompt/检索工具。不能从这个数据推导 CTR、CVR、GMV 或用户长期兴趣提升。

**可选推荐小实验**：如第三周有余力并且招聘方向转向推荐，单独用 MovieLens 这类有 `user_id/item_id/timestamp` 的数据，按用户时间切分、引入用户行为状态，对照 Content-based、ItemCF、Hybrid 召回，评价 Recall@K/NDCG@K。没有这一步，简历只写商品检索**工程复用**，不写“推荐效果泛化”。

---

# 5. 统一评价与强基线

## 5.1 质量、成本与轨迹

| QA 质量 | 商品质量 | 成本/系统 | 决策分析 |
|---|---|---|---|
| Answer EM、Token-F1 | Setting A 主报 nDCG@10 | Search/Rerank/Rewrite 调用数 | BM25/Dense/Hybrid 选择分布 |
| Evidence Recall@5/10 | MRR@10、标签敏感性 | Total Input/Output Tokens | Rewrite Success Rate |
| Supporting-Fact F1（可得时） | Setting B 已标注正例覆盖率 | P50/P95 端到端延迟 | 重复查询、早停和超预算率 |
| Search Success Rate | 分 Query 类型分析 | 每种工具耗时、Cost Proxy | 格式错误、证据缺失、答案推理错误 |

成本代理可定义为：

\[
 C=w_bN_{bm25}+w_dN_{dense}+w_hN_{hybrid}+w_rN_{rerank}+w_tN_{tokens}.
\]

`w` 是实验权重，不是云平台真实账单；Hybrid 内部如果已经包含 BM25+Dense，不得双重计费。**一定同时提供真实平均调用次数和观测时延**，不能只靠任意 Cost Proxy 给出结论。

评估必须固定模型版本、温度、Seed、数据 ID、索引、最大生成长度、预算、机器与并发。既做固定预算比较质量，也做相近质量比较延迟/成本，绘制 **Quality–Cost Pareto**。对主要差值按 Query-level bootstrap（如 1,000 次）估计 95% CI；对于样本不足的结论只写趋势，不夸大显著性。超时、格式错误和 OOM 样本必须计入端到端结果。

每个 `outputs/runs/<run_id>` 至少保存：`config.yaml`、`git_commit.txt`、`metrics.json`、`trajectories.jsonl`、`errors.csv`；一条轨迹需有每轮 Action、Query、检索 IDs、Rerank 决策、Observation Token、Latency、当前 Budget、Stop Reason。

## 5.2 不可缺的 Baseline

| ID | 方案 | 要证明的区别 | 优先级 |
|---|---|---|---|
| B0 | Direct LLM，不检索 | 测模型已有知识 | P0 |
| B1 | Static RAG + BM25，一次检索 | 单轮检索的价值 | P0 |
| B2 | Vanilla Multi-turn Agent + 固定 BM25 | 多轮 Agent 的额外价值 | P0 |
| B3 | **Always Hybrid + Rerank** | 强固定 Search Funnel，不让 Ours 只打败弱基线 | P0 |
| B4 | Rule Router | LLM Router 相比廉价启发式是否必要 | P0 |
| B5 | Static Rewrite + 固定检索 | 一次性改写增益 | P1 |
| B6 | Evidence-conditioned Rewrite + 固定检索 | 证据反馈的贡献 | P1 |
| Ours | Adaptive Query + Retriever + Rerank + Stop | 主方案 | P0 |

B4 的初始启发式可以是短实体/型号词 → BM25、较长语义表达 → Dense、混合情况 → Hybrid，但它们**只是待验证的规则**，阈值要在 validation 上选定。B2 固定 Retriever，才能把多轮收益与动态工具选择收益分开。B3 保留与 Ours 同等或更强的候选池，再通过**预算对齐**和**质量对齐**分别报告结果，不能刻意把强基线做弱。

主消融：`−Evidence-conditioned Rewrite`、`−Adaptive Retriever`、`Always Rerank`、`Never Rerank`、`Fixed Turns`。还要报告路由分布，检查 Ours 是否退化成“永远 Hybrid + Rerank”。

---
# 第一周｜V1：多轮 QA 基础复现与可信评估（Day 1～7）

> **第一周禁止启动 RL。** 先跑通一个可以解释、复现、与 Static RAG 公平比较的系统。以下日程按“一个人独立开发、资源有限”设计；日期是任务次序，不要求每天一定结束全部任务。

## Day 1｜锁定 Search-R1 源码，做最小闭环

**阅读**：Search-R1 官方 README、`infer.py`、检索脚本、`train_grpo.sh`，理解 Search Tool 作为环境交互动作时的数据格式。查看上游 commit 后记录在 `third_party/README.md`。注意官方 GRPO Shell 示例含 8 GPU 等大规模默认值，**第一周只看代码，不直接训练**。

**实现**：创建自己项目的 Git 仓库、Conda 环境；先用 Mock Search Tool 跑一次 `question → Action → Observation → final_answer`。定义 Action Schema、预算对象和异常返回格式。独立测试非法 JSON、未知工具、空结果、超过 Max Turns 的行为。

**当天验收**：`python -m tests.test_action_schema` 等冒烟测试可过；至少一条 JSONL 轨迹，记录 Seed、模型和请求参数；`nvidia-smi` / PyTorch CUDA 自检通过。

## Day 2｜BM25 检索服务与语料 ID

1. 确定初期 QA 检索设置。如果只有可控语料池，要在结果中明确标注 `corpus=qa_subset`，不可悄悄宣称 fullwiki；为每个 passage 生成稳定 ID 和标题/正文。
2. 建 BM25 CPU 索引，封装 `search(query, top_k)`，用 FastAPI 提供 `/health` 和 `/search`，返回 `id,text,score,rank,source`。固定相同 Query 的 TopK 顺序。
3. 预处理 100～300 条 NQ 调试题；写 `validate_splits.py` 保证每个 split 的 Question ID 不重叠。生成数据 Manifest。
4. 对空 Query、无命中、相同标题不同段落、重复 doc_id 做测试。

**验收**：10 条输入可批量检索且能定位每个返回段落；日志保留 API 真实耗时，索引版本可追踪。

## Day 3｜Direct 与 Static RAG

1. 选择一个小而稳定的推理模型，如 Qwen2.5-3B-Instruct；对于没有经过工具使用微调的 Base 模型，不假设它能立即稳定输出工具 JSON。
2. 完成 B0 `Question → LLM → Answer` 与 B1 `Question → BM25 → Context → LLM → Answer`；统一答案抽取/标准化、解码参数、上下文长度。
3. 实现 EM、Token-F1、检索次数、输入输出 tokens、P50/P95 latency 统计。先手工审查约 30 条，再批量 100 条 NQ。

**验收**：可以从同一个配置入口运行 B0/B1；`metrics.json`、失败样本与完整 run config 自动保存。

## Day 4｜Vanilla Multi-turn Agent（固定检索器）

1. 实现 `Reason → Search → Observe → Reason → Search/Answer` 循环，先强制使用 BM25，限制最多 3 次检索。
2. 加 `seen_queries`、每轮观察窗口截断、工具超时和预算耗尽兜底。保持原始 Query、改写后 Query 和 Observation 分字段记录。
3. 跑 B2，人工检查至少 10 条真多轮轨迹：是否找到了第二跳所需实体？是否在证据足够时停止？是否反复搜同一个词？

**验收**：真实完成两次以上不同 Query 的检索，超时/格式错误时不会无限循环；检索库与 B1 相同。

## Day 5｜HotpotQA 主基线

1. 选择清楚声明的 HotpotQA 设置并固定约 200 条验证题。若采用 distractor 的给定上下文，结果标题写“固定候选文档设置”；若搭独立 corpus，全方法用同一索引。
2. 对 B0、B1、B2 运行同一题目集合，记录 Answer EM/F1、Evidence Recall、Avg Search Turns、Tokens 和延迟。
3. 整理至少 20 个 Badcases，分为“检索未命中／缺少第二跳／Query 表达差／证据找到但推理错／无效重复搜索／过早停止／格式非法”等。

**验收**：一张严谨的 Baseline 总表、可从 JSONL 重放的典型样例，每个方法共享相同 Data Manifest 和 Budget 定义。

## Day 6｜回归、修正、第一版讲稿

针对最主要的 2～3 类 Badcase 修补 Pipeline，而不是急着加新工具。完成 Budget、Action Schema、稳定 doc_id、No-leakage 和 Metrics 的单元测试；增加 1～2 次不同 Seed 运行，查看模型解码波动是否大到影响结论。

**产物**：框架图 V1、基线总表、3 条成功案例与 3 条失败案例、代码 Tag `v1-baseline`。写 30 秒介绍：为什么 Static RAG 不够，多轮能解决什么问题，以及它带来了哪些额外成本。

## Day 7｜Stop Point 1：只有达标才进入动态检索

- [ ] B0/B1/B2 都能在同一固定数据、语料和预算下运行。
- [ ] 至少约 200 条样本有完整质量与真实成本日志。
- [ ] 能区分 Retriever Miss 和 Reasoning Error，而不是只展示几个好看的回答。
- [ ] 一键复现实验，有固定版本与模型设置。

若缺少任一关键项，第 8 天先补齐。V1 可以进面试做系统基础说明，但此时不要宣称已有自适应策略创新。

---

# 第二周｜V2：把 Agent 变成可比较的自适应 Search Funnel（Day 8～14）

> 第二周只把 **Evidence-conditioned Rewrite、Adaptive Retriever、Adaptive Rerank/Stop** 做透。Query Decomposition 可作为拓展，但不是所有功能都要成为独立可训练 Agent。

## Day 8｜Dense + Hybrid 的效果与成本

1. 选择一款与测试语言匹配的 Embedding 模型；将相同 QA corpus 编成稠密向量。小规模可用 FAISS Flat CPU，更大规模优先评估 HNSW/IVF CPU；确认 ID 映射不漂移。
2. 将 Dense Retrieval 作为同一 `/search` API 的 `method=dense`，与 BM25 共用 TopK 和索引版本记录。先比较各自在 100～200 条验证题上的 Evidence Recall、Latency 和内存。
3. 实现 RRF Hybrid，避免未经校准直接将不同 Retriever 的原始分数相加：

\[
\mathrm{RRF}(d)=\sum_{r\in\{bm25,dense\}}\frac{1}{k+\mathrm{rank}_r(d)}.
\]

4. 抽查三类 Query：实体/型号词、语义描述、复杂混合问题；先统计差异，再考虑规则 Router。

**验收**：三个 Retriever 可通过同一 API 调用，结果含一致的 IDs、得分、Rank 和工具耗时。

**2026-10-02 实际结果与调整（以这里为准）**

- 结果见 `docs/PROGRESS.md` 的"Day 8 检索部分完成"。要点：Dense（e5-base-v2）在 HotpotQA 上比 BM25 高 14.5 个点召回@3，一次检索就超过 V1 的 BM25 多轮 Agent；等权 RRF Hybrid 在 HotpotQA 上不如 Dense，在 2Wiki 上又显著强于 Dense。
- **第 4 条"考虑规则 Router"降级**：先量了路由上限（`experiments/routing_ceiling.py`），逐题理想路由只比最强的固定一路多 3～5 个点，57%～70% 的题两路打平。真实路由器拿不到上限的一半，落在噪声里 → 选检索器只做一个小消融，不作为 V2 主线卖点。
- **主线转向查询构造**：2Wiki 上整句问题召回 0.59，换成"实体 + 关系"的理想子查询 0.97，三路全落空从 42% 降到 2%。查询构造的影响远大于选哪一路 → Day 10 的改写 / 拆解是重点，Day 12 主实验的消融权重相应调整。
- **新增 2Wiki 分析集**（`data_prep/prepare_2wiki.py`，38.5 万段语料池、按题型分层的 2000 题）：用来测问题拆解和跨数据集稳健性；组合题（compositional）是 DECOMPOSE 的直接评测场。目前只有 analysis 划分，要当正式评测集需另抽 validation / test。

## Day 9｜Reranker 与最关键的强基线

1. 选择现成 Cross-Encoder/Reranker，不从零训练。候选池先取 Top20，Rerank 后输出 Top5 给 Agent。
2. 实现 B3 `Always Hybrid + Rerank`；与 B1/B2 测量相同题目上的质量与额外耗时。对比有无 Reranker 的质量收益以及无 gold evidence 时 Rerank 的无效性。
3. 记录 Reranker 平均调用时间、输入长度和候选数量；确保部署在单卡时与 Agent 推理串行，避免非预期显存竞争。

**验收**：强固定 Funnel 结果完整；新方案将来至少要对比这个基线，而不只和 Vanilla RAG 比。

**2026-10-02 对第 2 条的修正**：Day 8 量出"最强的一路（含要不要混合）依数据集而变"——HotpotQA 上 Dense 最强，2Wiki 上 Hybrid 显著更强。所以 B3 不能写死成 `Always Hybrid + Rerank`：**每个数据集都同时跑 `Always Dense + Rerank` 和 `Always Hybrid + Rerank`，取实测更强者为 B3，并在结果表里注明挑选依据**。强基线不能刻意做弱。

**2026-10-02 Day 9 实际结果（以这里为准）**：
- 重排模型 bge-reranker-base（GPU fp16，20 条候选约 16～22ms，显存约 0.7GB）。离线召回@3：HotpotQA Dense 0.710 → 0.823、Hybrid 0.688 → 0.828，重排后两路差距抹平；2Wiki 组合题受候选池限制（池@20 只有 0.57）。
- **HotpotQA 的 B3 = Static RAG + Dense + 重排（只搜一次）**：EM 0.405（3 个 seed 0.395～0.430），比 V1 Static RAG 高 7.5～9.5，4/4 显著；Hybrid + 重排不比它强（F1 3/4 显著更差）。
- **在这个强检索下，多轮 Agent 不比单次强**：Agent + Dense + 重排 − B3 的 EM 0/4 显著，成本是 3.9 倍输入 token、5 倍延迟。
- **V2 主线调整为"按需升级"（级联）**：默认走 B3，只在信号表明需要时升级为多轮。逐题取较好的上限 +8（稳定口径）～+13（单次口径）。Day 11 的 Policy 接口里，"升级 / 不升级"是核心动作；Day 12 主实验的 Ours 要和 B3 在"相同质量比成本 / 相同成本比质量"两个方向上比。2Wiki 的 B3 待端到端确认。

## Day 10｜Query Reformulation：把已有 Rewrite 经历用在新问题上

做三组公平对照：

- `Original Query`：不改写直接检索。
- `Static Rewrite`：模型首次检索前生成一个改写 Query。
- `Evidence-conditioned Rewrite`：在一次或多次检索后，根据 `original_question + previous_queries + retrieved_evidence` 修改下一跳 Query。

在模型策略里区分何时需要 Rewrite：当前证据缺失、实体指代不清、两跳链路未闭合、上一轮结果语义偏移。对每个 Rewrite 记下新实体是否来自可见 Evidence，避免凭空猜测“正确答案中的实体”。

**评测**：`New Evidence Recall`、最终 Answer F1、Avg Search Turns、Rewrite 调用次数和无关实体产生率；总结 5 个新增有效证据的例子和 5 个改坏的例子。`DECOMPOSE` 仅在不影响主线的前提下加 Prompt 版。

**验收**：能分清“改写增加了召回”与“最终模型碰巧答对”，能在相同 Search Budget 下比较 Static 和 Evidence-conditioned 方案。

**2026-10-02 Day 10 实际结果（以这里为准）**：用**固定检索计划**实现三组对照（不让模型决定搜几次，避免查询、停止、作答缠在一起），检索栈统一 Dense + 重排、作答提示词与 B3 一字不差：
- 静态改写**代替**原问题：EM −8.5 [−13.5, −3.5]、召回 −10.8，显著更差 → 只看问题的一次性改写不如原问题
- 两跳（原问题 → 静态改写）：EM −4.5 显著更差；**两跳（原问题 → 证据条件改写）：召回 +4.5 [+2.5, +6.5] 显著、EM +2.0 不显著**；两段全齐 0.685 → 0.770
- 证据改写 − 静态改写 EM +6.5 [+3.0, +10.5] 显著 → 收益来自"第二跳看到第一跳结果"；证据改写 − 原问题取前 6 条 EM +4.0（不显著）、召回 +2.0 → 不是靠多看段落
- **改写步"拒绝再搜"是很好的升级信号**：56% 的题模型看完第一跳直接给答案，这些题的证据和 B3 逐条相同、EM 与 B3 完全一致（零翻转）→ Day 11～12 的按需升级用它判断"要不要升"
- 上下文格式：证据条件改写的上下文要用 Agent 的对话格式（自己发过的 search 调用 + 工具返回）。放在用户消息里时 50 题 0 次用上证据实体
- **Day 10.4（2Wiki validation 800 题）**：证据改写 − B3 EM +4.0 [+2.1, +5.9] 显著，组合题 +14.0；**只看问题的静态拆解（DECOMPOSE）没用**（EM −3.1 显著更差，组合题桥接实体召回 0.077 → 0.077 不变），因为拆出来的子查询只能用问题里已有的名字。证据改写 − 静态拆解 EM +7.1 显著。B3 单次检索的桥接实体召回只有 0.066（点名实体 0.98）。"拒绝再搜"在 2Wiki 上 385 题零翻转 → 两个数据集都成立，Day 11 的按需升级直接用它

## Day 11｜统一 Policy 接口：固定策略、规则路由、LLM 路由

实现同一 `policy(state)->action` 接口，至少 3 种 Policy：

- **Fixed Policy**：固定 BM25 或 Hybrid；可设置 Always Rerank、Never Rerank、固定检索轮数。
- **Rule Policy**：使用 Query 长度、词面/实体线索、上一轮有效证据、剩余预算决定 retriever/rerank/stop；所有阈值只允许从 validation 得到。
- **LLM Policy**：用当前已观察 State 生成 Typed Action，随后由 Action Validator 检查预算与 JSON Schema，不能拿到 gold label。

当日完成：统一 `BudgetManager`，可以对 `Search Calls`、`Rerank Calls`、Token 上限分别设预算；日志拆分 Policy 推理时延、Retriever 时延和 Reranker 时延。

**验收**：更换 Policy 不修改 Agent Loop、Retriever Gateway 和评估器。验证 Rule Policy 是有竞争力的廉价对照。

## Day 12｜完整自适应策略主实验

运行 B3、B4、Ours，保持**相同测试题、索引、模型生成配置和预算**。额外做：

1. `No Evidence-conditioned Rewrite`：只让 Agent 选择 Retriever/Rerank。
2. `Fixed Hybrid Router`：保留 Rewrite，但不允许动态选 Retriever。
3. `Always Rerank`、`Never Rerank`：验证决定是否重排的贡献。
4. `Fixed Turns`：对比自由 Stop 和固定 3 轮。

主图画 `Answer F1 – Actual Latency` 和 `Answer F1 – Cost Proxy`，尽可能标出 Bootstrapped CI；再给 Retriever 选择分布、平均观察 Tokens、预算耗尽率。特别检查动态策略是否只学会“更多地使用 Hybrid”。

**验收**：可以实证回答“为什么不对所有 Query 都执行 Hybrid + Rerank？”若回答是“在这批样本里强固定策略更好”，也应保存结论并做错误归因，不能编造优势。

## Day 13｜新增商品搜索后端

1. 下载官方 Amazon Shopping Queries / ESCI 字段表，选择 US/small_version 子集，保留官方 split，从 train 划出 validation。
2. 对 `product_id` 的标题/品牌/少量描述建立 BM25/Dense；商品文本之外不把 ESCI 标签输入 Retriever 或 LLM。
3. 实现 `commerce` 的 Domain Adapter：输入商品 Query，返回商品候选和可选相关性排序。**复用** Action Schema、Search Gateway、Budget 与日志，但使用商品域 Prompt 和商品评价器。
4. 优先做 **Setting A 固定已标注候选池**：BM25、Dense、Hybrid、Hybrid+Rerank 的可重复排序；先用约 100 个 Query 测试整条链路。

**验收**：能稳定读取 ESCI、合并商品字段并得到 nDCG@10；各方法用相同固定候选集，知道哪里是固定候选排序、哪里才是全库检索。

## Day 14｜Stop Point 2：冻结 QA V2、保证商品可运行

- [ ] QA 至少有 B0/B1/B2/B3/B4/Ours 的正式对比及主要消融。
- [ ] 工具成本完整，包含 Rewrite/Rerank 的模型 Tokens 和推理耗时。
- [ ] 所有 Prompt、索引与策略阈值在 QA 正式测试前已冻结。
- [ ] 商品数据管线跑通，固定候选排序的四种检索/重排 Baseline 至少已有初步结果。

将 QA V2 Tag 为 `v2-adaptive-qa`。若商品数据没跑通，下一步先处理商品，不直接跳到 GRPO。

---

# 第三周｜商品迁移优先，GRPO 根据实际进度选做（Day 15～21）

## Day 15｜商品主实验：固定候选重排与动态路由

统一同一 ESCI Query、相同候选商品、同一 E/S/C/I 增益映射，运行：

| 实验 | 设置 | 必须报的指标 |
|---|---|---|
| E0 | BM25 | nDCG@10 / MRR@10 / Cost |
| E1 | Dense | 同上 |
| E2 | Hybrid RRF | 同上 |
| E3 | Always Hybrid + Rerank | 同上，强基线 |
| E4 | Rule Router | 同上，廉价路由 |
| E5 | Adaptive LLM Router | 同上，商品域 Prompt |

对商品品牌/型号、功能描述、多约束商品查询分桶分析；只有实验支持时才声称某个 Retriever 擅长某一类 Query。先完成固定候选集 nDCG，再根据资源决定是否扩展全商品池检索。

**验收**：第一张商品搜索质量与成本表，与 QA EM/F1 结果分开呈现。

## Day 16｜区分三种“迁移”，整理跨域 Badcase

明确写出以下三种不同结论：

1. **接口复用**：Tool Gateway、Action Schema、Budget、日志和评估实验框架能不重写核心逻辑地支持商品检索（工程结论）。
2. **结构复用**：Rule/LLM Policy 的状态机能保留，但更换索引与商品域 Prompt 后在商品数据上仍可评测（方法设计结论）。
3. **参数零样本迁移**：如真的测试保留 QA Prompt/模型策略不变迁移到商品域，必须单独报告结果；不能拿“换 Prompt 后有效”冒充零样本 Policy 迁移（模型效果结论）。

人工检查至少 10 条商品 Badcase。若扩展 Setting B，全库召回只报**已标注正例覆盖率**，未标注不是负例。对 Complement 标签降权/归零再做一组 nDCG 敏感性实验。

**验收**：一页迁移边界总结，能对推荐面试官准确解释哪些能力能复用，哪些还需要用户行为/协同信号实验。

## Day 17｜Stop Point 3：是否值得做 RL？

只有同时满足以下条件，才进入 A 分支：

- [ ] QA 主结果与消融已有完整表格、Badcase 和质量—成本图。
- [ ] 商品迁移有固定已标注候选池的可复现结果。
- [ ] 训练/validation/final eval ID 已固定；Reward 不会读取最终测试答案。
- [ ] 现有模型 Tool Action 合法率稳定，所有工具回调和轨迹都可调试。
- [ ] 当前租卡预算与面试时间允许 2～3 天 RL 探针，不影响交付现有项目。

不满足时选择 B 分支：补充样本/置信区间、第二 QA 数据集、商品池检索或可选推荐小实验。

## Day 18～20 A 分支｜小规模 GRPO（可选）

**按天推进，不因时间不足跳过验证：** Day 18 对齐上游训练环境、Tool Adapter、Reward 和 Action Mask，先跑 20～50 Step 的 Smoke Test；Day 19 在通过 Smoke Test 后扩充固定训练数据，跑 Answer-only 与 Cost-aware 两组可比训练；Day 20 冻结 Checkpoint，执行同一固定验证集评测、绘制质量—成本曲线、检查是否出现不搜索/只搜索的退化。如果 Day 18 仍无法稳定完成 Rollout，当天停止增加 RL 工程投入。

### A. 目标与实现顺序

**优先只在 QA 上做 GRPO**，不在三天内同时训练商品 Agent。先问“能否优化无效 Search/Rerank 与 Stop”，再研究更复杂的 Retriever Policy。

1. 固定能稳定输出工具动作的起点模型/Checkpoint。若工具使用很差，先用 Prompt/小规模 SFT 改善格式，不指望 RL 自动修好所有协议错误。
2. 对接 Search-R1 与其锁定 veRL 版本；若改用更新版 veRL/SGLang 多轮工具接口，应视为**移植工作**，不是把原版依赖混进当前环境。
3. 重点检查工具 Observation、Retriever 文本不应被当成 Agent 生成 token 计算训练梯度；校验 action mask、reward/advantage 分配是否正确。
4. 先用 20～50 Step Smoke Test 看 Reward、KL、OOM、格式有效率、工具调用分布及是否能续训，再决定是否扩大到约 200～400 Step。Step 范围是个人实验计划，不是性能保证。
5. 至少对照 `Before RL`、`Answer-only GRPO`、`Answer + Cost GRPO`；若基础实验可靠且有 gold supporting facts，再考虑 Evidence Reward。

### B. Reward 设计

\[
R_{answer}=\mathrm{EM}(\hat y,y), \quad
R_{format}=\mathbf 1[\text{tool actions valid}].
\]

\[
R=R_{answer}+\alpha R_{format}
 -\beta\operatorname{Norm}(C_{retrieval})
 -\gamma\operatorname{Norm}(C_{rerank})
 -\delta\operatorname{Norm}(C_{tokens}).
\]

实验次序：先 `Answer-only`，再 `Answer + Cost`，最后可选 `Answer + Evidence`。成本按训练集观测量纲归一化，不要因权重失衡鼓励“永远直接答、不搜索”。同时报告 `no-search% / always-hybrid% / always-rerank% / budget-exhausted% / invalid-action%`。

若 `Hybrid` 代理成本已包含内部分支 BM25/Dense 调用，公式里不要重复扣款。要用独立固定 QA Eval 验证训练前后质量与成本变化；不只给训练 Reward Curve。

### C. GPU 和训练风险

- 原 Search-R1 `train_grpo.sh` 的默认 8 卡、大 Batch、原版依赖**不是**个人单卡配置。减卡时必须共同缩小 Batch、Rollout Group、Search Turns、Observation/Response 长度，并确认 FSDP/LoRA/Offload 是否由当前代码版本支持。
- 1 × RTX 4090 可以尝试更小模型 + LoRA 的 RL 技术探针，但完整 3B 多轮全参训练不能预先保证。多张 80GB 卡也应先跑样本级 Smoke Test 再租长时实例。
- Search-R1 Retriever Service 与 Train/Rollout Service 分进程；资源不足时 ANN/Reranker 放 CPU、限制并发，追踪 OOM、Timeout 和 Rollout Token 长度。

**A 分支完成标准**：在相同固定 Eval 上拿到**训练前/后**的 EM/F1、Search Calls、总成本、格式错误率和路由分布；没有可重复改善时不得把“GRPO 优化有效”写到简历。

## Day 18～20 B 分支｜不用 RL 的高价值替代

**按天推进：** Day 18 先扩大商品实验并做 Gain 敏感性；Day 19 画 QA 不同搜索预算下的 Pareto 曲线并开展统计检验；Day 20 根据面试目标补充 2Wiki/更大语料的鲁棒性验证，或在已有余力时开始独立的用户行为推荐小实验。

按主线价值依次补：

1. 扩大 ESCI Query 样本，做 `nDCG@10 vs Latency`、不同 E/S/C/I Gain 的稳健性、Query-level bootstrap CI。
2. 在 QA 做 `Budget ∈ {1,2,3,5}` 的 Answer Quality–Cost 曲线，检查难题是否值得多搜、简单题是否早停。
3. 加入 2WikiMultihopQA 或更接近全库的统一 QA 检索池验证，不因增加“数据集数量”牺牲设定清楚和可复现。
4. **只有目标岗位非常偏推荐且有额外时间**，才增加用户行为/ItemCF 小实验，不在最后两天仓促说已完成推荐泛化。

## Day 21｜项目封版与面试交付

**代码**：一键运行 QA Baseline/Ours、ESCI Baseline/Ours；完整 README、数据下载/拆分说明、锁定版本、已测资源消耗、配置与种子。至少保留 `v1-baseline`、`v2-adaptive-qa`、`v2-commerce` 三个 Tag（如果实际完成），RL 独立分支。

**研究结果**：架构图一张、QA 强基线总表一张、模块消融一张、Quality–Cost Pareto 两张、ESCI 对照一张、3 个成功轨迹和 3 个失败轨迹。RL 若完成另给 Reward Ablation；不要将尚未做的实验放进结果部分。

**面试讲述**：30 秒概括、2 分钟技术路径、5 分钟深挖。固定叙事：“业务/研究问题 → Static/Fixed Funnel 不足 → State/Action/Tool 设计 → 与强基线的公平实验 → 质量成本权衡 → 跨域适用边界”。

---

# 6. 最终实验表模板（只用真实结果替换 TBD）

## 6.1 QA 主结果

| Method | EM↑ | F1↑ | Evidence Recall@5↑ | Avg Calls↓ | Rerank Calls↓ | Avg Tokens↓ | P95 Latency↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct LLM | TBD | TBD | — | 0 | 0 | TBD | TBD |
| Static BM25 RAG | TBD | TBD | TBD | 1 | 0 | TBD | TBD |
| Multi-turn BM25 | TBD | TBD | TBD | TBD | 0 | TBD | TBD |
| Always Hybrid + Rerank | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Rule Router | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Adaptive Policy / Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.2 策略消融

| Variant | EM/F1 | Evidence Recall | Avg Cost | Rerank% | Tool 选择分布 | 失败类型 |
|---|---|---|---|---|---|---|
| Full Strategy | TBD | TBD | TBD | TBD | TBD | TBD |
| − Evidence-conditioned Rewrite | TBD | TBD | TBD | TBD | TBD | TBD |
| − Adaptive Router | TBD | TBD | TBD | TBD | TBD | TBD |
| − Rerank Decision（Always） | TBD | TBD | TBD | 100% | TBD | TBD |
| − Dynamic Stop | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.3 ESCI 商品检索

| Method | nDCG@10↑ | MRR@10↑ | P50/P95 Latency↓ | Avg Tool Cost↓ |
|---|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |
| Always Hybrid + Rerank | TBD | TBD | TBD | TBD |
| Rule Router | TBD | TBD | TBD | TBD |
| Adaptive Policy | TBD | TBD | TBD | TBD |

注意标明 `Setting A：同一官方已标注候选池` 和 ES/C/I Gain 映射。若做 Setting B，另表报告已标注正例覆盖率，不与 Setting A 排序指标混用。

## 6.4 GRPO 消融（只有实际训练后才使用）

| Policy | EM/F1↑ | Avg Search Calls↓ | Avg Cost↓ | No-search% | Invalid-action% |
|---|---:|---:|---:|---:|---:|
| Before RL | TBD | TBD | TBD | TBD | TBD |
| Answer-only GRPO | TBD | TBD | TBD | TBD | TBD |
| Answer + Cost GRPO | TBD | TBD | TBD | TBD | TBD |
| + Evidence Reward（可选） | TBD | TBD | TBD | TBD | TBD |

---

# 7. 失败风险、降级方案与工程细节

| 风险 | 早期信号 | 解决办法 |
|---|---|---|
| CUDA/vLLM/FlashAttention 冲突 | Day 1 模型无法稳定推理 | 先用兼容的 Transformers 跑 V1；原版 RL 独立环境 |
| 检索池太大 | 索引/内存/磁盘爆炸 | CPU ANN，缩小并明确公开语料设定，不偷偷只给 gold 段落 |
| Rewrite 幻觉 | 新 Query 的实体来自模型猜测 | 记录实体的证据来源，限制改写的事实注入，统计 unsupported entity |
| Agent 重复搜索 | Search Calls 高、证据高度重复 | 查询去重、观察压缩、Budget 和 Stop Validator |
| Ours 打不过强固定 Funnel | 动态路由成本更高、质量反而差 | 优先做错误分桶和 Rule Router，找适用边界，不强行上 RL |
| ESCI Label/Recall 误用 | 未标注产品被当负例 | 主做固定已标注候选排序；全库检索只报已标注正例覆盖 |
| 测试集泄漏 | 根据 final eval 修改 Prompt/阈值 | 固定 Manifest，使用 validation 调参，冻结后运行测试 |
| RL 反复 OOM | Smoke Test 无法完成 | 取消 RL，补商品/QA 主线消融和统计；不得写未经验证 RL 提升 |
| 秋招面试临近 | Day 14 还没有强基线结果 | 立即冻结新功能，优先完成可解释的 QA/ESCI 对照 |

## 7.1 验证代码的最低门槛

- 工具 JSON 必须能通过 Schema Test；超时、空列表、异常结果都不能造成无限循环。
- 数据切分先按 `query_id/question_id` 去重；ESCI 选择官方 split 后再从 train 划 validation。
- 同一 Query 的 BM25/Dense/Hybrid 候选池有可校验 ID，不得把 Text 拼接顺序误当标签。
- 每个工具调用保存 Query、Retrieval Method、Candidate IDs、Latency、Tokens 和是否命中证据。
- 正式实验允许重复运行并具有稳定的 Manifest/Seed/版本，所有实际异常进入报告。

## 7.2 简历表述与三个面试重点

**V2 简历写法（按实际完成项删减）**：

> 基于 Search-R1 风格的多轮检索框架，统一证据驱动 Query Rewrite、BM25/Dense/Hybrid 召回及 Neural Rerank 为可动态调用的 Search Tools；设计状态感知的 Retriever/Rerank/Stop Policy，在 HotpotQA 上与 Static RAG、固定多轮 Agent、Always Hybrid + Rerank 和规则路由比较答案质量、证据召回及搜索成本，并将共享 Search Gateway 与策略结构扩展至 Amazon ESCI 商品相关性任务，分析跨任务复用和适用边界。

**只有确实完成时才能补充**：基于 GRPO 引入答案正确性与工具成本联合 Reward，分析训练前后平均 Search Calls、质量—成本曲线与无效工具行为变化。未测得确定收益时，写训练探索而不是性能改善。

- **问：为什么不用 Always Hybrid + Rerank？** 先承认它是强基线，再以相近质量下实际延迟/成本数据说明动态策略的价值；如果没有优势，清晰说明适用范围和瓶颈。
- **问：什么是 Agent 的技术创新？** 强类型动作、可观察状态与预算约束；在多轮证据条件下动态控制 Query/Recall/Rerank/Stop，并用模块消融而非 Demo 数量验证。
- **问：能迁移到推荐/广告吗？** Tool API、决策结构和质量—成本分析方法可复用；商品相关性实验不能直接证明个性化效果。推荐还需用户行为与协同反馈，广告还需收益预估、竞价/预算约束；相关技能与已有实习互补。

---

# 8. 推荐阅读与实现参考（官方资料优先）

以下链接是复现需要核查的源资料。依赖和脚本以**实际锁定的 Git commit**为准；本计划中每天任务、样本数、默认 Budget 和 Reward 权重均为建议，而不是官方实验既有结果。

1. [Search-R1 原作者 GitHub](https://github.com/PeterGriffinJin/Search-R1)：原始推理/训练框架、检索服务器、NQ 示例、可选的 PPO/GRPO。
2. [Search-R1 原版 GRPO Shell](https://github.com/PeterGriffinJin/Search-R1/blob/main/train_grpo.sh)：默认 Qwen2.5-3B、8 卡等超参；需要依据租用配置裁剪并测试，不能原样在单卡使用。
3. [Search-R1 Retriever 文档](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)：Sparse、Dense、CPU ANN 与 GPU Flat 的适用条件。
4. [HotpotQA 官方说明](https://hotpotqa.github.io/) 与 [官方代码](https://github.com/hotpotqa/hotpot)：区分 distractor/fullwiki、Gold Supporting Facts 和评估设定。
5. [Amazon Shopping Queries / ESCI 官方数据](https://github.com/amazon-science/esci-data)：ESCI 标签、商品字段、small/large 标记、train/test 分割及推荐用法。
6. [veRL 多轮工具调用说明](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)：如果从 Search-R1 原版迁移到新版 veRL/SGLang，需独立核对 Tool Schema、Rollout 和依赖兼容性。

---

# 9. 每天 5 分钟收工 Checklist

- [ ] 新功能有单测和至少一条可重放的真实 Agent Trajectory 吗？
- [ ] 与哪个 Baseline 比、是否同数据/同模型/同索引/同预算，都记录了吗？
- [ ] 质量变化是否只是更多 Search/Token 或更强模型带来的？
- [ ] 今天的 3 个关键 Badcase 和错误归因保存了吗？
- [ ] 如果明天突然开始高密度面试，当前版本是否已具备可完整讲述的结果？

**终极原则：先完成可审查、可复现、能打强基线的 V2，再决定是否追求 GRPO。真正有价值的不是把更多功能挂在 Agent 后面，而是证明动态 Search Policy 在什么条件下值得使用、能带来何种实际收益。**
