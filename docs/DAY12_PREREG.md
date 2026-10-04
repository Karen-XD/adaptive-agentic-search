# Day 12 test 预先登记（2026-10-04，跑 test 之前写好并 commit）

目的：在没碰过的 test 上一次性回答"按需升级比强固定基线 B3 好不好、贵多少"。本文件 commit 之后才跑 test；
test 结果出来后不改门槛、提示词、配置，不挑组；全部组都报告，好坏都写。

## 数据

| 数据集 | test | 来源 |
|---|---|---|
| HotpotQA | 500 题（`data/hotpotqa/v1/labels/test.jsonl`，Day 2 建好后从未运行过） | 官方 dev 抽样 |
| 2Wiki | 800 题，每类题型 200（`data/2wiki/v1/labels/test.jsonl`） | 官方 dev 分层抽样（`prepare_2wiki.py`，seed 20261004 = 20261002 + 2），和 train 不重叠 |

2Wiki 按题型分层，所以报告"全部"之外也按题型报告。

## 在线跑的组（贪心解码，seed 0，配置一字不改，和 validation 相同）

| 组 | HotpotQA 配置 | 2Wiki 配置 | 角色 |
|---|---|---|---|
| B3：Dense + 重排，搜一次 | `qwen3b_static_rag_dense_rerank` | `qwen3b_2wiki_static_rag_dense_rerank` | 强固定基线 |
| 全量多轮 Agent（Dense + 重排，最多 3 次搜索） | `qwen3b_agent_dense_rerank` | `qwen3b_2wiki_agent_dense_rerank` | 贵的对照 |
| cascade 每题都探测（rewrite） | `qwen3b_cascade_always` | `qwen3b_2wiki_cascade_always` | 升级上限端，离线推算用 |
| **cascade 分差门控（rewrite）** | `qwen3b_cascade_gap`（门槛 4.19） | `qwen3b_2wiki_cascade_gap`（门槛 5.69） | 主方法 1：省成本 |
| **cascade 每题都 agent 探测** | `qwen3b_cascade_always_agent` | `qwen3b_2wiki_cascade_always_agent` | 主方法 2：要质量 |

门槛沿用 validation 多 seed 验证过的"另一个数据集上选的"值，不在本数据集的 validation 上重选，因为多 seed 结论就是这组配置得出的。

离线推算组（贪心下精确，门控不影响探测看到的内容）：分差门控 + agent 探测（门槛同上）= B3 运行 + 每题 agent 探测运行逐题拼接。

## 指标和判定

- 主指标 EM；同时报 F1、证据召回、输入 token、模型调用次数、检索次数、p50 / p95 延迟
- 显著：和 B3 逐题配对 bootstrap（10000 次，seed 0），95% 区间下界 > 0
- 预期（来自 validation）：
  - H1 分差门控 > B3：2Wiki 预期显著（validation 多 seed 3/3），HotpotQA 预期为正但可能不显著（0/3）
  - H2 每题 agent 探测 > B3：两个数据集都预期显著（各 2/3）
  - H3 全量多轮 Agent 不显著优于 B3，输入 token ≥ 3 倍
- 质量–成本曲线：在 test 上用"每题都探测"运行离线扫门槛，**只作描述**，不用来选门槛

## 规则

- 每组在 test 上只跑一次。只有基础设施故障（服务挂掉、`valid=false`）才允许重跑，重跑要在 PROGRESS 里记原因
- 多 seed 只在 validation 上做（已完成，Day 11.6），test 只跑贪心
- 2Wiki 全量 Agent 在跑 test 前先在 validation 上跑一次（Day 12 新增的配置，确认能正常跑完）
