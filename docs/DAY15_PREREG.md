# Day 15 商品 test 预先登记（2026-10-05，跑 test 之前写好并 commit）

目的：在没碰过的 ESCI test（400 条查询，官方 test 划分，`data/esci/v1/labels/test.jsonl`）上，一次性检验"领域微调的重排模型是否优于通用方法"。test 结果出来后不换检查点、不改 max_length、不挑组。

## 设定

- Setting A：每条查询的全部已标注候选商品当固定候选池，所有方法对**完整候选池**排序
- 指标：nDCG@10，增益 E=3 / S=2 / C=1 / I=0（主），另报 C=0 口径、E 的 Recall@10、MRR
- 显著：逐条查询配对 bootstrap（10000 次，seed 0），95% 区间不含 0

## 组（一字不改，和 validation 运行 `20261004-234931` 相同）

| 组 | 说明 |
|---|---|
| dense | e5-base-v2 向量相似度 |
| generic_rerank | 通用 bge-reranker-base，max_length 256 |
| **ft_rerank** | 微调后的重排模型，**固定用最后一步检查点** `esci_reranker_v1/last`（训练 2 轮、10232 步，`model.safetensors` sha256 前 16 位见下方），不用按 validation 挑出的 `best` |

检查点：`/root/autodl-tmp/checkpoints/esci_reranker_v1/last`，训练代码 commit `5644c35`，评测代码 commit `cb4f538`，sha256 前 16 位 `8b7c60efcf20d5e2`。

用 `last` 而不是 `best` 的原因：`best`（第 9500 步，0.8788）是按 validation 挑出来的，`last`（0.8782）是固定训练预算的结果，没有用 validation 做任何选择；两者只差 0.0006。

## 预期（来自 validation 400 条）

- H1 ft_rerank > dense，显著（validation +2.9 [+1.8, +4.1]）
- H2 ft_rerank > generic_rerank，显著（validation +2.7 [+1.6, +3.8]）
- H3 generic_rerank 和 dense 无显著差异（validation +0.3 [−0.7, +1.2]）
- H4（描述性，样本小）含否定词的查询上收益最大（validation 30 条，+10.6）
- 按需升级（Dense → 微调重排）：预期门控信号 AUC 约 0.5，"每条都重排"即最优策略；只作描述

## 规则

- test 只跑一次；只有基础设施故障才允许重跑，并记录原因
- 训练数据只来自官方 train，剔除了 validation / debug 的 425 条查询；官方 test 和官方 train 的 query_id 零重叠（训练脚本里有断言）
