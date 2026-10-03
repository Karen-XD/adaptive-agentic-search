# 项目进度日志

> 每次会话开始先读本文件；每完成一个小步骤就更新。

## 当前位置

**V1 完成（Day 1～7，2026-09-29），标签 `v1-baseline`。总结见 `docs/V1_REPORT.md`（结论、框架图、主表、稳健性、3 成功 + 3 失败案例、30 秒介绍、Stop Point 1 检查）。**
结论：多轮搜索稳定地提高证据召回（+8.5～+12.5，4 种解码设定都显著），准确率没有稳定优势（EM 差距 −2.0～+4.5）；错误从"搜不到"转移成"读不对"和"停不准"。复现性：四种方法重跑 800/800 逐字一致。

**⏸ Day 11 进行中（2026-10-03 15:05 因断网暂停，下次从"Day 11 下一步"第 2 条接着做）：cascade 代码、离线推算、HotpotQA 在线、agent 探测 +8.0 的拆解（11.4）已完成；2Wiki 在线（11.5）断网时还在 tmux 里跑，多 seed、文档还没做。最新 commit 见 git log（cascade 代码在 `ccc54ad`）。**

**断网时留在服务器上跑的任务（tmux `cascade2wiki`，预计 15:30 前全部结束）**：
1. ✅ 2Wiki 分差门控（门槛 5.69，样本外）：运行 `20261003-144150-qwen3b-2wiki-cascade-gap-validation`。**EM 0.379，− B3 +3.2 [+2.0, +4.6] 显著**；探测 30%、再搜 24%、检索 1.24 次、输入 994 token、p50 / p95 326 / 1490ms → 和离线推算（0.379 / 30% / 994）完全一致，2Wiki 上离线推算也是精确的
2. ⏳ 2Wiki agent 探测（每题都探测）：运行 `20261003-144949-qwen3b-2wiki-cascade-always-agent-validation`，断网时 780/800
3. ⏳ 2Wiki 格式诊断（`user_answer_only,tool_agent` 两组，约 15～20 分钟）：输出 `outputs/runs/<时间>-day11-answer-format-qwen3b-2wiki-static-rag-dense-rerank-validation`，日志 `/root/autodl-tmp/logs/answer_format_2wiki.log`

回来后先检查：`tmux ls`（`cascade2wiki` 不在了 = 跑完或被关机打断）；`tail -3 /root/autodl-tmp/logs/answer_format_2wiki.log` 最后一行是 `-> outputs/runs/...` 才算跑完；看 2、3 的输出目录里有没有 `metrics.json`。缺哪个就重跑哪个（先按恢复清单起 vLLM 8000；2 还要起 2Wiki 检索服务 8101，3 不需要检索服务）：
```bash
python -m evaluation.run_eval --config configs/qwen3b_2wiki_cascade_always_agent.yaml --split validation   # 2
python -m experiments.day11_answer_format_probe --context-run outputs/runs/20261002-164844-qwen3b-2wiki-static-rag-dense-rerank-validation \
    --labels data/2wiki/v1/labels/validation.jsonl --arms user_answer_only,tool_agent                     # 3
```
被打断的运行留下的不完整目录删掉再重跑（没有 `metrics.json` 的就是不完整的）。

**11.4 结论：agent 探测的 +8.0 ≈ +5.5「模型有把握时自己直接答」+ +2.5「真的再搜一次」；单纯换作答格式不显著 → B3 不改。**
- 按 outcome 和 B3 同题配对：answered 92 题 40 → 51（赢 15 输 4，证据和 B3 逐字相同、召回都是 0.891）；escalated 91 题 35 → 40；format_error 14 / duplicate 3 题不变
- 格式 2×2 诊断（`experiments/day11_answer_format_probe.py`，运行 `20261003-144020`；证据直接取 B3 轨迹里存的检索结果，"用户消息 × 只能作答"组和 B3 200/200 逐字一致）：

  | 证据位置 × 系统提示 | EM | − B3 |
  |---|---|---|
  | 用户消息 × 只能作答（= B3） | 0.405 | — |
  | 用户消息 × Agent（想搜就按预算用完强制作答） | 0.385 | −2.0 [−7.0, +3.0] |
  | 工具返回 × 只能作答（17.5% 格式错误：想调用没声明的 search） | 0.390 | −1.5 [−7.0, +4.0] |
  | 工具返回 × Agent | 0.445 | +4.0 [−1.0, +9.0] |

  "工具返回 × Agent"的 +4 拆开：模型自己选择直接作答的 92 题 +11，想搜却被强制作答的 108 题 −3 → 收益不是"格式更好"，而是"**有把握就用 Agent 格式答、没把握就别硬答**"。B3 整体换格式不显著，且会动到所有基线，不改
- 离线推算 agent 探测的变体（探测看到的内容和门控无关，贪心下精确；"每题都探测 + 再搜"复现在线 0.485）：

  | HotpotQA | EM | − B3 | 检索 | 输入 token |
  |---|---|---|---|---|
  | agent 探测，想搜就再搜（在线） | 0.485 | +8.0 [+3.0, +13.0] | 1.46 | 1125 |
  | agent 探测，不再搜（想搜的退回 B3 答案） | 0.460 | +5.5 [+1.5, +10.0] | 1.00 | 1045 |
  | 分差门控 4.19 + agent 探测 | 0.425 | +2.0 [−1.5, +5.5] | 1.19 | 783 |
  | 分差门控 5.69 + agent 探测 | 0.425 | +2.0 [−0.5, +5.0] | 1.14 | 724 |

- **分差门控和 agent 探测不互补**：直接作答赢的 15 题里 11 题分差 ≤ 4.19（门控放过）。分差门控挑的是"证据不够、该再搜"的题，而直接作答的收益在"证据已经够、B3 读错了"的题上（answered 92 题里 64 桥接、28 比较）
- 目前 HotpotQA 的质量–成本前沿两个点：分差门控（rewrite 探测）0.430 / 846 token；每题 agent 探测 0.485 / 1125 token（全量多轮 Agent 0.390 / 2314 token）。要等 2Wiki 和多 seed 确认

做什么：B3 先搜一次（Dense + 重排）→ **门控**决定要不要花一次模型调用去**探测** → 探测时模型写了新查询才搜第二次，否则用第一跳证据作答。代码 `agent/cascade.py`，配置 `configs/qwen3b_cascade_*.yaml`、`configs/qwen3b_2wiki_cascade_*.yaml`，脚本 `experiments/run_cascade.sh`、`experiments/day11_cascade_offline.py`，测试 `tests/test_cascade.py`（共 165 个测试通过）。
- 门控 `gate`：`never`（= B3）/ `always` / `rerank_gap`（重排第 1、2 名分差 > 门槛才探测，不调模型）
- 探测 `probe`：`rewrite`（Day 10 证据改写提示词，只有 search 工具）/ `agent`（Agent 提示词，可以直接 final_answer，省一次作答调用）
- 设计关键：`gate=always + probe=rewrite` 看到的内容和 `two_hop_evidence` 逐字相同、`gate=never` 和 B3 逐字相同 → 贪心解码下门控策略可以用 B3 和两跳两次运行**离线精确推算**

**离线推算（validation，`python -m experiments.day11_cascade_offline`）**：

| | HotpotQA：EM / 输入 token / 耗时 | 2Wiki：EM / 输入 token / 耗时 |
|---|---|---|
| B3（从不探测） | 0.405 / 589 / 321ms | 0.346 / 683 / 322ms |
| 每题都探测 | 0.425 / 1312 / 1086ms | 0.386 / 1569 / 1151ms |
| 理想门控（只升有益的题，上限） | 0.435 / 618 | 0.403 / 751 |
| 分差门控，本数据集选门槛 | 门槛 5.69：探测 25%、EM 0.425、771 | 门槛 4.19：探测 40%、EM 0.390、1099 |
| **分差门控，另一个数据集选的门槛（样本外）** | 门槛 4.19：探测 34%、EM 0.430、846 | 门槛 5.69：探测 30%、EM 0.379（拿到升级收益的 81%）、994 |

- 信号：重排分差（大 → 需要升级）预测"升级有益"的 AUC 约 0.74（两个数据集都是）；前 3 名平均分稍弱，第 1 名分数最弱
- 和随机门控比：探测 30% 的题时，分差门控拿到升级收益的 125% / 81%（HotpotQA / 2Wiki），随机只有 27% / 30%
- 模型"拒绝再搜"从不误伤：升级有益的题（6 / 45）全部愿意再搜；有害的题（2 / 13）也全部愿意再搜 → 门控只负责省探测成本，"要不要真的再搜"交给模型

**HotpotQA 在线验证（validation 200 题，commit `ccc54ad`，运行 `20261002-205730` / `210114` / `210317`）**：

| | EM | − B3（配对 bootstrap） | 检索次数 | 模型调用 | 输入 token | p50 / p95 |
|---|---|---|---|---|---|---|
| B3 | 0.405 | — | 1.00 | 1.00 | 589 | 308 / 368ms |
| 两跳证据改写（Day 10） | 0.425 | +2.0 [−0.5, +5.0] | 2.00 | 2.00 | 1312 | 1099 / 1644ms |
| cascade 每题都探测（rewrite） | 0.425 | +2.0 [−0.5, +5.0] | 1.45 | 2.00 | 1312 | 1104 / 1636ms |
| **cascade 分差门控（门槛 4.19，样本外）** | **0.430** | **+2.5 [+0.5, +5.0]，显著** | 1.21 | 1.34 | **846** | **338** / 1475ms |
| cascade 每题都探测（agent 提示词） | **0.485** | **+8.0 [+3.0, +13.0]，显著** | 1.46 | 1.54 | 1125 | 905 / 1535ms |

- **离线推算精确**：在线"每题都探测"和两跳证据改写的最终答案 200/200 逐字一致；在线分差门控和离线推算 200/200 一致 → 以后调门槛可以离线做，不用重跑
- **分差门控是目前最好的质量–成本点**：只探测 34% 的题、21% 的题真的再搜，拿到了全部升级收益（EM 甚至比每题都探测高 0.5），输入 token 只有 B3 的 1.44 倍（每题都探测是 2.2 倍），p50 几乎不变（338 vs 308ms），只有被升级的题拉长 p95
- **agent 探测的 +8.0**（outcome：escalated 91 / answered 92 / format_error 14 / duplicate 3）：来源已在 11.4 拆清（见本节开头），不是单纯的作答格式，B3 不改

**Day 11 下一步（按顺序）**：
1. ~~拆 agent 探测 +8.0 的来源~~（11.4 完成，见本节开头）
2. 2Wiki 在线收尾：确认上面 2、3 跑完后，拆 2Wiki 的 agent 探测（和 11.4 同一个口径，按题型看；2Wiki 标签里题型字段同样是 `type`）：
   `python -m experiments.day11_agent_probe_breakdown --labels data/2wiki/v1/labels/validation.jsonl --b3 outputs/runs/20261002-164844-qwen3b-2wiki-static-rag-dense-rerank-validation --agent outputs/runs/20261003-144949-qwen3b-2wiki-cascade-always-agent-validation --gap 4.19 5.69`
   重点看：agent 探测的收益在 2Wiki 上是否还是"直接作答"占大头；组合题（桥接实体在第一跳拿不到）上"再搜"是否占大头；格式诊断 tool_agent 组是否仍不显著（决定"B3 不改格式"在两个数据集上都成立）
3. 多 seed：分差门控和 agent 探测各跑 3 个采样 seed（temperature 0.7），按 V2 口径（多数 seed 显著才算显著）。采样下离线推算不再精确，要在线跑
4. 更新 `docs/LEARNING_NOTES.md`（级联 / 门控 / 离线推算的原理 + 面试问答）、`docs/PROJECT_OVERVIEW.md`、计划文件的 Day 11 部分
5. 之后：计划里 Day 11 的统一 Policy 接口 / BudgetManager（cascade 已经覆盖"升级 / 不升级"这个核心动作，看是否还需要单独抽象），再进 Day 12 主实验

**服务状态（2026-10-03 15:05 断网前）**：tmux `vllm`（8000）、`retriever_2wiki`（8101）、`cascade2wiki`（实验链）在跑；`retriever`（8100，HotpotQA）没起。用户可能设了定时关机：关机后 tmux 全部消失，按下面的恢复清单重启；做 HotpotQA 实验时要停掉 8101、起 8100。

**Day 10.4 完成（2026-10-02）：2Wiki 上证据条件改写的收益更大、显著（EM +4.0、组合题 +14.0）；只看问题的静态拆解没用；"拒绝再搜"信号在 2Wiki 上 385 题零翻转。**

2Wiki validation（每类题 200，共 800）、贪心、Dense + 重排、作答提示词和 B3 相同：

| EM（召回） | 比较 | 桥接比较 | 组合 | 推理 | 全部 |
|---|---|---|---|---|---|
| **B3：原问题搜一次** | 0.560 (0.980) | 0.655 (0.500) | 0.105 (0.542) | 0.065 (0.573) | 0.346 (0.649) |
| 原问题取前 6 条 | 0.445 | 0.555 | 0.105 | 0.095 | 0.300 (0.668) |
| 两跳：静态拆解 | 0.510 | 0.580 | 0.105 (0.550) | 0.065 | 0.315 (0.653) |
| **两跳：证据改写** | 0.560 (0.998) | 0.640 (0.549) | **0.245 (0.740)** | 0.100 (0.667) | **0.386 (0.738)** |

- **B3 选 Dense + 重排**：Hybrid + 重排 EM +0.2 [−1.1, +1.6]，打平；少维护一路，和 HotpotQA 一致。（Day 8 离线 Hybrid 略强 +0.5 的召回差没有变成准确率差）
- **单次检索碰不到第二跳**：把金标段落分成"问题里点了名的实体"和"桥接实体"（名字只出现在第一跳段落里），B3 的召回分别是 **0.980 和 0.066**。Day 8 的"候选池里就没有"现在有了精确的数字
- **证据改写 − B3：EM +4.0 [+2.1, +5.9]、召回 +9.0，都显著**；组合题 EM +14.0 [+9.0, +19.0]、召回 +19.8；推理题 EM +3.5（显著）。桥接实体召回 0.066 → **0.257**（组合题 0.077 → 0.474）
- **静态拆解没有用**：EM −3.1 [−5.8, −0.6] 显著更差、召回 +0.5 不显著；组合题 EM 0.105 → 0.105。拆出来的子查询只能用问题里已有的名字（"Who is the director of film X?"、"Alice Claypoole Vanderbilt's husband"），桥接实体召回 0.077 → 0.077，和 B3 一模一样；1600 个子查询里有 4 个写成占位符（"When was [director name] born?"）
- **证据改写 − 静态拆解：EM +7.1 [+4.2, +10.1]**（组合题 +14.0）；**证据改写 − 前 6 条：EM +8.6 [+6.0, +11.2]**（组合题 +14.0）→ 和 HotpotQA 同一个结论，收益来自"第二跳看到第一跳的结果"，不是多搜、不是多看段落
- **多看段落会伤答案**：前 6 条在比较类题上 EM −11.5 / −10.0（显著），57 题由对变错里 **48 题是 yes/no 题答成了 "false"/"False"** → 段落多了，模型更倾向于照抄段落里的措辞而不是按问题形式作答（答案形式问题，V1 就有）；静态拆解同理（比较类掉 5～7.5）。证据改写的组在比较题上大多拒绝再搜，所以没有掉
- **"拒绝再搜"信号**：800 题里 **385 题拒绝再搜，EM 和 B3 完全相同（零翻转）**；415 题写了查询，45 题错→对、13 题对→错。按题型：比较题 169/200 拒绝（全部第一跳已全齐）；组合题 159/200 再搜（错→对 29、对→错 1）。桥接比较题 127 题拒绝但第一跳证据**全都不齐**（这类题要 4 段，前 3 条不可能齐）→ 信号在这里是"模型觉得够了"，而不是"证据真的齐了"，EM 照样不变
- 成本（每题平均）：B3 输入 683 token、p50 310ms；证据改写 1569 token（2.3 倍）、p50 1148ms（3.7 倍）
- 运行：`20261002-164844`～`171917`（commit `c2ae5ef`）；脚本 `experiments/run_2wiki.sh`；检索服务 2Wiki 在 8101，HotpotQA 的 8100 期间停掉（显存）

**Day 10 完成（2026-10-02）：证据条件改写有效（召回 +4.5 显著），一次性的静态改写反而有害；改写步"拒绝再搜"是很好的按需升级信号。**

检索栈全部是 Dense + 重排（和 B3 相同），作答提示词一字不差，validation 200 题、贪心：

| 组 | 检索 | EM | 证据召回 | 两段全齐 | in-tok | p50 |
|---|---|---|---|---|---|---|
| **B3 原问题** （Day 9） | 原问题，1 次 | 0.405 | 0.823 | 0.685 | 589 | 308ms |
| 静态改写**代替**原问题 | 模型改写，1 次 | 0.320 | 0.715 | 0.505 | 857 | 569ms |
| 原问题取前 6 条 | 原问题，1 次 | 0.385 | 0.848 | 0.715 | 962 | 333ms |
| 两跳：原问题 → 静态改写 | 2 次 | 0.360 | 0.833 | 0.695 | 945 | 672ms |
| **两跳：原问题 → 证据改写** | 2 次 | **0.425** | **0.868** | **0.770** | 1312 | 1099ms |
| Agent + Dense + 重排（Day 9） | 模型自己决定 | 0.390 | 0.807 | 0.660 | 2314 | 1525ms |

配对 bootstrap（对照都是 B3）：
- **静态改写代替原问题：EM −8.5 [−13.5, −3.5]、召回 −10.8 [−14.2, −7.2]，都显著更差** → 只看问题、不看证据的一次性改写，比原问题差得多。这也解释了 V1 里"Agent 第一个查询往往不如原问题"
- 原问题取 6 条：召回 +2.5 显著，EM −2.0 不显著 → 多给段落救不了答案
- 两跳静态改写：EM −4.5 [−8.0, −1.5] 显著更差 → 第二跳如果不看证据，等于白搜
- **两跳证据改写：EM +2.0 [−0.5, +5.0]（不显著）、召回 +4.5 [+2.5, +6.5]（显著）**；两段全齐 0.685 → 0.770
- **证据改写 − 静态改写：EM +6.5 [+3.0, +10.5]，显著** → 收益来自"第二跳看到第一跳的结果"，不是"多搜一次"
- 证据改写 − 原问题@6：EM +4.0 [−0.5, +8.5]、召回 +2.0 [−0.2, +4.2] → 第二跳拿到的是新证据，不是多看的段落

**改写步本身是升级信号（本轮最有用的发现）**：证据改写那一步，模型手里有第一跳的全部段落、提示词让它再搜一次，结果 **200 题里 111 题（56%）直接写出了答案、拒绝再搜**（代码记成 fallback，退回原问题检索）：
- 这 111 题里 90 题第一跳证据已经全齐；证据和 B3 逐条相同，**EM 和 B3 完全一样（52 → 52，零翻转）** → 模型说"够了"几乎没有误判
- 另外 89 题写了第二跳查询的：47 题其实第一跳已全齐（有点想多），EM 29 → 33；其中补到新金标的 18 题里 **6 题由错变对**，没补到新金标的 71 题只错了 2 题 → 搜错了代价也很小
- 成本：B3 基础 1 次检索 + 1 次作答；证据改写组 2 次检索 + 1 次改写 + 1 次作答，输入 token 2.2 倍、p50 3.6 倍。**"该不该升级"判对一次，省下的是整组多出来的成本** → Day 11～12 的按需升级就用这个信号
- 运行：`20261002-1606xx`～`1609xx`（commit `28cf4af`）；脚本 `experiments/run_rewrite.sh`，配置 `configs/qwen3b_{rewrite_rag,two_hop_static,two_hop_evidence,static_rag_dense_rerank_top6}.yaml`

**Day 9 完成（2026-10-02）：HotpotQA 的 B3 强固定基线定为 Static RAG + Dense + 重排（只搜一次）；在这个强检索下，多轮 Agent 不比单次检索强。**

| validation 200 题，EM | 贪心 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| Static RAG + BM25（V1 B1） | 0.320 | 0.320 | 0.310 | 0.340 |
| Static RAG + Dense | 0.300 | 0.305 | 0.295 | 0.320 |
| **Static RAG + Dense + 重排（B3）** | **0.405** | 0.395 | 0.405 | 0.430 |
| Static RAG + Hybrid + 重排 | 0.390 | 0.380 | 0.380 | 0.405 |
| Agent + BM25（V1 B2） | 0.355 | 0.300 | 0.355 | 0.335 |
| Agent + Dense + 重排 | 0.390 | 0.360 | 0.440 | 0.365 |
| Agent + Hybrid + 重排 | 0.390 | 0.370 | 0.420 | 0.370 |

配对 bootstrap，按 V2 口径（多数 seed 显著才算显著）：
- B3 − V1 Static RAG：EM +7.5～+9.5，**4/4 显著**；F1 +11.9～+13.0，4/4 显著 → 换强检索 + 重排是这一阶段最大的单项提升
- 重排的贡献（Dense + 重排 − Dense）：EM +9.0～+11.0，**4/4 显著**
- Hybrid + 重排 − Dense + 重排（Static RAG）：EM −1.5～−2.5，0/4 显著；F1 −2.2～−3.1，**3/4 显著变差** → Dense + 重排当 B3：不比 Hybrid 差，而且少跑一路 BM25
- **Agent + Dense + 重排 − B3：EM −6.5～+3.5，0/4 显著**；F1 1/4 显著变差。Agent + Hybrid + 重排同样 0/4
- 成本：B3 每题 1 次检索、输入 589 token、端到端 p50 308ms；Agent + Dense + 重排 1.72 次检索、输入 2314 token（3.9 倍）、p50 1525ms（5 倍）。证据召回 B3 0.823 反而高于 Agent 的 0.807（Agent 改写的查询不如原问题，Day 3 已见过）
- **按需升级的空间**：B3 和 Agent + Dense + 重排逐题取较好的一个，EM 0.505～0.545（比 B3 高 11～13 个点，含解码噪声）；4 个设定里 Agent 稳定赢（≥3 次）16 题、B3 稳定赢 22 题。Agent 稳定赢的 16 题里 **10 题 B3 第一次检索就已证据全齐** → 升级的触发信号不能只看"证据缺不缺"
- 运行：贪心 4 组（commit `2d14968`）、seed 15 组（commit `6a12252`）；脚本 `experiments/run_b3_candidates.sh`、`experiments/run_seeds_b3.sh`

**对 V2 主线的影响**：要打败的不再是"Agent 多搜几次"，而是一个只搜一次、又便宜又强的 B3。V2 的问题变成**按需升级（cascade，级联）**：默认走 B3，只在信号表明需要时才升级成多轮。上限约 +8（稳定口径）～+13（单次口径），成本主要省在"不升级"的题上。

> **更正（2026-10-02）**：Day 8.6 写的"检索已经不是瓶颈"说过头了。Dense 提高的是多段并集的召回，但两段金标同时进前 3 的题只从 56 涨到 95；加重排后涨到 137，EM 才跟着涨（0.300 → 0.405）。瓶颈在**前 3 条的精度**（两段金标是否同时进前 3），不在召回广度。阅读问题也仍在（B3 里"证据齐 + 答错"71 题）。

**下一步：** Day 10 查询改写（原问题 / 静态改写 / 证据条件改写），在 B3 的检索栈（Dense + 重排）上做；同时准备按需升级的信号（答案一致性 × 证据信号，见待办）。2Wiki 的 B3 还没端到端跑（离线 Hybrid + 重排略强 +0.5）。

**Day 8 检索部分完成（2026-10-02）：BM25 / Dense / Hybrid 三路检索走同一个 `/search`，验收通过。** validation 200 题、原问题搜一次的发现（运行 `20261002-114642-day8-retriever-compare-validation-dirty`，代码未提交时跑的，重跑逐题一致）：
- **Dense（e5-base-v2）大幅领先 BM25**：召回@3 0.565 → 0.710（+14.5 [+9.8, +19.3]），两段都找齐@3 0.28 → 0.475；比较题 0.585 → 0.976，桥接题 0.560 → 0.641
- **Dense 搜一次（0.710）已经高于 V1 Agent 用 BM25 搜多轮（0.677）**：换检索器的收益比多轮的收益还大。同一批 Agent 查询换成 Dense 重放：0.677 → 0.765（+8.8 [+4.5, +13.0]）
- **等权 Hybrid 没有超过 Dense**：整体召回@3 −2.2 [−5.8, +1.3]；比较题显著更差（−12.2 [−18.3, −6.1]），BM25 把只"提到"实体的段落挤进前排；只在前 10～20 条略高（0.848 vs 0.825）
- 互补性有限：前 5 条里的金标，两路都找到 229 段、只有 Dense 79 段、只有 BM25 20 段（全是桥接题）、都没找到 72 段
- 成本（CPU、进程内）：BM25 中位数约 8ms（原问题）/ 27ms（Agent 短查询，原因未查）；Dense 约 40～55ms（查询编码约 21ms + FAISS 暴力搜约 27ms）；Hybrid 约 65ms。V1 每次模型调用中位数 492ms → 检索耗时不是成本大头。内存：BM25 +0.9GB、Dense +2.2GB；磁盘 449MB / 1.8GB

**Day 8.6 换检索器重跑基线（2026-10-02）：召回大幅提高，准确率没有跟着提高。**

| validation 200 题 | Static RAG (BM25) | Static RAG (Dense) | Agent (BM25) | Agent (Dense) |
|---|---|---|---|---|
| EM | 0.320 | 0.300 | 0.355 | 0.375 |
| F1 | 0.428 | 0.433 | 0.487 | 0.507 |
| 证据召回 | 0.565 | **0.710** | 0.677 | **0.735** |
| 证据全齐的题 | 56 | **95** | 98 | 108 |
| 平均搜索次数 | 1.00 | 1.00 | 1.88 | 1.84 |

- 配对 bootstrap：Static RAG EM **−2.0 [−8.0, +4.0]**，Agent EM **+2.0 [−4.5, +8.0]**，都不显著
- **错因四分法**（证据是否全齐 × 是否答对），Static RAG：
  - BM25：齐+对 26 / 齐+错（读不对）30 / 缺+对（蒙对）38 / 缺+错（搜不到）106
  - Dense：齐+对 38 / 齐+错 **57** / 缺+对 22 / 缺+错 83
  - 证据全齐多了 39 题，但只有 12 题从错变对，另外 27 题**补上证据以后还是答错**；"蒙对"从 38 降到 22
  - Agent：齐+对 47→51，齐+错 51→**57**，缺+错 78→68
- **召回变化和答案变化对不上**：Static RAG 里召回变高的 69 题净 +7，召回变低的 18 题净 −6，**召回不变的 113 题净 −5**（换了检索器，即使召回水平一样，答案也会变，7 题由对变错）
- ~~结论：检索已经不是瓶颈了~~ → **Day 9 更正**：说过头了。Dense 提高的是召回广度，但两段金标同时进前 3 的题不够多；加重排后 EM 从 0.300 涨到 0.405（4/4 显著）→ 瓶颈在前 3 条的精度。见"当前位置"的更正
- 运行：`20261002-124659-qwen3b-static-rag-dense-validation`、`20261002-124808-qwen3b-agent-dense-validation`（commit `fbb162b` 之后，代码干净）；配置 `configs/qwen3b_{static_rag,agent}_dense.yaml`，脚本 `experiments/run_baselines_dense.sh`

**Day 8.7 路由上限分析（2026-10-02）：结论是"换检索器"这条线的空间很小，而且换第二个数据集也救不回来；真正的空间在查询构造。** 两数据集同一套代码（`experiments/routing_ceiling.py`）：

| 口径（金标召回@3） | HotpotQA validation | 2Wiki analysis |
|---|---|---|
| 固定 BM25 | 0.565 | 0.530 |
| 固定 Dense | **0.710** | 0.588 |
| 固定 Hybrid | 0.688 | **0.602** |
| 按题型路由 | — | 0.605 |
| 逐题理想（路由上限） | 0.765（+5.5） | 0.637（+3.4） |
| 三路都找不到的金标 | — | 桥接比较 52.5% / 组合 46% / 推理 41.7% / 比较 3.4% |

- **最强的一路依数据集而变**：HotpotQA 上 Dense 显著赢 Hybrid（−2.2 [−5.8, +1.3]，不显著）；2Wiki 上 Hybrid 显著赢 Dense（+1.4 [+0.7, +2.2]）。→ "总是混合检索"这个计划默认值在一个数据集上对、另一个上错，说明 B3 强基线必须按数据集用数据挑，不能写死
- **路由上限只有 3～5 个点，真实路由器再打个对折，落在噪声里**。2Wiki 上 Dense 赢 437 题、BM25 赢 168 题、**1395 题打平**（70%）；HotpotQA 上 69 / 18 / 113
- **理想子查询（4563 跳，来自标签里的（实体, 关系, 值）三元组，金标是该实体的段落）**：Dense 0.968、BM25 0.771、Hybrid 0.910、逐跳理想 0.980，三路全落空只有 2%
  - 换成"实体 + 关系"的干净查询，Dense 几乎解决任务；**Hybrid 比 Dense 低 5.8 [−6.6, −5.0]，显著**（BM25 弱 20 个点，融合时被拖累）
  - 同一个语料、同一个模型，只换查询构造方式，召回从 0.588（整句问题）到 0.968（实体 + 关系）→ **查询构造的影响 ≫ 选哪一路检索器**
- 组合题（2Wiki 的 compositional，如"Mina Gerhardsen 父亲的出生日期"）：整句提问 0.524，拆解成子查询 0.975。这是计划里标为"拓展"的 Query Decomposition 的直接数据支持
- 按题型路由 ≈ 固定最强那一路（2Wiki 上 0.605 vs 0.602），题型规则没有额外价值

**下一步：** ① 提交 Day 8 代码，在干净 commit 上重跑对比；② 换检索器重跑 Static RAG / Agent（真反事实，要起 vLLM），看召回优势能不能转成准确率；③ Day 9 Reranker + 强固定基线（B3 = 每个数据集上实测最强的那一路 + 重排，见待办）；④ 2Wiki 作为第二个分析集，用来测 Query Reformulation / Decomposition 和跨数据集稳健性。
V2 评测口径已定：贪心 + 3 个采样 seed，多数 seed 显著才算显著；检索侧主指标证据召回。

> 2026-09-29 思考题回顾（"换 seed 答案就变，能不能变成信号"）：用户答"一致说明分布尖锐、确定；不一致说明不确定，可以多给搜索次数"。数据（Agent 3 个采样 seed）：
> - 支持：3 个一致 72 题多数票 EM 0.56、召回 0.74；2 个一致 76 题 0.30、0.65；全不一样 52 题 0.00、0.59 → 一致性是很准的置信度信号，不确定的题证据也更少
> - 补充一：被"不一致"标出的比例，过早停止 66%，**证据够了仍读错 77%**，答案形式 69% → 只能发现"有问题"，分不清是检索还是阅读；"不确定就多搜"会把搜索浪费在阅读错误上，要和证据侧信号组合路由
> - 补充二：3 个一致的 72 题里 32 题答错（11 题证据全齐）→ 一致 ≠ 正确，不能单独当停止依据
> - 补充三：线上不能整条轨迹跑 3 遍；只在作答步用 vLLM `n=K` 采样（一次预填充、K 个短答案），或读答案 token 的 logprobs
> - 另：3 个 seed 多数票 EM 0.32，低于贪心 0.35 → 在这个设定下投票本身不提分，只当信号用

> 2026-09-29 思考题回顾（"检索错误 83 → 12，最后只多对 7 题，多轮搜索有没有用"）：用户答"范式有用、优势没发挥出来，可能是模型推理能力不足，要换更大的模型"。补算宽松口径（答案形式算对）：Direct 0.215 / Static RAG 0.455 / **Agent 0.535** / Oracle 0.775；Agent − Static RAG **+8.0 [+1.0, +15.0]，显著**（桥接 +8.8 [+1.3, +16.4]，比较 +4.9 [−12.2, +22.0]），EM 下是 +3.5 [−3.0, +10.0]
> - 支持"范式有用"：EM 低估了多轮的收益，Agent 多答对的题有不少被答案写法吃掉了 → Day 6 答案规范要先做
> - "换大模型"要打问号：阅读侧 74 题里 35 题是写法问题，不是能力问题；换模型会让所有方法一起涨，证明不了范式的价值，范式要在同一模型下比。模型尺寸消融放在后面，在 Oracle 设定下做
> - 注意：宽松口径是诊断，不替代 EM；答案规范修完后要看 EM 本身能不能显著
> - **2026-09-29 更正**：宽松口径 +8.0 只在贪心和 seed 2 上显著，seed 1（+3.0 [−3.5, +9.5]）、seed 3（+3.5 [−3.5, +10.5]）都不显著 → "宽松口径下显著"是只看一次贪心结果得出的，不成立（见 6.2）

> 2026-09-29 思考题回顾（"0 新文档就强制作答"会怎样误伤、怎么验证）：用户答"检索超时会误伤；验证做消融"。补充：
> - 超时：检索出错是 `ok=False` + `tool_error`，但 `num_new_docs` 同样是 0 → 规则只能作用在 `ok=True` 的搜索上。validation 上检索出错 0 次，现在的实验看不出这种误伤
> - 离线估算（validation 现有轨迹）：触发 30 题，**只能省 9 次搜索（2.4%）**；证据已齐 7 题（停得对）/ 没齐且之后也没找到 19 题（不亏，但本来就在白搜）/ **没齐但之后又找到新金标 4 题（2 题最终答对，误伤）** → 不值得做硬停止规则
> - 0 新文档更多意味着"这条查询路子卡住了"，而不是"信息找完了" → V2 里把它当**换路信号**（拆子问题 / 换向量检索），而不是停止信号
> - 验证分三层：离线反事实（零调用，初筛）→ **前缀重放**（只对触发的题，重放轨迹前缀后发强制作答；贪心解码下是精确反事实，约 30 次调用）→ 同预算完整消融（配对 bootstrap，质量和成本一起报）

> 2026-09-29 思考题回顾（"第一次检索永远用原问题"会怎样）：用户答"准确率升、成本降"。离线估算（validation 现有轨迹，未跑新实验）：
> - 支持的部分：原问题一次就找齐金标的 56 题（28%）上，Static RAG EM 0.46 > Agent 0.38，Agent 改写反而丢分；Agent 找齐证据后 78%（76/98）立刻停，停止能力尚可 → 简单题上准确率可能回升、成本下降；另外第一轮不用调模型写查询，省一次模型调用。
> - 要修正的部分："只会更好"没有保证：① 搜索上限 3 次，已有 45 题用满、36 题被强制作答，原问题占掉一次会挤占难题的搜索名额（若不计入预算则成本上升）；② 8 题改写更好，第一轮会变差；③ 贪心解码有路径依赖，换了第一轮观察整条轨迹都变。唯一有保证的是证据召回 ≥ Static RAG（并集），准确率没有单调性。
> - 结论：列为 V2 消融"原问题保底召回"，分计入 / 不计入预算两种，**必须在同等预算下比较**（见待办）。

### 🔌 服务器重启后的恢复清单（2026-09-27 关机前写）

关机（非释放实例）后两个盘都在，代码、数据、索引、模型、记忆文件都还在；**只有 tmux 会话会消失**。

```bash
# 1. 确认资产都在（应输出 6 行都存在）
ls -d /root/autodl-tmp/adaptive-agentic-search/data/hotpotqa/v1 \
      /root/autodl-tmp/adaptive-agentic-search/indexes/hotpot_pool_v1_bm25 \
      /root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct \
      /root/autodl-tmp/adaptive-agentic-search/indexes/hotpot_pool_v1_e5 \
      /root/autodl-tmp/hf_models/e5-base-v2 \
      /root/autodl-tmp/hf_models/bge-reranker-base

# 2. 重启检索服务（Day 3 跑实验前必须启动；Day 8 起带 --dense-index，Day 9 起带 --reranker；加载约 30s、内存约 3.5GB、显存约 0.7GB）
#    先起 vLLM 再起检索服务也可以，两者显存加起来约 21.5GB
tmux new -d -s retriever "source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 \
  && cd /root/adaptive-agentic-search \
  && python -m retrieval.server --index indexes/hotpot_pool_v1_bm25 --dense-index indexes/hotpot_pool_v1_e5 \
     --reranker /root/autodl-tmp/hf_models/bge-reranker-base --port 8100"
curl -s http://127.0.0.1:8100/health   # methods 里应有 bm25 / dense / hybrid，num_docs 都是 507494；reranker 不为空

# 3. 重启 vLLM 模型服务（Day 3 起，约 50s 就绪，显存占约 19GB）
#    --guided-decoding-backend 必须加：vLLM 0.6.3 默认后端 outlines 缺依赖，每个请求都会 500
mkdir -p /root/autodl-tmp/logs
tmux new -d -s vllm "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl_env \
  && python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3b-instruct --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --seed 0 --disable-log-requests --guided-decoding-backend lm-format-enforcer \
  2>&1 | tee /root/autodl-tmp/logs/vllm.log"
curl -s http://127.0.0.1:8000/v1/models   # 应列出 qwen2.5-3b-instruct

# 3b. 跑 2Wiki 时另起一个检索服务（端口 8101）；三个服务一起显存 23.4/24.5GB 太紧，先停 HotpotQA 的 retriever
# tmux new -d -s retriever_2wiki "source /root/miniconda3/etc/profile.d/conda.sh && conda activate dsr1 \
#   && cd /root/adaptive-agentic-search && python -m retrieval.server --index indexes/2wiki_pool_v1_bm25 \
#      --dense-index indexes/2wiki_pool_v1_e5 --reranker /root/autodl-tmp/hf_models/bge-reranker-base --port 8101"

# 4. 自检
cd /root/adaptive-agentic-search && conda activate dsr1 && pytest tests/ -q   # 165 passed
```

若资产丢失（例如释放了实例），按本文件「数据与索引位置」一节的命令重建；模型用 `/root/Search-R1/download_model_modelscope.sh` 重新下载。

> 2026-09-27 用户反馈：讲解和提问要宏观优先（每步做什么 / 为什么 / 结论 / 全局位置），实现细节由 Claude 决定并记在决策表，不逐条提问。已写入 `CLAUDE.md` 和记忆；宏观全景见 `docs/PROJECT_OVERVIEW.md`。

## Day 11 子步骤

- [x] 11.1 离线分析：门控信号（重排分差 / 前 3 平均分 / 第 1 名分数）的 AUC、质量–成本曲线、和随机门控对比、门槛跨数据集迁移（`experiments/day11_cascade_offline.py`）
- [x] 11.2 `agent/cascade.py` + `Escalation` 轨迹字段 + `probe_rate` / `escalation_rate` / `escalation_outcomes` 指标；`run_eval` 读配置里的 `cascade:` 段；测试 11 个（含和 B3、两跳证据改写的逐字等价）
- [x] 11.3 HotpotQA 在线：每题都探测（验证离线推算）、分差门控、agent 探测（结果见"当前位置"）
- [x] 11.4 拆 agent 探测 +8.0 的来源：outcome 分组配对 + 作答格式 2×2 诊断（`experiments/day11_answer_format_probe.py`）+ 离线推算 agent 探测变体（结果见"当前位置"）
  - 诊断脚本不连检索服务：证据直接取 B3 轨迹里存的检索结果（`Context.model_validate`），"用户消息 × 只能作答"组直接调 `run_episode`，必须和 B3 逐字一致才算复用没走样（200/200）
  - Agent 系统提示下模型想搜：回"预算用完"的报错（和 Agent 循环同一条路径）再让它作答；格式错误照常回报错重试，最多 max_turns 轮
- [ ] 11.5 2Wiki 在线：分差门控 ✅（在线和离线推算一致，+3.2 显著）；agent 探测、格式诊断断网时在跑，回来先确认跑完（见"当前位置"）
- [ ] 11.6 多 seed 稳健性
- [ ] 11.7 文档：学习笔记、项目全景、计划文件

## Day 10 子步骤

- [x] 10.1 改写流程 `agent/rewrite.py`：固定检索计划（原问题 / 静态改写 / 证据条件改写），证据去重合并后交给作答步；作答提示词和 B3 一字不差。预算校验按计划长度（rewrite_rag 1、两跳 2）
  - `SearchRecord` 记录每次检索的查询、来源、改写原文、fallback、新文档数、耗时和 token；上下文记录 `num_searches()`，成本口径和 Agent 一致
  - 解析失败退回原问题并记 fallback（不重试，重试次数会让各组成本不同）；退回的检索照样执行、照样计费，固定流程的预算必须相同
- [x] 10.2 debug 上踩的两个坑（都记进决策）：
  - 第一版把证据放在用户消息里交给改写步 → **50 题里 0 次用上证据中的新实体、0 次先写推理**，几乎都在复述原问题；同一个模型在 Agent 循环里第二次搜索前 76% 会先写推理、35% 用上证据实体。改成 Agent 的对话格式（自己发过的 search + 工具返回）后，用上证据实体的比例升到 22%，先写推理 50/50 → 不训练的模型，改写的上下文格式要顺着它的微调格式
  - 只看问题的静态改写改得动：去掉"Which/What"、加"birth year"、丢年份，和 Agent 第一次改写的行为一致 → V1 的"改写不如原问题"是真的，不是 Agent 特有的
- [x] 10.3 validation 四组 + 与 B3 / Agent 对照：结果见"当前位置"
- [x] 10.4 2Wiki 上的问题拆解和证据改写（结果见"当前位置"）
  - 2Wiki 新增 validation（每类 200）和 debug（每类 25）划分，从 train 中 analysis 没抽到的题里抽、三者互不重叠；语料和 analysis 逐字节不变（sha256 校验），索引不用重建
  - 静态拆解 `two_hop_decompose`：新工具 `decompose`（参数是子查询列表），一次调用拆出最多 2 个子查询依次检索；不够 2 个或解析失败的位置退回原问题。和两跳证据改写同样是 2 次检索 + 1 次模型调用
  - 改写的成本统计改成"模型调用次数"和"由改写给出查询的检索次数"分开（拆解一次调用对应两次检索）；退回率 / 重复率的分母是后者
  - 2Wiki 检索服务另起在 8101（`configs/qwen3b_2wiki_base.yaml`）；三个服务一起时显存 23.4/24.5GB，太紧 → 跑 2Wiki 期间停掉 HotpotQA 的检索服务
  - 新指标：金标分"点名实体"（标题去掉括号后出现在问题里）和"桥接实体"，分开算召回

## Day 9 子步骤

- [x] 9.1 重排模型：bge-reranker-base（XLM-RoBERTa base，1.1GB，hf-mirror 下载，sha256 `ced967c4…` 和 HF 一致）
  - 选它的原因：现成、base 尺寸、在 MS MARCO 等问答相关性数据上训练过；MiniLM 系列更快但更弱，v2-m3 更强但大 2 倍（568M 参数）、要更多显存，先用 base 量出重排有没有用
  - 冒烟（40 题 × 20 候选）：GPU fp16 每次 18ms（p95 25ms），CPU fp32 888ms → 必须放 GPU；fp16 和 fp32 前 3 名顺序 36/40 一致，最大 logit 差 0.012；显存约 0.7GB
- [x] 9.2 `retrieval/rerank.py`（`CrossEncoderReranker` + `RerankedSearchTool`，耗时拆成检索 / 重排 / 候选数）；服务 `/search` 加 `rerank` 开关（和 `method` 独立）；客户端、`run_eval`（配置 `retrieval.rerank`）跟着改；测试 135 个通过
- [x] 9.3 离线重排对比 `experiments/day9_rerank_compare.py`（6 种组合：3 个检索器 × 重排与否，候选池 20）
  - HotpotQA validation（召回@3）：BM25 0.565 → **0.740**（+17.5）、Dense 0.710 → **0.823**（+11.3）、Hybrid 0.688 → **0.828**（+14.0），都显著；两段都齐@3：Dense 0.475 → 0.685
  - 重排后 Dense ≈ Hybrid（−0.5 [−2.0, +0.7]）：**重排抹平了检索器之间的差距**，候选池里只要有金标，重排就能把它排上来
  - 补上的缺口（重排后@3 − 重排前@3）/（候选池@20 − 重排前@3）：BM25 88%、Dense 73%、Hybrid 77%
  - 比较题：Dense / Hybrid + 重排前 3 条召回 1.000；桥接题 Dense + 重排 0.777（候选池@20 只有 0.830，上限就在这里）
  - V1 Agent 查询重放（top_k=3）：BM25 0.677 → 0.795、Dense 0.765 → 0.828、Hybrid 0.760 → 0.830
  - 2Wiki analysis：Dense 0.588 → 0.651、Hybrid 0.602 → **0.657**（重排后 Hybrid 略强，+0.5 [+0.2, +0.9]，显著但很小）；组合题只到 0.546、桥接比较题 0.516 → **重排救不了"候选池里没有"**（组合题候选池@20 只有 0.573）
  - 耗时：每次重排 20 条候选，GPU 中位数 16ms（HotpotQA）/ 19ms（2Wiki），p95 23～29ms；输入长度中位数 135 token、p95 255
- [x] 9.4 端到端 B3 候选：Static RAG / Agent × Dense + 重排 / Hybrid + 重排，贪心 + 3 个 seed；结果见"当前位置"
  - 检索服务带 `--reranker` 启动后显存 21.5GB（vLLM 约 19.5GB + 重排约 0.7GB + CUDA 上下文），单并发串行，没有显存竞争
  - 服务里重排首次调用 370ms（冷启动），热身后约 22ms；端到端检索耗时 p50：Dense 65ms、Dense + 重排 81ms、Hybrid + 重排 101ms
  - 第一次跑四组时，第 1 组跑完后我改了 PROGRESS.md，runner 拒绝了后 3 组 → 提交后在 `2d14968` 上四组全部重跑，残缺的那组挪到 `/root/autodl-tmp/aborted_runs/`（Day 6 踩过同一个坑：**批量运行期间不改仓库文件**）
- [x] 9.5 Static RAG 错因四分法（贪心）：BM25 / Dense / Dense + 重排 / Hybrid + 重排的"齐+对"26 / 38 / **66** / 64，"齐+错"30 / 57 / 71 / 74，"缺+错"106 / 83 / **48** / 48；第 1 条就是金标 147 / 167 / 182 / 184 题
  - 两种都证据全齐的 93 题：Dense EM 36 → Dense + 重排 39；3 条段落完全相同、只差顺序的 28 题：13 → 16 → 顺序本身有一点影响，但重排的收益主要来自把第二段金标挤进前 3
  - 按题型 EM（BM25 / Dense / Dense + 重排）：桥接 0.289 / 0.270 / **0.403**，比较 0.439 / 0.415 / 0.415 → 收益全在桥接题

## Day 8 子步骤

- [x] 8.1 向量索引（2026-10-02）：e5-base-v2（ModelScope 下载，sha256 和 HF 一致）；`dsr1` 装 faiss-cpu 1.15.1（numpy 2.2.6 不变）；`retrieval/dense.py`；50.7 万段 GPU fp16 编码 229s，`IndexFlatIP` 1.8GB
  - 冒烟：2000 段小语料，按标题搜自己 20/20 排第一；fp16 和 fp32 编码的余弦相似度 ≥ 0.9999995
- [x] 8.2 混合检索 `retrieval/hybrid.py`（等权 RRF，k=60，每路取 20）；服务 `/search` 加 `method`；客户端、`run_eval`（配置 `retrieval.method`）跟着改；新增 `tests/test_dense_hybrid.py`，测试 131 个通过
- [x] 8.3 检索对比 `experiments/day8_retriever_compare.py`（validation 200 题）：结果见"当前位置"。BM25 重放 V1 Agent 的 419 个查询，和轨迹里的结果逐条一致（ID 映射没漂移）；两次运行逐题一致
- [x] 8.4 抽查只有一路找到的金标（前 5 条）：
  - 只有 BM25（20 段，全是桥接题）：问题里的罕见词出现在段落正文里，但段落讲的是另一个实体（"Shipwrecker" → Mayfair Games，"Johannes Bergion" → Diablo Swing Orchestra）。Dense 把整段压成一个向量，正文里只提一次的罕见词被稀释
  - 只有 Dense（79 段，比较题 25）：① 用描述代替名字（"Queen of Denmark" → Hamlet (1996 film)，"中国官方通讯社的社长" → Liao Chengzhi）；② 要找"讲的就是这个实体"的段落：Mick Jagger 被很多段落提到，BM25 把"提到"的排前面，Dense 把"关于他"的排前面
  - 一句话：**BM25 匹配"提到"，Dense 匹配"关于"**
- [x] 8.5 验收：检索服务带 `--dense-index` 起在 tmux `retriever`；三种 method 返回统一的 ID / 分数 / 名次 / 来源 / 耗时；不传 method 默认 bm25；未加载的 method 400、未知的 422
- [x] 8.6 提交（`fbb162b`）后在干净 commit 上重跑：① 检索对比结论不变；② 换 Dense 重跑 Static RAG / Agent（vLLM + 三路检索服务），结果见"当前位置"——**召回大涨、准确率不动，检索不再是瓶颈**
- [x] 8.7 路由上限分析（2026-10-02）：`data_prep/prepare_2wiki.py` + `experiments/routing_ceiling.py`；结果见"当前位置"
  - 2Wiki 数据：语料池 38.5 万段（train + dev 的上下文去重），分析集从 train 按题型分层各抽 500 题（2000 题）；金标 = supporting_facts 里的标题（比较 / 组合 / 推理题 2 段，桥接比较题 4 段），和 HotpotQA 同一口径
  - 2Wiki 的 evidences 三元组给了"理想子查询"的构造：每个（实体, 关系）造一个查询，实体名去掉括号里的消歧义词（Agent 从正文里读到的名字不带它）
  - 脚本先在 HotpotQA 上跑通，数字和 Day 8 完全一致（0.565 / 0.710 / 0.688，逐题赢家 69 / 18 / 113）

## Day 7 子步骤

- [x] 7.1 V1 收尾（2026-09-29）：`docs/V1_REPORT.md`
  - 成功案例的选法：Agent 在贪心 + 3 个 seed 下全对、Static RAG 4 次全错、而且靠第二次搜索补上金标 → 正好 3 题，都是桥接题（第二跳实体只有读了第一跳段落才知道）
  - 4 种解码设定下的稳定胜负：Agent 稳定胜 7 题，Static RAG 稳定胜 5 题
  - 失败案例选"多轮反而搞砸"：Willie Almond（第 1 次就拿到金标，多搜带来噪声，最后说找不到）、Germantown（查询带着错误假设，答案跟着查询走）、Shabbona（过早停止，4 次都答同一个错答案）；另记 Kooskia：和 Static RAG 证据完全相同，仍答错
- [x] 7.2 复现性验证：commit `72a6eaf` 重跑四种方法，每轮输出 / 检索结果 / 最终答案 800/800 逐字一致（服务重启过）
- [x] 7.3 Stop Point 1：四项全部达标（见报告第 9 节）→ 进入 V2；标签 `v1-baseline`


- [x] 6.1 答案规范：**试过、已回退**（2026-09-29，commit `f4d64fe` → revert `d8f6335`）
  - 改法：`final_answer` 工具描述加规则（是非题只答 yes/no；照抄文档里的答案原文，人名用全名，保留单位；不写句子）。四种方法共用
  - 结果（validation，逐题配对 bootstrap）：Direct **+6.0 [+2.0, +10.5]**（16 好 4 差）；Static RAG −1.0 [−5.0, +3.0]（8 好 10 差）；Oracle −3.0 [−9.0, +3.0]（15 好 21 差）；Agent **−6.0 [−12.0, 0.0]**（12 好 24 差），F1 −8.4 [−14.1, −2.6]
  - 原因一：**金标口径本身不一致**。"要全名"修对了 Michele Marie Bachmann、Eric Michael Hilton，又改错了 Tupac Shakur → Tupac Amaru Shakur；"照抄原文"让 Oracle 答成 "George Wallace (born July 21, 1952)"。答案形式那 35 题里有相当一部分是评测口径噪声，提示词修不干净
  - 原因二：**对 Agent 没有"局部改动"**。工具描述在系统提示词里，每轮都看得到；改完 154/200（77%）题的查询序列变了，62 题第一轮查询就不同。变差的 24 题里 20 题是搜索路径变了；只因答案写法变化的只有 4 题 → Agent 的 −6 主要是轨迹扰动，不是答案规范本身
  - Direct 是单轮，改动只影响那一次作答，所以效果干净 → **同一改动，单轮和多轮方法反应完全不同：多轮的扰动沿轨迹累积**
  - 事故：批量运行中途在仓库里新建了 `compare_runs.py`，runner 发现工作区脏了，拒绝后三种方法（只有 Direct 跑完）→ 提交后四种方法全部在同一 commit 重跑。批量运行期间不改仓库文件，临时分析脚本放 `/tmp`
  - 新增 `experiments/compare_runs.py`：两次运行逐题配对对比（指标变化、EM/F1 配对 bootstrap、变好 / 变差的题）
- [x] 6.2 解码波动（2026-09-29，commit `57d8266`）：`run_eval` 加 `--temperature` / `--seed`（run_id 带 `-t0.7-s1`）；`experiments/run_seeds.sh` 跑 Static RAG 和 Agent 各 3 个 seed，temperature 0.7
  - 为什么要温度采样：主结果是贪心解码，贪心下换 seed 输出不变，测不出波动

    | EM | 贪心 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |
    |---|---|---|---|---|---|
    | Static RAG | 0.320 | 0.320 | 0.310 | 0.340 | 0.323 ± 0.015 |
    | Agent | 0.355 | 0.300 | 0.355 | 0.335 | 0.330 ± 0.028 |

    | Agent − Static RAG（逐题配对 95%） | 证据召回 | 宽松准确率 | F1 | EM |
    |---|---|---|---|---|
    | 贪心 | **+11.2 [+7.5, +15.0]** | +8.0 [+1.0, +15.0] | +5.9 [−0.4, +12.3] | +3.5 [−3.0, +10.0] |
    | seed 1 | **+9.8 [+6.0, +13.5]** | +3.0 [−3.5, +9.5] | −1.4 [−7.8, +5.0] | −2.0 [−8.5, +4.5] |
    | seed 2 | **+12.5 [+9.0, +16.0]** | +7.5 [+0.5, +14.5] | +6.2 [−0.1, +12.4] | +4.5 [−2.0, +11.0] |
    | seed 3 | **+8.5 [+4.8, +12.2]** | +3.5 [−3.5, +10.5] | +0.6 [−5.9, +7.1] | −0.5 [−7.0, +6.0] |

  - **稳的结论**：证据召回优势 4 种设定都显著（+8.5～+12.5）
  - **不稳的结论**：EM 差距均值 +0.7、标准差 3.4，符号会翻；宽松准确率只有 2/4 显著
  - 逐题稳定性：3 个 seed 里结果不一致的题，Static RAG 13 题（6%），**Agent 55 题（28%）** → 多轮把采样随机性放大了（每轮都采样一次，差异沿轨迹累积）
  - 成本稳定：Agent 搜索 1.86～1.97 次、输入 token 2397～2520；Static RAG 固定 1 次、约 570


- [x] 5.1 错题分类脚本 `experiments/day5_error_taxonomy.py`（2026-09-29）：每题只归一个主因，按"最小修复"顺序判断（格式 → 标签 → 答案形式 → 证据够了仍读错 → 过早停止 → 查询写得差 → 无效重复搜索 → 缺第二跳 / 缺一个实体 → 检索未命中）；`--show qid` 逐轮重放；逐题结果写 `<运行>/error_taxonomy.csv`
  - 第一版把"过早停止"判成 38 题（最大一类），核实后修两处：① 答案形式改按词集合判断（"Michele Bachmann" vs "Michele Marie Bachmann" 子串判不出）；② 桥接题答案原文已在检索结果里 → 算证据够了（停下没错，是读错），纯数字答案除外（"17" 被无关段落碰巧命中）
  - `tests/test_error_taxonomy.py` 15 条，五处改坏检查都能抓到（第一轮漏了"搜满后作答不算过早停止"，补了测试）；测试 105 → 120
- [x] 5.2 分布（validation）：

  | 主因 | Agent | Static RAG | Oracle |
  |---|---|---|---|
  | 格式非法 / 标签问题 | 3 / 1 | 0 / 1 | 0 / 1 |
  | 答案形式 | 35 | 27 | 46 |
  | 证据够了仍读错 | 39 | 25 | 45 |
  | 过早停止 / 查询写得差 / 无效重复搜索 | 29 / 3 / 7 | — | — |
  | 缺第二跳 / 缺一个实体 / 检索未命中 | 5 / 2 / 5 | 49 / 11 / 23 | — |
  | 答错合计 | 129 | 136 | 92 |

  - Agent 把检索器能力的错误从 83 压到 12，但净提升只有 7 题：省下的大多变成了阅读侧和搜索行为的错误
  - **检索干扰**：Direct 对、Agent 错 10 题（反过来 57 题），其中 6 题证据够了仍读错；Static RAG 9 vs 49
- [x] 5.3 `docs/BADCASES.md`：21 条样例人工核对，17 条和自动分类一致；不一致的是"第一跳读错导致后面走偏"（Ghost Town）、别名（Kyle O'Reilly = Kyle Greenwood）、粒度（1932 vs 20 世纪）、以及原问题能搜到但真问题在阅读（Avengers）
  - 两处原先的推测被数据否掉：Aokigahara 题不是"词汇不匹配"（简介里写着 Suicide Forest，是 BM25 让电影页面压过了实体页，单独搜 "Suicide Forest" 也排不进前 3），模型自己也不知道答案（Direct 答 Kurama-ji Forest）


- [x] 4.1 格式失误加固（2026-09-29）
  - 盘点 validation 四种方法全部格式错误轮次：Agent 643 轮里错 32 轮，**26 轮是普通轮用纯文字写出结论**（之后 16 轮去搜、5 轮作答、5 轮又错）；强制轮 JSON 完整 + 后缀 4、引号没转义 1、纯文字 1；强制轮不听话去搜 2（不算格式错误）。只能作答的三种方法共 13 轮，多为 `})` 笔误（重试能改对）和同一题的引号问题（Static RAG 重试 4 次都一样）
  - 解析器：取 `<tool_call>` 后第一个完整 JSON，后缀丢弃并记 `Step.ignored_suffix`；单字符串参数骨架下修未转义引号，记 `Step.repaired_quotes`；`metrics.json` 新增 `lenient_parses`
  - 循环：解析失败的报错改用 user 消息发回，不再包进 `<tool_response>`
  - 离线重放（精确反事实，旧轨迹逐轮重新解析）：Agent 没作答 8 → 3、EM 0.350 → 0.355（救回 5 题只有 1 题答对）；Static RAG 救回 1 题（答错）；没有任何一题的轨迹分叉 → 重跑后的其余变化都来自报错角色的改动
  - 测试 96 → 105，四处改坏检查都能抓到

- [x] 4.2 试改"解析失败的报错改用 user 消息发回" → **被数据否掉、已回退**（2026-09-29）
  - 动机：解析失败说明没有合法工具调用，包进 `<tool_response>` 有点误导（纯文字作答后 16/26 轮又去搜）
  - 结果（validation 重跑）：Agent EM 0.350 → 0.355，和"只放宽解析"的离线重放完全一致 → 那 0.5 个点和报错角色无关；**Direct 变差**：同一个 `})` 笔误旧版第 2 轮改对，新版连着两轮原样重复，多 2 题没作答
  - 回退后（commit `0602ba7`）四方法整体重跑，和 Day 3.5 的表逐项只差解析放宽那一处：Direct 0.120→0.120、Static RAG 0.320→0.320、Agent 0.350→**0.355**（+0.5，bootstrap [0.0, +1.5]，1 题变好 0 题变差）、Oracle 0.540→0.540；Static RAG 格式错误率 2.5%→0；Agent 没作答 8→3，搜满/强制作答 36→41（救回来的题去多搜了一次）；`lenient_parses` 记下每种放宽各救回几轮（未闭合 18 / 丢后缀 4 / 修引号 1）
  - 其他数字完全没动（证据召回、搜索次数、token、轮数逐项相同）→ 这一处改动只影响解析失败的那些轮
- [x] 4.2c 观察窗口截断：**决定不做**（2026-09-29）。语料段落 P50 428 / P99 1350 / max 8237 字符，每轮 3 段最坏约 2500 token，加上限 3 次搜索，输入有硬上限、撑不到 8192，规则在 V1 里触发不了；留到 V2 换向量检索、top_k 调大后再做
- [x] 4.3 人工看多轮轨迹（2026-09-29，validation `20260929-210819`，按停止原因 / 失败模式抽 7 条，再对发现的现象全量计数）
  - 能找到第二跳：Sesame Street 题第 1 次搜得太宽（关键词全堆一起），第 3 次写成 "original executive producer Sesame Street" 才命中；桥接题多轮是有用的
  - 证据够了会停：找齐证据的 98 题里 76 题立刻作答（见思考题回顾）
  - 反复搜同一个词：完全相同的重复被拦 20 次；**词干化后相同但没拦住的只有 1 次**（"albums producer" / "albums producers"）→ 不放宽去重规则
  - **预算分配失衡**：21 题搜索次数用完后还想搜。例：比较题"谁出的专辑多"，3 次全花在 Thin White Rope 上，Graham Coxon 一次没搜，答 "unknown" → 模型不知道还剩几次。V1 按决策不告诉它；列为 V2 消融（"告知剩余搜索次数"）
  - 靠参数记忆答对：Benjamin Furly 题 9 段检索结果都没有 "1632"，证据召回 0.5，仍答对。全量统计：Agent 答对的非 yes/no 题里 3/62（5%）答案没出现在任何检索结果里，Static RAG 1/56 → EM 会略微高估检索能力，报表必须同时报证据召回
  - 纯文字输出 26 次，整段等于金标 0 次（都是带推理的句子）→ 严格解析不改
- [x] 4.3b 补看 6 条真多轮轨迹（凑够 plan 要求的 10 条），再全量计数：
  - **白搜**：实际执行 376 次搜索，0 篇新文档 32 次（9%，常见是在旧查询后加 "more detailed" 之类，BM25 结果不变）；证据已齐还在搜 24 次（6%）
  - **查询堆砌**：Parelaphostrongylus 题三次都把整句问题当关键词，一次没搜第一跳实体（寄生虫名），召回 0
  - **答案形式**：EM=0 但答案和金标互相包含 26 题（13%），如 "River Calder" vs "Calder"、"Vallejo, California" vs "California"；比 4.1 格式修复救回的多一个量级
  - **标注噪声**：金标答案超过 12 个词的 1 题（Neil Jordan 题的金标是一整句简介）
- **Day 4 验收**：多次不同查询 ✓；超时 / 格式错误不会无限循环 ✓（max_turns 兜底 + 测试）；检索库与 B1 相同 ✓（同一检索服务）；原始 / 改写查询 / 观察分字段 ✓（`Trajectory.question` / `step.action.arguments.query` / `step.observation`，Day 1 起就这样记，不用改）
- [x] 4.4 原地打转提醒：**写了、量化后没合入**（2026-09-29）。模型连着两轮写出一字不差的文字时插一条提醒；validation 上四种方法一共只会触发 4 题，且都已答错或最终作答 → 收益说不清，按"说不清收益就不加"不合入。空答案 `{"answer": ""}` 当前代码下只有 Direct 1 轮，schema 本来就拒绝（`min_length=1`），模型重试即可；只补了一条回归测试。测试 104 → 105


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
- [x] 3.5 人工看轨迹 + validation 批量跑四种方法（2026-09-28）
  - 人工看 debug 上 Agent 和 Static RAG 结果不同的 12 题 + 所有格式错误轮次：模型纯文字写出答案后，报错提示先举 search 的例子，它就又去搜了（4 次，其中一题因此没作答）→ 报错提示按可用工具生成、先给作答写法
  - `--split` 覆盖划分（一个方法一份配置）、test 要加 `--final`、Oracle 在建目录前就拒绝 test
  - `experiments/day3_query_rewrite_analysis.py`：Agent 查询 vs 原问题的金标覆盖；`experiments/run_baselines.sh`：四种方法依次跑
  - 正式运行（commit `e59506d`，validation 200 题，失败率均为 0）：

    | 方法 | EM | F1 | 证据召回 | 全找齐 | 搜索次数 | 输入 token | 输出 token | P50 / P95 延迟 | 格式错误率 |
    |---|---|---|---|---|---|---|---|---|---|
    | B0 Direct `232955` | 0.120 | 0.182 | 0 | 0 | 0 | 232 | 24 | 0.21s / 0.25s | 2.9% |
    | B1 Static RAG `233043` | 0.320 | 0.428 | 0.56 | 0.28 | 1 | 570 | 24 | 0.26s / 0.42s | 2.5% |
    | Agent `233230` | 0.350 | 0.482 | 0.68 | 0.49 | 1.88 | 2415 | 185 | 1.69s / 3.78s | 5.0% |
    | Oracle（诊断上限）`233141` | 0.540 | 0.699 | 1 | 1 | 0 | 459 | 22 | 0.21s / 0.26s | 1.0% |

  - 按题型 EM：桥接题（159）0.069 / 0.289 / 0.333 / 0.528；比较题（41）0.317 / 0.439 / 0.415 / 0.585
  - Agent − Static RAG（逐题 EM，bootstrap 95% 区间）：全体 +3.0 [−3.5, +9.5]，26 胜 20 负；桥接 +4.4 [−2.5, +11.3]；比较 −2.4 [−19.5, +14.6] → **都不显著**，debug 上"比较题 Agent 更差"的趋势方向仍在，但 41 题说明不了
  - 查询改写：Agent 第一个查询比原问题变差 26 题、变好 8 题（覆盖 0.56 → 0.52）；多轮后整条轨迹 0.68 → 多轮的收益来自多搜，不是改写本身写得好
  - 错因：Agent 证据齐了的 98 题里仍答错 52 题；Oracle 答错 92 题，其中 52 题 F1 > 0（部分对，多为答案形式）
  - Agent 没作答 8 题（debug 只有 1 题）：多数是强制作答那一轮 JSON 写完整后跟了乱码 token（`hendace`、`hendrix`），或答案里有未转义的引号

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
| 2Wiki 原始数据（373MB） | `data/raw/2wiki/` | hf-mirror `xanhho/2WikiMultihopQA` 的 train / dev parquet，sha256 见 manifest |
| 2Wiki 语料池 + 分析集（175MB） | `data/2wiki/v1/` | `python -m data_prep.prepare_2wiki` |
| 2Wiki BM25 索引（293MB） | `indexes/2wiki_pool_v1_bm25/` | `python -m retrieval.bm25 build --corpus data/2wiki/v1/corpus.jsonl --index indexes/2wiki_pool_v1_bm25` |
| 2Wiki Dense 索引（1.3GB） | `indexes/2wiki_pool_v1_e5/` | `python -m retrieval.dense build --corpus data/2wiki/v1/corpus.jsonl --index indexes/2wiki_pool_v1_e5 --model /root/autodl-tmp/hf_models/e5-base-v2` |
| bge-reranker-base（1.1GB） | `/root/autodl-tmp/hf_models/bge-reranker-base/` | hf-mirror `BAAI/bge-reranker-base` + aria2c，只下 safetensors 和分词器文件（sha256 `ced967c4…`，和 HF 一致） |
| e5-base-v2（438MB） | `/root/autodl-tmp/hf_models/e5-base-v2/` | ModelScope `intfloat/e5-base-v2` + aria2c，只下 PyTorch 推理要的文件（sha256 `d0d559c4…`，和 HF 一致） |
| Dense 索引（1.8GB） | `indexes/hotpot_pool_v1_e5/` | `python -m retrieval.dense build --corpus data/hotpotqa/v1/corpus.jsonl --index indexes/hotpot_pool_v1_e5 --model /root/autodl-tmp/hf_models/e5-base-v2`（GPU，约 4 分钟；vLLM 占着显存时先停） |
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
| 2026-09-28 | 格式报错里附的正确写法按方法可用的工具生成；Agent 的提示先给 final_answer 再给 search | 只能作答的方法没有 search；最常见的格式错误是纯文字写出答案，先举 search 会把它带去重搜 |
| 2026-09-28 | 一个方法一份配置，划分用 `--split` 指定；run_id 带划分名；跑 test 必须加 `--final` | 同一方法在 debug / validation / test 上配置完全一样；test 只在最后跑一次，平时误跑会忍不住按它调参 |
| 2026-09-28 | 批量运行的日志写在仓库外（`/root/autodl-tmp/logs/`） | 写在仓库里会让工作区变脏，runner 拒绝运行（第一次启动就被拦下了，把关有效） |
| 2026-09-28 | 方法间比较报逐题配对的 bootstrap 95% 区间，不只报均值 | 200 题上 3 个点的差距在噪声范围内；只看均值会把噪声当结论 |
| 2026-09-29 | 答案规范提示词回退，不再逐条调提示词 | Agent −6.0 [−12.0, 0.0]；金标口径本身不一致，且对 Agent 任何提示词改动都会扰动 77% 的轨迹，调提示词等于在噪声里调参 |
| 2026-09-29 | **V2 起方法比较报：贪心 + 3 个采样 seed（temperature 0.7）**，每个 seed 单独做逐题配对 bootstrap，另报 seed 间均值和标准差；只有多数 seed 都显著才写"显著" | Agent 的 EM 差距 seed 间标准差 3.4 个点，和方法差距同一量级；只看贪心会把一次采样当结论（宽松口径就是这么被误判为显著的） |
| 2026-09-29 | 检索侧主指标用证据召回，准确率指标用 EM + F1 | 证据召回在所有解码设定下都稳定显著；EM 对答案形式和采样都敏感 |
| 2026-09-29 | 暂不扩大 validation（仍 200 题） | 3 个 seed × 200 题已能看出波动；扩大要从 train 再抽，先看 V2 的效应量再定。V2 的差距若在 3～5 个点，再考虑扩到 500 |
| 2026-09-29 | 错题每题只归一个主因，按"最小修复"顺序判断（越便宜的修复越先判） | 多标签会让各类题数加起来不等于错题数，回答不了"先修哪类"；先排除便宜的原因，剩下的才算检索器能力问题 |
| 2026-09-29 | "证据够了" = 金标全找到，或桥接题答案原文已出现在检索结果里（纯数字除外）；比较题不看答案原文 | 答案在文本里时停下没错，该修的是阅读；比较题答案是问题里的实体名，出现在文本里说明不了什么 |
| 2026-09-29 | 错题分类结果写进运行目录（`error_taxonomy.csv`），不单独建目录 | 派生文件和它的来源放在一起；运行目录不进 git |
| 2026-09-29 | 不放宽重复查询规则（不做词干化去重）；不加原地打转提醒 | 量化后收益分别只有 1 次、4 题；新增控制逻辑会改变轨迹、增加提示词变量，收益说不清就不加 |
| 2026-09-29 | 解析失败的报错试过改用 user 消息发回，一天内回退 | Agent 无稳定收益（0.5 个点全部来自解析放宽），Direct 重试变弱（同一笔误连着两轮重复）；一次只改一个变量，才分得清收益来自哪 |
| 2026-09-29 | 解析规则改为"取 `<tool_call>` 后第一个完整 JSON 对象，后面的内容丢弃并记录"（闭合、未闭合都一样），推翻 3.3 的"JSON 后面还有文字就报错" | 和停止词语义一致：写了 `</tool_call>` 时后面的一切本来就被截掉、只执行第一个动作；只因漏了结尾标签就判没作答，口径不一致。validation 强制轮 4 次都是这种（乱码 `hendrix`、半个新调用） |
| 2026-09-29 | 引号修复只在 `{"name": 工具, "arguments": {参数: "值"}}` 单字符串参数骨架下做，值取到最后一个 `"}}`；值里像有第二个参数就不修；修完照常走结构层校验 | 两个工具都只有一个字符串参数，边界无歧义；模型不会转义引号，同一题重试 4 次都一样，靠重试救不回。`})` 这类笔误不修，重试能改对 |
| 2026-09-29 | 解析失败的报错用 user 消息发回；合法调用的返回（检索结果、重复查询、超预算）仍用 tool 消息 | 解析失败 = 没有合法工具调用，就没有"工具返回"；包进 `<tool_response>` 时纯文字作答后 16/26 轮去搜，疑似把报错当成检索结果。效果待 validation 重跑验证 |
| 2026-10-02 | 嵌入模型用 e5-base-v2 | Search-R1 的向量检索用的就是它，V3 接 Search-R1 时口径一致；英文；base 尺寸 768 维，50 万段 4 分钟编完 |
| 2026-10-02 | 段落编码 = `passage: 标题\n正文`，截断 512 | 和 BM25 一样带标题（标题就是实体名）；512 只截掉 0.05% 的段落，256 会截 2.4%；按长度排序分批，padding 少，不怎么变慢 |
| 2026-10-02 | 向量索引用 FAISS `IndexFlatIP`（精确检索），不用 HNSW / IVF | 50 万段 CPU 暴力搜约 27ms，可以接受；精确检索没有近似召回损失，和 BM25 比较时少一个变量。近似索引留到 fullwiki（520 万段）再评估 |
| 2026-10-02 | faiss-cpu 装进 `dsr1` | 项目代码的环境；`verl_env` 不装新包；CPU 版不和 vLLM 抢显存 |
| 2026-10-02 | 段落 GPU fp16 编码；查询 CPU fp32 编码 | 段落只编一次，fp16 快，和 fp32 的余弦 ≥ 0.9999995；查询单条 10～20ms，放 CPU 不占显存、不和 vLLM 抢，fp32 换机器重跑可逐位复现 |
| 2026-10-02 | `/search` 加 `method`，默认 bm25；没加载的 method 返回 400，`run_eval` 开跑前检查服务有没有这一路 | V1 配置不传 method，原样可复现；不悄悄退回 BM25，否则对比会失真 |
| 2026-10-02 | 融合用等权 RRF（k=60，每路取 20），同分按 doc_id；融合前检查两个索引的 doc_id 顺序完全一致 | 两路分数量纲不同，RRF 不需要校准；k=60 沿用 Cormack 2009；doc_id 是标题哈希，同分时不偏向哪一路；ID 对不上就报错，不融合错的东西 |
| 2026-10-02 | 检索对比脚本只跑 validation / debug（拒绝 test）；输出 `per_question.jsonl` 代替 `trajectories.jsonl` / `errors.csv` | 阈值和结论只在 validation 上得；纯检索分析没有作答轨迹 |
| 2026-10-02 | 重排模型用 bge-reranker-base，放 GPU fp16，候选池 20 → 输出 top_k | CPU 上一次 0.9s，太慢；GPU 上 18ms、约 0.7GB 显存，和 vLLM（约 19.5GB）共存没问题；Agent 单并发串行，重排和生成不会同时抢 GPU。候选池 20 沿用计划起步值 |
| 2026-10-02 | 重排输入 = （查询, `标题\n正文`），超长只截段落（`truncation="only_second"`）；同分按 doc_id | 和 BM25 / 向量检索一样带标题；查询要完整保留；fp16 下几乎重复的段落可能同分，要固定顺序 |
| 2026-10-02 | `rerank` 是 `/search` 上和 `method` 独立的开关；没加载重排模型时 `rerank=true` 返回 400 | V2 策略里"选检索器"和"要不要重排"是两个独立动作；不悄悄跳过重排，否则对比失真 |
| 2026-10-02 | HotpotQA 的 B3 = **Static RAG + Dense + 重排**（只搜一次），不是计划里的 Always Hybrid + Rerank，也不是多轮 | Dense + 重排 EM 不比 Hybrid + 重排差（0/4 显著）、F1 更好（3/4 显著）、少跑一路 BM25；Agent 版 EM 0/4 显著、成本 4～5 倍。强基线取实测最强的、最便宜的那个 |
| 2026-10-02 | 改写对照用**固定检索计划**（1 次 / 2 次），不让模型决定搜几次 | Day 10 只比"查询写得好不好"；Agent 循环里查询、停止、作答缠在一起，改一处 77% 轨迹都变（Day 6）。固定计划下两跳的组每题都恰好 2 次，预算相同 |
| 2026-10-02 | 证据条件改写的上下文用 **Agent 的对话格式**（自己发过的 search 调用 + `<tool_response>` 工具返回），不用"用户消息里给段落" | debug 实测：后者 50 题 0 次用上证据实体；前者 22%。不训练的模型在工具调用这类格式上只认微调时见过的样子（Day 3 已经遇到过一次） |
| 2026-10-02 | 改写解析失败退回原问题，**不重试**；退回的检索照样执行计费 | 重试次数会让各组成本不同，固定流程的预算必须相同；fallback 率本身还是有用的信号（见"当前位置"） |
| 2026-10-02 | 按需升级的探测**复用证据改写提示词**，不另写一份 | `gate=always` 就逐字等于两跳证据改写、`gate=never` 逐字等于 B3，门控策略可以用已有的两次运行离线精确推算（在线验证 200/200 一致），调门槛不用反复跑模型 |
| 2026-10-02 | 分差门槛**跨数据集选**：HotpotQA 用 2Wiki 上选的 4.19，2Wiki 用 HotpotQA 上选的 5.69 | 在同一批题上选门槛再报结果会偏乐观；两个数据集互为样本外，报出来的就是"换一个数据集照搬门槛"的真实效果 |
| 2026-10-02 | 探测时模型原样重搜原问题 → 不执行，记 `duplicate`，用第一跳证据作答；格式错误 → 记 `format_error`，不重试 | 重搜原问题的结果和第一跳相同，白花一次检索；不重试是为了成本口径和固定流程一致 |
| 2026-10-02 | cascade 的 `context.latency_ms` 只算第一次检索，第二次检索和探测的耗时记在探测那一步 | `run_eval` 把 context 耗时和每步耗时相加算端到端延迟，分开记不会重复计算 |
| 2026-10-02 | 2Wiki 的 B3 也定为 Static RAG + Dense + 重排 | Hybrid + 重排 EM +0.2 [−1.1, +1.6] 打平；两个数据集用同一个检索栈，跨数据集比较少一个变量 |
| 2026-10-02 | 静态拆解做成单独的 `decompose` 工具（参数是列表），不让模型一轮写多个 search | 生成在第一个 `</tool_call>` 就停（每轮一个动作的规则），一轮写不出多个调用；单独的工具也让"拆解"和"改写"在日志里分得开 |
| 2026-10-02 | 2Wiki 的划分从 train 里分层抽（每类同样多），官方 dev 留作以后的 test | 和 HotpotQA 同一套规则；推理题只占 2.6%，不分层几乎抽不到；按题型报告，不报混合平均 |
| 2026-10-02 | V2 主线从"自适应选检索器 / 重排"调整为"**按需升级**：默认 B3，需要时升级成多轮" | 选检索器上限 3～5 点、重排已经默认要做（16ms 换 +10 EM）；能省的成本在"多轮 vs 单次"上（4～5 倍），而逐题取较好的上限有 +8～+13 |
| 2026-10-03 | **B3 不换作答格式**（保持"证据放用户消息 + 只能作答"） | 2×2 诊断里四组只有"工具返回 × Agent"高，+4.0 [−1, +9] 不显著，而且全来自模型自己选择直接作答的题；想搜被强制作答的题反而 −3。换格式会动到 V1 以来所有基线，收益又不显著，不值得 |
| 2026-10-03 | 格式诊断复用已有运行轨迹里的检索结果，不重新检索 | 四组看到的段落逐字相同，差别只剩格式；不用起检索服务。复用是否走样用"B3 写法组和 B3 运行逐字一致"来检查（200/200） |
| 2026-09-25 | 进度靠 `CLAUDE.md` + `docs/PROGRESS.md` + `docs/LEARNING_NOTES.md` 保存，并定期 push 到 GitHub | 对话记录会被压缩或清理；仓库在数据盘上，实例释放即丢失 |

## 已有资产（上一次 Search-R1 复现留下，位于系统盘 `/root/Search-R1/`）

- `data/nq_hotpotqa/{train,test}.parquet`（632M）：Search-R1 官方处理后的 NQ + HotpotQA 数据。
- `data/bm25_index/`（2.2G）：wiki-18 的 BM25（Lucene）索引；对应语料 `wiki-18.jsonl` 原先放在数据盘，已丢失。
- `eval/data/corpus/hotpotqa_corpus.jsonl`：41,897 个段落；`eval/data/eval/hotpotqa_dev.jsonl`：500 道题。
- `eval/src/`：上次写的 base / rag / agent 评测和 GRPO 代码。

## 待办 / 开放问题

- **两跳证据改写的输入 token 是 B3 的 2.3 倍**：升级成本的大头是第二次作答时上下文里多出来的 3 段，而不是改写调用本身。Day 11 做按需升级时，"拒绝再搜"的题直接用 B3 的答案，不再搜第二次，省下的就是这部分
- 多看段落伤答案（2Wiki 比较题前 6 条 EM −11.5，大多是 yes/no 答成 false）：按需升级时也要注意，升级只应该发生在"想再搜"的题上
- **按需升级的信号（Day 11～12 主线）**：Day 10 已经找到两个可用的信号：① **改写步拒绝再搜**（有第一跳证据后让它写第二个查询，它直接给答案 → 56% 的题、其中 90/111 第一跳确实已全齐、这些题的 EM 和 B3 一模一样）→ 等价于零成本的"要不要继续"判据；② 答案一致性（见下）。两者可以组合：先看模型愿不愿意再搜，愿意再搜但证据仍缺时才用采样一致性判断
- 旧的候选信号：① 作答步答案一致性（vLLM `n=K` 采样或 logprobs，Day 6 已证明一致性是准的置信度信号）；② 证据侧信号（重排分数的绝对值 / 第 1 和第 3 名的分差、问题里的实体是否都在前 3 条里出现）。Agent 稳定赢的 16 题里 10 题 B3 已证据全齐 → 只看证据侧会漏掉一大半，要和答案侧信号组合
- 升级时 Agent 从哪里开始：从零开始（现在的 Agent）还是带着 B3 的检索结果接着搜（"原问题保底召回"的变体，见下）。后者更省、也避免"Agent 改写的查询不如原问题"
- 2Wiki 的 B3 端到端：离线 Hybrid + 重排比 Dense + 重排高 0.5 个点（显著但很小），端到端跑 Static RAG 两种都试；2Wiki 的组合题候选池@20 只有 0.57，重排救不了，要靠拆解
- Agent + Dense + 重排的证据召回（0.807）低于 B3（0.823）：Agent 第一次查询是改写过的，比原问题差（Day 3 现象在强检索下仍在）→ Day 10 的"原问题 vs 改写"对照要在 Dense + 重排上重做

- **B3 强固定基线的定义（Day 9）**：HotpotQA 上 Dense 最强、2Wiki 上 Hybrid 最强 → 每个数据集都同时跑 Always Dense + Rerank 和 Always Hybrid + Rerank，取实测更强者，并在表里注明是怎么挑的。强基线不能刻意做弱
- **"选检索器"降级为一个小消融**，不作为主线卖点：路由上限只有 3～5 个点、真实路由器更少，如实报告。主线的成本论证改成"重排 / 改写要不要做"和"何时停"
- **主线跟着数据走：查询构造 ≫ 检索器选择**（整句提问 0.588 → 实体 + 关系 0.968）。Day 10 的三组改写对照（原问题 / 静态改写 / 证据条件改写）是重点，2Wiki 的组合题可以直接验证问题拆解（DECOMPOSE）的收益
- 2Wiki 上理想子查询里 Hybrid < Dense（−5.8，显著）：BM25 弱 20 个点，融合被拖累。→ 融合的两路实力差距大时要降权或先做分数校准，这条经验写进 Day 10 的结论
- 2Wiki 的 2000 题分析集用均匀按题型分层（每类 500），和真实分布不同（inference 只占 2.6%）；只按题型报告，不报一个混合平均。要报混合数就用自然分布重抽
- 2Wiki 目前只有 analysis 划分；若以后当正式评测集，要另抽 validation / test 且与分析集不重叠（脚本里已写明）
- 加权 RRF（Dense 权重更高）或调 k：只在 validation / analysis 上调；调多了会过拟合
- 换检索器重跑 Static RAG / Agent（真反事实）：Agent 查询重放只说明"同样的查询换检索器"，换了检索器 Agent 会写出不同的查询
- e5 预训练数据里有维基的（标题, 段落）对，和 HotpotQA 段落形式一致 → 这里 Dense 的优势可能偏大；商品搜索（Day 13）型号词、品牌词多，要重新验证，不能直接外推
- BM25 在 Agent 短查询上比原问题慢（27ms vs 8ms），原因没查（bm25s 实现细节）；不影响结论，成本分析时按实测报

- ~~观察结果用 `role="tool"` 拼回，要核对 Qwen2.5 对话模板的实际渲染~~ → 2026-09-28 已核对：包进 `<tool_response>`，方式正确。
- 3.5 看轨迹时专门看查询改写：每个 Agent 查询的金标排名 vs 原问题的金标排名，统计改写是得是失（例：`Mary Gordon birth year` 让金标 1 → 2，`H. L. Mencken birth year` 让 5 → 10）。
- 用了停止词后 `num_tool_calls` 永远 ≤ 1，"模型想并行调用"的信息丢了。需要时可在 debug 集上不设停止词单独统计。
- ~~服务整体挂掉时应提前中止~~ → 3.3 已加（连续 5 题）。
- ~~`config.yaml` 补模型名、解码参数、版本；真实客户端异常映射到可重试类型~~ → 3.3 已完成。
- ~~3.5 修格式报错提示~~ → 已按可用工具生成（validation 基线已用新提示）。
- ~~3.5 看比较题 Agent 是否不如 Static RAG~~ → validation 上 −2.4 个点、区间 [−19.5, +14.6]，41 题说明不了；方向和 debug 一致。
- ~~Day 4 修两类格式失误~~ → 4.1 已修（见 Day 4 子步骤）。原文：validation 上 Agent 8 题因此没作答：① JSON 写完整后跟乱码 token（考虑用 `json.JSONDecoder.raw_decode` 只取第一个完整 JSON 对象，放宽要窄、要打标记）；② 答案里有未转义的双引号。
- V2 消融候选（Day 10～11）：**原问题保底召回**——第一次检索固定用原问题，之后交给 Agent。两种变体：计入搜索预算（总上限不变，考验名额挤占）/ 不计入（多给一次，要按成本折算）。和 Agent、Static RAG 在同一预算曲线上比较。离线依据见"当前位置"的思考题回顾。
- V2 停止 / 路由信号候选：**作答步答案一致性**（`n=K` 采样或 logprobs）× **证据侧信号**（本轮新文档数、证据是否覆盖问题里的实体）：不确定 + 证据缺 → 继续搜；不确定 + 证据够 → 换动作（重排 / 重读），不再搜；确定 → 停。依据见 Day 6 思考题回顾
- V2 前缀重放的第一个用途：**过早停止 29 题**——重放到停止那一轮后强制多搜一次，看能不能找到缺的证据，确认停止策略的提升上限
- V2 查询改写候选：**比较题按实体拆开搜**（Robert Palmer、George Stevens 两题都是两个人名写进同一个查询，一个总排不进前 3）
- V2 实验工具：**前缀重放**——给定轨迹和截断轮次，原样重放前缀，再注入新规则的动作（如强制作答），得到精确反事实。评估停止 / 路由规则时先用它，再跑完整消融
- V2 路由信号候选：**0 新文档 → 换路**（拆子问题 / 换检索器），不是停止（离线依据见思考题回顾）
- V2 消融候选：**告知剩余搜索次数**（观察里写 "searches left: N"）。4.3 里 21 题预算用完还想搜，比较题出现 3 次全花在一个实体上的情况；Day 3 决定 V1 不告诉（锚定效应、干扰"会不会自己停"的测量），V2 作为零成本对照
- 比较题 41 题太少，方法间差异的置信区间 ±17 个点。V2 做按题型路由时要么扩大 validation 里的比较题，要么用 test 以外的 train 题做分析集。
- 3.5 / validation 上调答案规范：Oracle 下比较题仍错的 7 题里，多数是答案形式问题（问"谁"却答了年份、答 "true" 而不是 "yes"、把标题 "Firehose (band)" 原样抄下来），真正推理错约 2～3 题。可以在提示词里加答案格式说明，只在 validation 上调。
- 错因拆分（`20260928-212118`，debug 50 题）：答错 36 题中 26 题证据没找齐、10 题证据齐了仍答错；这 10 题约 5 题是 EM 口径（意思对）、1 题标签问题、约 4 题真读错 → 当前瓶颈主要在检索。
- 候选诊断：**Oracle context**（直接给金标段落，衡量纯阅读能力上限）。它要把 labels 里的金标段落放进 prompt，和"标签只给评测器"的防泄漏规则冲突 → 只能作为明确标注的诊断上限、只在 validation/debug 上跑、不作为方法参与比较；**实现前先和用户确认这个例外**。
- 候选消融：模型尺寸（Qwen2.5-3B vs 7B，7B bf16 权重约 15GB，4090 能放下；14B 需要量化）。放在 Oracle 设定下比较才能测纯阅读能力；建议 Day 5～6 基线表定下来后再做。
- 答案常写成句子或带多余修饰（"Atlanta" vs "Atlanta, Georgia"、答比较题时写年份），EM 偏严 → 3.4 补 Token-F1。
- Direct 基线怎么配：`max_search_calls=0` 时提示词仍说可以搜，模型想搜会浪费一轮并被记成 forced_answer。Day 3 决定是用 `max_turns=1`，还是给 Direct 单独一份不带工具的提示词。
- ~~轨迹还没记录观察的 token 数~~ → 每轮记了服务端的 `prompt_tokens`，观察 token 由相邻两轮差值得到（见决策记录）。
- `/root/Search-R1`（`verl_env` 可编辑安装）与子模块 `third_party/Search-R1` 的关系，到 V3 再决定。
