"""Agent 的核心数据结构：动作（Action）、观察（Observation）、预算（Budget）。

这里只定义"数据长什么样"，不关心模型用哪种文字格式写出动作；
文字 → Action 的解析放在 agent/parser.py，换格式时不用改这里。
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # extra="forbid"：模型多写了未定义的参数就报错，而不是悄悄忽略
    # str_strip_whitespace：先去掉首尾空白再校验，"   " 会被当成空字符串拦下
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------- 动作 ----------------

class SearchArgs(_Strict):
    # 只有 query；top_k 由 Budget 固定，不交给模型，保证和 Static RAG 拿到同样多的证据
    query: str = Field(min_length=1, max_length=512)


class FinalAnswerArgs(_Strict):
    answer: str = Field(min_length=1)


class SearchAction(_Strict):
    name: Literal["search"]
    arguments: SearchArgs


class FinalAnswerAction(_Strict):
    name: Literal["final_answer"]
    arguments: FinalAnswerArgs


# 按 name 字段选择用哪个结构校验 arguments（pydantic 的"可辨识联合"）
Action = Annotated[Union[SearchAction, FinalAnswerAction], Field(discriminator="name")]
# 解析器先用它区分"未知工具"和"参数错误"；新增工具时两处都要改
TOOL_NAMES = ("search", "final_answer")


# ---------------- 观察 ----------------

class ErrorCode(str, Enum):
    # 语法层：文字里解析不出一个动作
    NO_ACTION = "no_action"
    INVALID_JSON = "invalid_json"
    # 结构层：解析出来了，但不符合 schema
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    # 策略层：动作合法，但不该执行
    DUPLICATE_QUERY = "duplicate_query"
    BUDGET_EXCEEDED = "budget_exceeded"
    # 执行层：工具本身出错（超时、服务不可用）
    TOOL_ERROR = "tool_error"


class Doc(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    source: str  # 来自哪一路检索（bm25 / dense / mock），多路召回融合时要用


class Observation(BaseModel):
    """一个动作执行后的结果：拼回上下文给模型看，同时写进轨迹日志。

    "检索成功但没结果"是 ok=True、docs=[]，不是错误——
    就像召回为空和召回服务超时是两种不同的告警，badcase 分析时要分开。
    """
    ok: bool
    docs: list[Doc] = Field(default_factory=list)
    error_code: Optional[ErrorCode] = None
    message: str = ""  # 给模型看的说明；出错时要告诉它怎么改


# ---------------- 预算 ----------------

class Budget(BaseModel):
    """一道题的资源上限（配置）。所有方法共用同一个定义，比较才公平。默认值只是占位，要在 validation 上调。"""
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_turns: int = Field(default=5, ge=1)         # 模型最多生成几轮，写错格式的轮次也算
    max_search_calls: int = Field(default=3, ge=0)  # 最多真正执行几次检索
    top_k: int = Field(default=3, ge=1)             # 每次检索返回几条


class BudgetState(BaseModel):
    """一道题已经用掉的资源（状态）。和 Budget 分开：Budget 进 config.yaml，BudgetState 进轨迹。"""
    turns_used: int = 0
    search_attempts: int = 0    # 模型想搜的次数，含被拦下的重复查询和超预算搜索
    search_calls_used: int = 0  # 实际执行的次数，成本按它算

    def can_take_turn(self, budget: Budget) -> bool:
        return self.turns_used < budget.max_turns

    def can_search(self, budget: Budget) -> bool:
        return self.search_calls_used < budget.max_search_calls


# ---------------- 轨迹 ----------------

class StopReason(str, Enum):
    ANSWERED = "answered"            # 模型自己决定作答
    FORCED_ANSWER = "forced_answer"  # 被预算截停后作答：最后一轮，或搜索次数用完后仍想搜
    NO_ANSWER = "no_answer"          # 轮数用完仍没给出合法答案
    ERROR = "error"                  # 模型服务重试后仍失败；计 0 分，但留在准确率的分母里


class Step(BaseModel):
    turn: int                                # 从 1 开始
    generated: str                           # 模型本轮原始输出，离线算宽松诊断指标要用
    forced: bool                             # 本轮是否只许作答
    num_tool_calls: int
    unclosed_tool_call: bool = False         # 缺 </tool_call> 但 JSON 完整，按规则补上了（见 agent/parser.py）
    ignored_suffix: str = ""                 # JSON 对象后面被丢掉的内容（见 agent/parser.py 的放宽规则）
    repaired_quotes: bool = False            # 参数值里有没转义的引号，按单参数骨架取了值
    action: Optional[Action] = None          # 解析失败时为空
    observation: Optional[Observation] = None  # 作答那一轮没有观察
    num_new_docs: int = 0                    # 本轮检索结果里之前没见过的文档数，衡量这次搜索的边际收益
    llm_latency_ms: float                    # 含重试等待，是用户实际等的时间
    tool_latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None      # 本轮输入 token（整个上下文都要重新算，是多轮的主要成本）；假模型为 None
    completion_tokens: Optional[int] = None  # 本轮输出 token
    finish_reason: Optional[str] = None      # length = 写到 max_tokens 被截断，格式错误时先看它
    budget_state: BudgetState                # 本轮结束后的用量快照


class SearchRecord(BaseModel):
    """固定流程里的一次检索（Day 10 改写对照）：用什么查询、查询怎么来的、搜到了什么、花了多少。"""
    kind: Literal["original", "static_rewrite", "evidence_rewrite"]
    query: str
    generated: Optional[str] = None   # 改写那次模型的原始输出；原问题检索为空
    fallback: bool = False            # 改写没解析出合法的 search 调用，退回用原问题检索
    observation: Observation          # 这次检索的原始结果（名次是这次检索里的名次）
    num_new_docs: int = 0             # 之前几次检索没见过的段落数
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Context(BaseModel):
    """答题前就放进提示词的证据（Static RAG / Oracle / 改写流程），不是模型在作答循环里自己搜来的。"""
    source: Literal["retrieval", "oracle"]
    query: Optional[str] = None  # Static RAG 拿原问题检索；Oracle 没有查询，也不算检索成本
    observation: Observation     # 模型看到的证据；多次检索时是去重合并后的结果，名次按展示顺序重排
    latency_ms: float = 0.0      # 含改写的模型调用
    searches: list[SearchRecord] = Field(default_factory=list)  # 多次检索 / 改写的明细；Static RAG 为空

    def num_searches(self) -> int:
        return len(self.searches) if self.searches else int(self.query is not None)


class Trajectory(BaseModel):
    qid: str  # 只用于日志对齐；不会传给检索工具
    question: str
    method: str = "agent"               # agent / direct / static_rag / oracle，见 agent/methods.py
    context: Optional[Context] = None
    budget: Budget
    steps: list[Step]
    budget_state: BudgetState  # 最终用量；模型第一轮就失败时 steps 为空，统计成本靠它
    final_answer: Optional[str] = None
    stop_reason: StopReason
    error: Optional[str] = None  # stop_reason=error 时记录异常类型和信息
