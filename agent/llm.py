"""模型接口：循环只依赖"给消息、返回一次生成结果"这一个能力，换真模型、假模型都不用改循环。

真模型走 vLLM 的 OpenAI 兼容接口（chat/completions）：
- 服务端套对话模板（以模型目录里的 tokenizer_config.json 为准，只有一份），客户端不自己拼
- 返回的 usage 直接给出输入输出 token 数，一个请求 = 轨迹里的一轮，成本可以按轮记

错误分两类（决策记录 2026-09-27）：
- 临时性（超时、连不上、5xx）：RetryingLLM 指数退避重试
- 永久性（上下文超长、模型名写错等 4xx）：重试也没用，直接抛出，循环记 stop_reason=error
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx
import openai


class LLMTimeoutError(TimeoutError):
    """模型服务超时。继承内置 TimeoutError，RetryingLLM 的默认规则就认得它。"""


class LLMConnectionError(ConnectionError):
    """连不上模型服务，或服务端 5xx。"""


class LLMPermanentError(RuntimeError):
    """重试也没用的错误。"""


@dataclass
class Generation:
    """一次调用的结果：文字 + 成本。循环只用 text 做决策，其余写进轨迹。"""
    text: str
    prompt_tokens: Optional[int] = None      # 假模型没有 token 数，记 None 而不是 0，避免被当成真实成本
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None      # stop = 自己停或命中停止词；length = 写到 max_tokens 被截断
    stop_str: Optional[str] = None           # 命中的停止词，没命中为 None


class LLM(Protocol):
    def generate(self, messages: list[dict]) -> Generation: ...


class ScriptedLLM:
    """按剧本依次返回预设输出的假模型。结果完全确定，测试失败时能确定是循环的问题，而不是模型的随机性。"""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.seen_messages: list[list[dict]] = []  # 每次调用时看到的上下文，测试里用来检查报错有没有拼回去

    def generate(self, messages: list[dict]) -> Generation:
        self.seen_messages.append(list(messages))
        if not self._outputs:
            raise RuntimeError("ScriptedLLM: script exhausted")
        return Generation(text=self._outputs.pop(0))


class SearchThenTitleLLM:
    """规则假模型：第一轮拿原问题去搜，第二轮把第一条结果的标题当答案。
    只用来在没有真模型时跑通整条实验流程，它的准确率没有任何意义。"""

    def generate(self, messages: list[dict]) -> Generation:
        observations = [m["content"] for m in messages if m["role"] == "tool"]
        if not observations:
            question = messages[1]["content"].removeprefix("Question: ")
            return Generation(text=_tool_call("search", query=question))
        top = re.match(r"\[1\] \(Title: (.*?)\)", observations[-1])
        return Generation(text=_tool_call("final_answer", answer=top.group(1) if top else "unknown"))


def _tool_call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


class VLLMClient:
    """vLLM 的 OpenAI 兼容接口。

    停止词：原生模板允许一个回复里写多个调用，提示词压不住，模型会"搜 A、搜 B、作答"一口气写完，
    作答时根本没看到结果。在第一个 </tool_call> 处截断，一轮只剩一个动作——规则靠代码强制
    （同 Search-R1 在 </search> 截断）。停止词本身保留在输出里，解析器按完整标签定位。
    """

    def __init__(self, base_url: str, model: str, *, temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 512, seed: Optional[int] = None, stop: Optional[list[str]] = None,
                 timeout_s: float = 60.0, http_client: Optional[httpx.Client] = None):
        self.model = model
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed}
        self.stop = list(stop or [])
        # max_retries=0：openai 库默认自己重试 2 次，会和 RetryingLLM 叠加，重试次数就说不清了
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s, max_retries=0,
                                    http_client=http_client)  # http_client 只给测试注入假服务

    def server_info(self) -> dict:
        """服务端实际加载的模型（路径、最大长度），记进 config.yaml：配置里写的名字不一定是实际跑的模型。"""
        # vLLM 0.6.3 没有 /v1/models/{id}，只能从列表里找
        m = next((m for m in self.client.models.list() if m.id == self.model), None)
        if m is None:
            raise LLMPermanentError(f"model service does not serve {self.model!r}")
        # 软件版本不从 /version 取：vLLM 0.6.3 缺版本文件，那里只返回 "dev"（runner 另外读服务端环境的包版本）
        return {"id": m.id, "root": getattr(m, "root", None), "max_model_len": getattr(m, "max_model_len", None)}

    def warmup(self) -> float:
        """首个请求要初始化 CUDA graph 等，约 6s；正式计时前先发一个，否则 P95 延迟量的是初始化。返回耗时（ms）。"""
        t0 = time.perf_counter()
        self._create([{"role": "user", "content": "hi"}], max_tokens=1, stop=[])
        return (time.perf_counter() - t0) * 1000

    def generate(self, messages: list[dict]) -> Generation:
        return self._create(messages, **self.params, stop=self.stop)

    def _create(self, messages: list[dict], stop: list[str], **params) -> Generation:
        # 值为 None 的参数不发：openai 库会把 None 原样发成 null，而不是省略
        params = {k: v for k, v in params.items() if v is not None}
        if stop:
            params.update(stop=stop, extra_body={"include_stop_str_in_output": True})  # 后者是 vLLM 专有参数
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, **params)
        except openai.APITimeoutError as e:  # 要先于 APIConnectionError：前者是后者的子类
            raise LLMTimeoutError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMConnectionError(f"HTTP {e.status_code}: {e.message}") from e
            raise LLMPermanentError(f"HTTP {e.status_code}: {e.message}") from e
        choice = r.choices[0]
        return Generation(text=choice.message.content or "", prompt_tokens=r.usage.prompt_tokens,
                          completion_tokens=r.usage.completion_tokens, finish_reason=choice.finish_reason,
                          stop_str=getattr(choice, "stop_reason", None))  # vLLM 把命中的停止词放在 stop_reason


class RetryingLLM:
    """只重试临时性错误（超时、连接断开），间隔指数增长；重试完仍失败就抛给循环，记成 stop_reason=error。
    上下文超长这类错误重试也没用，不在 retry_on 里，直接抛出。"""

    def __init__(self, llm: LLM, max_retries: int = 3, base_delay_s: float = 1.0,
                 retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)):
        self.llm, self.max_retries, self.base_delay_s, self.retry_on = llm, max_retries, base_delay_s, retry_on
        self.num_retries = 0  # 整次运行累计重试了几次，记进 metrics，是服务稳定性的信号

    def generate(self, messages: list[dict]) -> Generation:
        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.generate(messages)
            except self.retry_on:
                if attempt == self.max_retries:
                    raise
                self.num_retries += 1
                time.sleep(self.base_delay_s * 2 ** attempt)
