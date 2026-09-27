"""模型接口：循环只依赖"给消息、返回文字"这一个能力，换真模型、假模型都不用改循环。"""
from __future__ import annotations

import json
import re
import time
from typing import Protocol


class LLM(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


class ScriptedLLM:
    """按剧本依次返回预设输出的假模型。结果完全确定，测试失败时能确定是循环的问题，而不是模型的随机性。"""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.seen_messages: list[list[dict]] = []  # 每次调用时看到的上下文，测试里用来检查报错有没有拼回去

    def generate(self, messages: list[dict]) -> str:
        self.seen_messages.append(list(messages))
        if not self._outputs:
            raise RuntimeError("ScriptedLLM: script exhausted")
        return self._outputs.pop(0)


class SearchThenTitleLLM:
    """规则假模型：第一轮拿原问题去搜，第二轮把第一条结果的标题当答案。
    只用来在没有真模型时跑通整条实验流程，它的准确率没有任何意义。"""

    def generate(self, messages: list[dict]) -> str:
        observations = [m["content"] for m in messages if m["role"] == "tool"]
        if not observations:
            question = messages[1]["content"].removeprefix("Question: ")
            return _tool_call("search", query=question)
        top = re.match(r"\[1\] \(Title: (.*?)\)", observations[-1])
        return _tool_call("final_answer", answer=top.group(1) if top else "unknown")


def _tool_call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


class RetryingLLM:
    """只重试临时性错误（超时、连接断开），间隔指数增长；重试完仍失败就抛给循环，记成 stop_reason=error。
    上下文超长这类错误重试也没用，不在 retry_on 里，直接抛出。"""

    def __init__(self, llm: LLM, max_retries: int = 3, base_delay_s: float = 1.0,
                 retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)):
        self.llm, self.max_retries, self.base_delay_s, self.retry_on = llm, max_retries, base_delay_s, retry_on

    def generate(self, messages: list[dict]) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.generate(messages)
            except self.retry_on:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.base_delay_s * 2 ** attempt)
