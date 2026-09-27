"""模型接口：循环只依赖"给消息、返回文字"这一个能力，换真模型、假模型都不用改循环。"""
from __future__ import annotations

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
