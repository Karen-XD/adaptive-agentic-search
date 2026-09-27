import json


def call(name: str, **arguments) -> str:
    """拼出一个模型本轮的 tool_call 输出。"""
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"
