"""Day 3.2 探针：同一批 debug 题，比较两种工具说明写法的格式有效率（2026-09-28 结果：占位 0/8，原生 8/8）。

用法（先起 vLLM 服务，见 docs/PROGRESS.md 恢复清单）：python -m experiments.day3_prompt_format_probe
不设停止词，这样能看到模型一轮想写几个调用；只读题目文件，不读答案。
3.3 会把 NATIVE 写法落进 agent/prompts.py，本脚本保留作为这个决策的证据。
"""
from __future__ import annotations

import json

from openai import OpenAI
from transformers import AutoTokenizer

from agent.parser import parse_action

# 占位提示词的原样副本（自包含，之后改 agent/prompts.py 不影响本脚本复现结论）
PLACEHOLDER = """Answer the question by searching a document collection.

In every turn, think briefly, then call exactly one tool:
- Search: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
- Answer: <tool_call>{"name": "final_answer", "arguments": {"answer": "..."}}</tool_call>

Search results are shown after each search. Search again if the information is not enough.
The final answer should be a short phrase such as a name, date or number, not a sentence."""

MODEL_DIR = "/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct"
TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description": "Search a document collection with keywords. Returns the top matching passages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search keywords"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Give the final answer and end. The answer should be a short phrase such as a name, date or number, not a sentence.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string", "description": "Short final answer"}},
                       "required": ["answer"]}}},
]
TASK = ("Answer the question by searching a document collection. Call exactly one function per turn. "
        "Search again with different keywords if the information is not enough; call final_answer when you can answer.")


def native_system_prompt() -> str:
    # 让对话模板生成 "# Tools / <tools>" 段，取出 system 部分：保证和模型微调时见过的写法逐字一致
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rendered = tok.apply_chat_template([{"role": "system", "content": TASK}, {"role": "user", "content": "Q"}],
                                       tools=TOOLS, tokenize=False)
    return rendered.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]


def main() -> None:
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    with open("data/hotpotqa/v1/questions/debug.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f][:8]
    for name, system in [("placeholder", PLACEHOLDER), ("native", native_system_prompt())]:
        valid = 0
        print(f"########## {name}")
        for q in questions:
            r = client.chat.completions.create(
                model="qwen2.5-3b-instruct", temperature=0, max_tokens=300, seed=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Question: {q['question']}"}])
            out = r.choices[0].message.content
            p = parse_action(out)
            valid += p.action is not None
            result = p.action.model_dump() if p.action else p.error.error_code.value
            print(f"[{q['qid']} in={r.usage.prompt_tokens} out={r.usage.completion_tokens} n_calls={p.num_tool_calls}] {result}")
        print(f"{name}: format valid {valid}/{len(questions)}\n")


if __name__ == "__main__":
    main()
