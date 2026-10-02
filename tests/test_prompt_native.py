"""提示词要和模型目录里的对话模板逐字一致，否则就不是"模型微调时见过的写法"了。

需要模型目录里的 tokenizer_config.json；模型不在时跳过，不影响其他测试。
运行：pytest tests/ 或 python -m tests.test_prompt_native
"""
import os

import pytest

from agent.native_tools import DECOMPOSE_TOOL, FINAL_ANSWER_TOOL, SEARCH_TOOL, native_system_prompt
from agent.prompts import (ANSWER_ONLY_SYSTEM_PROMPT, ANSWER_ONLY_TASK, DECOMPOSE_SYSTEM_PROMPT, DECOMPOSE_TASK,
                           EVIDENCE_REWRITE_SYSTEM_PROMPT, EVIDENCE_REWRITE_TASK, STATIC_REWRITE_SYSTEM_PROMPT,
                           STATIC_REWRITE_TASK, SYSTEM_PROMPT, TASK, TOOLS)

MODEL_DIR = os.environ.get("MODEL_DIR", "/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct")


def _from_chat_template(task: str, tools: list[dict]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rendered = tok.apply_chat_template([{"role": "system", "content": task}, {"role": "user", "content": "Q"}],
                                       tools=tools, tokenize=False)
    return rendered.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="模型目录不存在，跳过逐字比对")
@pytest.mark.parametrize("prompt, task, tools", [
    (SYSTEM_PROMPT, TASK, TOOLS),
    (ANSWER_ONLY_SYSTEM_PROMPT, ANSWER_ONLY_TASK, [FINAL_ANSWER_TOOL]),  # B0 / B1 / Oracle：同一种写法，少了 search
    (STATIC_REWRITE_SYSTEM_PROMPT, STATIC_REWRITE_TASK, [SEARCH_TOOL]),    # Day 10 改写
    (EVIDENCE_REWRITE_SYSTEM_PROMPT, EVIDENCE_REWRITE_TASK, [SEARCH_TOOL]),
    (DECOMPOSE_SYSTEM_PROMPT, DECOMPOSE_TASK, [DECOMPOSE_TOOL]),           # Day 10.4 拆解：参数是列表
], ids=["agent", "answer_only", "static_rewrite", "evidence_rewrite", "decompose"])
def test_matches_chat_template(prompt, task, tools):
    assert prompt == _from_chat_template(task, tools)


def test_agent_prompt_is_what_the_loop_sends():
    assert SYSTEM_PROMPT == native_system_prompt(TASK, TOOLS)
    assert '"name": "search"' in SYSTEM_PROMPT and '"name": "final_answer"' in SYSTEM_PROMPT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
