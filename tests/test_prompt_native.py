"""提示词要和模型目录里的对话模板逐字一致，否则就不是"模型微调时见过的写法"了。

需要模型目录里的 tokenizer_config.json；模型不在时跳过，不影响其他测试。
运行：pytest tests/ 或 python -m tests.test_prompt_native
"""
import os

import pytest

from agent.native_tools import FINAL_ANSWER_TOOL, native_system_prompt
from agent.prompts import SYSTEM_PROMPT, TASK, TOOLS

MODEL_DIR = os.environ.get("MODEL_DIR", "/root/autodl-tmp/hf_models/Qwen2.5-3B-Instruct")


def _from_chat_template(task: str, tools: list[dict]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rendered = tok.apply_chat_template([{"role": "system", "content": task}, {"role": "user", "content": "Q"}],
                                       tools=tools, tokenize=False)
    return rendered.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="模型目录不存在，跳过逐字比对")
@pytest.mark.parametrize("tools", [TOOLS, [FINAL_ANSWER_TOOL]], ids=["agent", "answer_only"])
def test_matches_chat_template(tools):
    # answer_only 给 3.4 的 Direct 基线用：同一种写法，只是少了 search 工具
    assert native_system_prompt(TASK, tools) == _from_chat_template(TASK, tools)


def test_agent_prompt_is_what_the_loop_sends():
    assert SYSTEM_PROMPT == native_system_prompt(TASK, TOOLS)
    assert '"name": "search"' in SYSTEM_PROMPT and '"name": "final_answer"' in SYSTEM_PROMPT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
