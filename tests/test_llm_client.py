"""真模型客户端：请求参数、结果解析、错误分类。用假的 HTTP 服务代替 vLLM，不需要 GPU。

错误分类决定"重试还是直接记 error"：分错了，要么服务挂了还在傻等，要么一次抖动就丢一道题。
运行：pytest tests/ 或 python -m tests.test_llm_client
"""
import json

import httpx
import pytest

from agent.llm import LLMConnectionError, LLMPermanentError, LLMTimeoutError, RetryingLLM, VLLMClient

OK_BODY = {
    "id": "chat-1", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "stop_reason": "</tool_call>",
                 "message": {"role": "assistant", "content": '<tool_call>{"name": "search"}</tool_call>'}}],
    "usage": {"prompt_tokens": 298, "completion_tokens": 29, "total_tokens": 327},
}


def client(handler, **kwargs) -> VLLMClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return VLLMClient("http://fake/v1", "m", http_client=http, **kwargs)


def test_request_carries_sampling_and_stop():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    gen = client(handler, temperature=0.0, max_tokens=256, seed=7, stop=["</tool_call>"]).generate(
        [{"role": "user", "content": "q"}])
    assert seen["stop"] == ["</tool_call>"] and seen["include_stop_str_in_output"] is True
    assert (seen["temperature"], seen["max_tokens"], seen["seed"]) == (0.0, 256, 7)
    assert gen.text.endswith("</tool_call>")  # 停止词留在输出里，解析器按完整标签定位
    assert (gen.prompt_tokens, gen.completion_tokens) == (298, 29)
    assert (gen.finish_reason, gen.stop_str) == ("stop", "</tool_call>")


def test_no_stop_means_no_stop_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    client(handler).generate([{"role": "user", "content": "q"}])
    assert "stop" not in seen and "include_stop_str_in_output" not in seen


@pytest.mark.parametrize("status, expected", [
    (500, LLMConnectionError),  # 服务端临时故障：重试
    (503, LLMConnectionError),
    (400, LLMPermanentError),   # 上下文超长等：重试也没用
    (404, LLMPermanentError),   # 模型名写错
])
def test_status_errors_are_classified(status, expected):
    c = client(lambda r: httpx.Response(status, json={"detail": "x"}))
    with pytest.raises(expected):
        c.generate([{"role": "user", "content": "q"}])


def test_timeout_and_connection_errors_are_retryable():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMTimeoutError):
        client(timeout).generate([])
    with pytest.raises(LLMConnectionError):
        client(refused).generate([])


def test_retrying_llm_retries_real_client_errors():
    # 前两次 503，第三次成功：RetryingLLM 要认得 VLLMClient 抛出的异常类型
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={}) if calls["n"] <= 2 else httpx.Response(200, json=OK_BODY)

    llm = RetryingLLM(client(handler), max_retries=3, base_delay_s=0)
    assert llm.generate([]).completion_tokens == 29
    assert calls["n"] == 3 and llm.num_retries == 2

    permanent = {"n": 0}

    def bad(request):
        permanent["n"] += 1
        return httpx.Response(400, json={})

    with pytest.raises(LLMPermanentError):
        RetryingLLM(client(bad), base_delay_s=0).generate([])
    assert permanent["n"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
