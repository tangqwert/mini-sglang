"""M10 验收：OpenAI 兼容 HTTP 服务。

跑法：pytest tests/test_server.py -x

两层：
  ① 用 Fake LLMEngine 测 HTTP 契约（请求校验、响应形状、usage、stop、stream 拒绝）
  ② 用真 gpt2 + LLMEngine 跑一条端到端，确认服务真的能生成
"""
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from engine.llm import ChatResult, format_chat, truncate_at_stop  # noqa: E402
from engine.server import create_app  # noqa: E402


class FakeLLM:
    """只实现 server 用到的接口，用来测 HTTP 层。"""

    def __init__(self):
        self.config = SimpleNamespace(model_name="fake-model")
        self.device = "cpu"
        self.use_graph = False
        self.radix = None
        self.calls = []

    def chat(self, messages, max_new_tokens=64, sampling=None, stop=None):
        self.calls.append({
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "sampling": sampling,
            "stop": stop,
        })
        return ChatResult(text="hello world", prompt_tokens=5,
                          completion_tokens=2, finish_reason="stop",
                          generated_token_ids=[1, 2])


@pytest.fixture()
def client():
    llm = FakeLLM()
    return TestClient(create_app(llm)), llm


class TestHttpContract:
    def test_health_and_models(self, client):
        c, _ = client
        h = c.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"
        assert h.json()["model"] == "fake-model"

        m = c.get("/v1/models")
        assert m.status_code == 200
        assert m.json()["data"][0]["id"] == "fake-model"

    def test_chat_completion_shape_and_params(self, client):
        c, llm = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 5,
            "seed": 42,
            "stop": ["\n"],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "fake-model"
        assert body["choices"][0]["message"]["content"] == "hello world"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2,
                                 "total_tokens": 7}
        assert body["id"].startswith("chatcmpl-")

        call = llm.calls[0]
        assert call["max_new_tokens"] == 8
        assert call["sampling"].temperature == 0.7
        assert call["sampling"].top_p == 0.9
        assert call["sampling"].top_k == 5
        assert call["sampling"].seed == 42
        assert call["stop"] == ["\n"]

    def test_stream_is_rejected(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert r.status_code == 400
        assert "stream" in r.json()["detail"]

    def test_empty_messages_rejected(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 422


class TestHelpers:
    def test_chat_template_fallback(self):
        class Tokenizer:
            chat_template = None
        prompt = format_chat(Tokenizer(), [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "Hi"},
        ])
        assert prompt == "System: Be nice\nUser: Hi\nAssistant:"

    def test_truncate_at_stop(self):
        assert truncate_at_stop("hello world", ["world"]) == ("hello ", True)
        assert truncate_at_stop("hello world", ["xxx"]) == ("hello world", False)
        assert truncate_at_stop("a\nb\nc", ["\n"]) == ("a", True)


@pytest.fixture(scope="module")
def real_client():
    """真模型端到端：确认 LLMEngine + 分页调度 + HTTP 能串起来。"""
    from engine.llm import LLMConfig, LLMEngine
    llm = LLMEngine(LLMConfig(model_name="gpt2", device="cpu",
                              num_blocks=128, block_size=8))
    return TestClient(create_app(llm)), llm


class TestRealGPT2:
    def test_generates_through_http(self, real_client):
        c, llm = real_client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
            "temperature": 0.0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "gpt2"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] <= 2
        assert body["choices"][0]["finish_reason"] in {"stop", "length"}
        assert llm.pool.num_blocks == 128
