"""Unit tests for the OpenCode-backed AIChecker worker."""
import json

import httpx
import pytest

from workers import ai_checker as ai_checker_module
from workers.ai_checker import AIChecker


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeAsyncClient:
    """Minimal async stand-in for httpx.AsyncClient that records requests."""

    calls: list[tuple[str, str, dict]] = []
    post_payload: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeAsyncClient.calls.append(("POST", url, json or {}))
        request = httpx.Request("POST", url)
        if url.endswith("/session"):
            return httpx.Response(200, json={"id": "sess-123"}, request=request)
        return httpx.Response(200, json=_FakeAsyncClient.post_payload, request=request)

    async def delete(self, url):
        _FakeAsyncClient.calls.append(("DELETE", url, {}))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("DELETE", url))


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.post_payload = {}


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(ai_checker_module.httpx, "AsyncClient", _FakeAsyncClient)
    return AIChecker(
        opencode_url="http://opencode.test",
        provider_id="agentrouter",
        model_id="deepseek-v4-flash",
    )


def _verdict_msg():
    return {
        "info": {
            "sessionID": "sess-123",
            "tokens": {"input": 100, "output": 25, "total": 125},
            "reasoning": {"tokens": 10},
        },
        "parts": [
            {
                "type": "text",
                "text": '```json\n{"likelihood_score": 80, "reasoning": "robotic", "is_ai_generated": true}\n```',
            }
        ],
    }


@pytest.mark.anyio
async def test_check_calls_opencode_session(checker):
    _FakeAsyncClient.post_payload = _verdict_msg()

    result = await checker.check("some document text")

    assert result["parsed"] == {
        "likelihood_score": 80,
        "reasoning": "robotic",
        "is_ai_generated": True,
    }
    assert result["raw"]["provider"] == "opencode"
    assert result["raw"]["usage"]["total_tokens"] == 125
    assert result["raw"]["usage"]["is_estimated"] is False


@pytest.mark.anyio
async def test_check_posts_model_and_prompt(checker):
    _FakeAsyncClient.post_payload = _verdict_msg()

    await checker.check("hello world")

    methods = [(m, u) for m, u, _ in _FakeAsyncClient.calls]
    assert ("POST", "http://opencode.test/session") in methods
    assert ("POST", "http://opencode.test/session/sess-123/message") in methods
    assert ("DELETE", "http://opencode.test/session/sess-123") in methods

    message_call = next(
        body for method, url, body in _FakeAsyncClient.calls
        if url.endswith("/message")
    )
    assert message_call["model"] == {
        "providerID": "agentrouter",
        "modelID": "deepseek-v4-flash",
    }
    assert "hello world" in message_call["parts"][0]["text"]


@pytest.mark.anyio
async def test_check_session_deleted_on_empty_response(checker):
    _FakeAsyncClient.post_payload = {"info": {}, "parts": []}

    result = await checker.check("text")

    assert "parse_error" in result["parsed"]
    urls = [u for _, u, _ in _FakeAsyncClient.calls]
    assert "http://opencode.test/session/sess-123" in urls


def test_init_requires_provider_and_model(monkeypatch):
    monkeypatch.delenv("OPENCODE_PROVIDER_ID", raising=False)
    monkeypatch.delenv("OPENCODE_MODEL_ID", raising=False)
    with pytest.raises(ValueError):
        AIChecker(opencode_url="http://opencode.test")


def test_parse_response_repairs_eaten_brace(checker):
    parsed = checker._parse_response('json\n"likelihood_score": 10, "reasoning": "ok", "is_ai_generated": false}')
    assert parsed["likelihood_score"] == 10
    assert parsed["is_ai_generated"] is False
