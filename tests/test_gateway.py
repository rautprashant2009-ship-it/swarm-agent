"""Comprehensive automated test suite for Milestone 1 (R1):
Multi-Provider Gateway & Free-Tier Pool Router.
"""

import asyncio
import json
import time
from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from swarm.config import GatewayConfig, load_config
from swarm.core.exceptions import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderPoolExhaustedError,
    ProviderQuotaExhaustedError,
    ProviderUnavailableError,
)
from swarm.core.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    ToolCall,
    Usage,
)
from swarm.main import create_app
from swarm.providers.base import BaseProvider, ProviderState
from swarm.providers.cerebras import CerebrasProvider
from swarm.providers.dashscope import DashScopeProvider
from swarm.providers.google import GoogleProvider
from swarm.providers.groq import GroqProvider
from swarm.providers.moonshot import MoonshotProvider
from swarm.providers.openrouter import OpenRouterProvider
from swarm.providers.siliconflow import SiliconFlowProvider
from swarm.providers.zhipu import ZhipuProvider
from swarm.router.aliases import AliasResolver, CandidateTarget
from swarm.router.pool import ProviderPool


# ==============================================================================
# 1. Pydantic Models & Schemas Verification
# ==============================================================================

def test_chat_completion_request_schema():
    req = ChatCompletionRequest(
        model="deepseek-v4-pro",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, world!"),
        ],
        temperature=0.7,
        stream=False,
    )
    assert req.model == "deepseek-v4-pro"
    assert len(req.messages) == 2
    assert req.messages[0].role == "system"
    assert req.messages[1].content == "Hello, world!"
    assert req.temperature == 0.7


def test_tool_call_request_and_response_schema():
    tool_call = ToolCall(
        id="call_test123",
        function=FunctionCall(name="get_weather", arguments='{"city": "Tokyo"}'),
    )
    msg = ChatMessage(role="assistant", tool_calls=[tool_call])
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "get_weather"

    resp = ChatCompletionResponse(
        model="deepseek-v4-pro",
        choices=[
            Choice(
                index=0,
                message=msg,
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.usage.total_tokens == 15


def test_streaming_chunk_schema():
    chunk = ChatCompletionChunk(
        model="gemini-3.8-flash-high",
        choices=[
            ChunkChoice(
                index=0,
                delta=Delta(content="Hello"),
                finish_reason=None,
            )
        ],
    )
    assert chunk.object == "chat.completion.chunk"
    assert chunk.choices[0].delta.content == "Hello"


# ==============================================================================
# 2. Dual-Source Config Loading & Env Overrides
# ==============================================================================

def test_config_loading_all_8_providers():
    cfg = load_config()
    assert len(cfg.providers) == 8
    expected_providers = {
        "google",
        "siliconflow",
        "moonshot",
        "zhipu",
        "dashscope",
        "openrouter",
        "groq",
        "cerebras",
    }
    assert set(cfg.providers.keys()) == expected_providers


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-secret-123")
    monkeypatch.setenv("SWARM_GATEWAY_PORT", "9999")
    monkeypatch.setenv("SWARM_GATEWAY_HOST", "0.0.0.0")

    cfg = load_config()
    assert cfg.providers["google"].api_key == "env-gemini-secret-123"
    assert cfg.server.port == 9999
    assert cfg.server.host == "0.0.0.0"


# ==============================================================================
# 3. Frontier Model Aliases & Swarm Role Mapping
# ==============================================================================

def test_frontier_alias_resolution():
    resolver = AliasResolver()

    # DeepSeek-V4/Pro
    targets = resolver.resolve("deepseek-v4-pro")
    assert len(targets) >= 2
    assert targets[0].provider == "siliconflow"
    assert "deepseek" in targets[0].model.lower()

    # Kimi-K3
    targets = resolver.resolve("kimi-k3")
    assert targets[0].provider == "moonshot"

    # GLM-5.3
    targets = resolver.resolve("glm-5.3")
    assert targets[0].provider == "zhipu"

    # Qwen 3.8
    targets = resolver.resolve("qwen-3.8")
    assert targets[0].provider == "dashscope"

    # Gemini 3.8 Flash High
    targets = resolver.resolve("gemini-3.8-flash-high")
    assert targets[0].provider == "google"

    # Muse Spark
    targets = resolver.resolve("muse-spark")
    assert targets[0].provider == "openrouter"


def test_swarm_role_resolution():
    resolver = AliasResolver()

    # Swarm roles
    arch_targets = resolver.resolve("swarm-architect")
    assert arch_targets[0].provider == "google"

    coder_targets = resolver.resolve("swarm-coder")
    assert coder_targets[0].provider == "dashscope"

    reasoner_targets = resolver.resolve("swarm-reasoner")
    assert reasoner_targets[0].provider == "siliconflow"

    judge_targets = resolver.resolve("swarm-judge")
    assert judge_targets[0].provider == "zhipu"


def test_passthrough_model_resolution():
    resolver = AliasResolver()

    # Direct Groq model
    targets = resolver.resolve("llama-3.3-70b-versatile")
    assert any(t.provider == "groq" for t in targets)

    # Direct Gemini model
    targets = resolver.resolve("gemini-2.5-flash")
    assert targets[0].provider == "google"


# ==============================================================================
# 4. Circuit Breaker State Machine & Error Detection
# ==============================================================================

def test_circuit_breaker_initial_state():
    p = BaseProvider(name="test_p", api_key="secret", enabled=True)
    assert p.state == ProviderState.HEALTHY
    assert p.is_available() is True

    # Disabled if no key
    p_nokey = BaseProvider(name="test_nokey", api_key="", enabled=True)
    assert p_nokey.state == ProviderState.DISABLED
    assert p_nokey.is_available() is False


def test_circuit_breaker_transitions_and_backoff():
    p = BaseProvider(name="test_p", api_key="secret", base_cooldown=10.0)

    # 1. Failure transitions to COOLING_DOWN
    cooldown = p.record_failure(status_code=429, error_body="Rate limit exceeded")
    assert p.state == ProviderState.COOLING_DOWN
    assert p.is_available() is False
    assert cooldown == 10.0  # 10 * 2^0

    # 2. Second failure doubles backoff
    cooldown2 = p.record_failure(status_code=429, error_body="Rate limit exceeded")
    assert cooldown2 == 20.0  # 10 * 2^1

    # 3. Explicit Retry-After header
    cooldown3 = p.record_failure(status_code=429, error_body="Rate limit", retry_after="45")
    assert cooldown3 == 45.0

    # 4. Cooldown expiration transitions to HALF_OPEN
    p.cooldown_until = time.time() - 1.0  # simulate time passing
    assert p.is_available() is True
    assert p.state == ProviderState.HALF_OPEN

    # 5. Success transitions to HEALTHY
    p.record_success()
    assert p.state == ProviderState.HEALTHY
    assert p.consecutive_failures == 0


def test_circuit_breaker_401_auth_error_permanently_disables():
    p = BaseProvider(name="test_p", api_key="bad-key")
    p.record_failure(status_code=401, error_body="Unauthorized: bad token")
    assert p.state == ProviderState.DISABLED
    assert p.is_available() is False


def test_quota_exhaustion_keyword_detection():
    p = BaseProvider(name="test_p", api_key="test")
    assert p.is_quota_or_rate_limit(429, "") is True
    assert p.is_quota_or_rate_limit(403, "") is True
    assert p.is_quota_or_rate_limit(503, "") is True

    # Body substring matching
    assert p.is_quota_or_rate_limit(200, "RESOURCE_EXHAUSTED: quota exceeded") is True
    assert p.is_quota_or_rate_limit(200, "error: tokens per minute limit reached") is True
    assert p.is_quota_or_rate_limit(200, '{"code": 20014, "message": "Rate limit"}') is True
    assert p.is_quota_or_rate_limit(200, "AllocationExhausted") is True


# ==============================================================================
# 5. Multi-Provider Gateway Transparent Failover (Non-Streaming)
# ==============================================================================

class MockProvider(BaseProvider):
    def __init__(self, name: str, fail_with: Optional[Exception] = None, response_text: str = "Mock answer"):
        super().__init__(name=name, api_key="mock-key", enabled=True)
        self.fail_with = fail_with
        self.response_text = response_text
        self.call_count = 0

    async def chat_completion(self, request, model_override=None):
        self.call_count += 1
        if self.fail_with:
            self.record_failure(429, str(self.fail_with))
            raise self.fail_with
        self.record_success()
        return ChatCompletionResponse(
            model=model_override or request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=self.response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def stream_chat_completion(self, request, model_override=None):
        self.call_count += 1
        if self.fail_with:
            self.record_failure(429, str(self.fail_with))
            raise self.fail_with
        self.record_success()
        for token in ["Hello", " from ", self.name]:
            yield ChatCompletionChunk(
                model=model_override or request.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=Delta(content=token),
                        finish_reason=None,
                    )
                ],
            )
        yield ChatCompletionChunk(
            model=model_override or request.model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=Delta(),
                    finish_reason="stop",
                )
            ],
        )


@pytest.mark.asyncio
async def test_transparent_failover_non_streaming():
    pool = ProviderPool()
    # Mock Provider 1 (SiliconFlow) with 429 Quota Exhausted
    p1_fail = ProviderQuotaExhaustedError("Quota exceeded", "siliconflow", 429)
    mock_p1 = MockProvider("siliconflow", fail_with=p1_fail)

    # Mock Provider 2 (OpenRouter) with success
    mock_p2 = MockProvider("openrouter", response_text="Success from OpenRouter fallback")

    pool.providers["siliconflow"] = mock_p1
    pool.providers["openrouter"] = mock_p2

    req = ChatCompletionRequest(
        model="deepseek-v4-pro",
        messages=[ChatMessage(role="user", content="Test prompt")],
        stream=False,
    )

    response = await pool.route_chat_completion(req)

    assert mock_p1.call_count == 1
    assert mock_p1.state == ProviderState.COOLING_DOWN
    assert mock_p2.call_count == 1
    assert mock_p2.state == ProviderState.HEALTHY
    assert response.choices[0].message.content == "Success from OpenRouter fallback"
    assert pool.total_failovers == 1
    assert pool.total_routed_requests == 1


@pytest.mark.asyncio
async def test_all_providers_exhausted_raises_error():
    pool = ProviderPool()
    p_fail = ProviderQuotaExhaustedError("429 TPM Limit", "mock", 429)

    for p_name in pool.providers:
        pool.providers[p_name] = MockProvider(p_name, fail_with=p_fail)

    req = ChatCompletionRequest(
        model="kimi-k3",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    with pytest.raises(ProviderPoolExhaustedError) as exc_info:
        await pool.route_chat_completion(req)

    assert "All providers configured in pool for alias 'kimi-k3' are currently exhausted" in exc_info.value.message


# ==============================================================================
# 6. Multi-Provider Gateway Transparent Failover (Streaming)
# ==============================================================================

@pytest.mark.asyncio
async def test_transparent_failover_streaming():
    pool = ProviderPool()

    # Candidate 1 fails on initial connection probe
    p1_fail = ProviderQuotaExhaustedError("Stream initial 429", "siliconflow", 429)
    mock_p1 = MockProvider("siliconflow", fail_with=p1_fail)

    # Candidate 2 succeeds
    mock_p2 = MockProvider("openrouter")

    pool.providers["siliconflow"] = mock_p1
    pool.providers["openrouter"] = mock_p2

    req = ChatCompletionRequest(
        model="deepseek-v4-pro",
        messages=[ChatMessage(role="user", content="Stream prompt")],
        stream=True,
    )

    tokens = []
    async for chunk in pool.route_stream_chat_completion(req):
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)

    assert "".join(tokens) == "Hello from openrouter"
    assert mock_p1.call_count == 1
    assert mock_p1.state == ProviderState.COOLING_DOWN
    assert mock_p2.call_count == 1
    assert pool.total_failovers == 1


# ==============================================================================
# 7. Interface Contract: dispatch_role
# ==============================================================================

@pytest.mark.asyncio
async def test_dispatch_role_contract():
    pool = ProviderPool()
    mock_google = MockProvider("google", response_text="Architect directive")
    pool.providers["google"] = mock_google

    response = await pool.dispatch_role(
        role="swarm-architect",
        messages=[ChatMessage(role="user", content="Design architecture")],
    )
    assert response.choices[0].message.content == "Architect directive"


# ==============================================================================
# 8. FastAPI Endpoints Integration Tests
# ==============================================================================

def test_health_endpoint(test_client: TestClient):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "providers" in data
    assert len(data["providers"]) == 8
    assert "google" in data["providers"]
    assert "siliconflow" in data["providers"]
    assert "total_routed_requests" in data
    assert "total_failovers" in data


def test_models_catalog_endpoint(test_client: TestClient):
    resp = test_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "deepseek-v4-pro" in ids
    assert "kimi-k3" in ids
    assert "glm-5.3" in ids
    assert "qwen-3.8" in ids
    assert "gemini-3.8-flash-high" in ids
    assert "swarm-architect" in ids
    assert "swarm-coder" in ids


def test_chat_completions_non_streaming_endpoint(test_client: TestClient):
    app = test_client.app
    pool = app.state.router
    mock_dashscope = MockProvider("dashscope", response_text="Qwen code generated")
    pool.providers["dashscope"] = mock_dashscope

    payload = {
        "model": "qwen-3.8",
        "messages": [{"role": "user", "content": "Write hello world in python"}],
        "stream": False,
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Qwen code generated"


def test_chat_completions_streaming_endpoint(test_client: TestClient):
    app = test_client.app
    pool = app.state.router
    mock_zhipu = MockProvider("zhipu")
    pool.providers["zhipu"] = mock_zhipu

    payload = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "Review this solution"}],
        "stream": True,
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "data: " in text
    assert "data: [DONE]" in text
    assert "Hello" in text


def test_chat_completions_all_exhausted_returns_429(test_client: TestClient):
    app = test_client.app
    pool = app.state.router
    p_fail = ProviderQuotaExhaustedError("All keys quota depleted", "mock", 429)

    for p in pool.providers.values():
        p.record_failure(429, "All quota exhausted")

    payload = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "Analyze document"}],
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "quota_exhausted_all_providers"
    assert data["error"]["type"] == "provider_pool_exhausted_error"


def test_model_not_found_returns_400(test_client: TestClient):
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "model_not_found"


def test_openrouter_headers_injection():
    provider = OpenRouterProvider(api_key="or-key-123", referer="http://custom.test", app_title="CustomApp")
    headers = provider.get_headers()
    assert headers["HTTP-Referer"] == "http://custom.test"
    assert headers["X-Title"] == "CustomApp"
    assert headers["Authorization"] == "Bearer or-key-123"


def test_provider_specific_error_matching():
    sf = SiliconFlowProvider(api_key="key")
    assert sf.is_quota_or_rate_limit(200, '{"code": 20014, "message": "Rate limit exceeded"}') is True

    zhipu = ZhipuProvider(api_key="key")
    assert zhipu.is_quota_or_rate_limit(200, '{"error": {"code": "1301", "message": "concurrency limit"}}') is True

    dash = DashScopeProvider(api_key="key")
    assert dash.is_quota_or_rate_limit(200, '{"code": "AllocationExhausted"}') is True

    moon = MoonshotProvider(api_key="key")
    assert moon.is_quota_or_rate_limit(200, '{"error": {"type": "insufficient_quota"}}') is True


def test_retry_after_regex_extraction():
    p = BaseProvider(name="test", api_key="k", base_cooldown=10.0)
    cooldown = p.record_failure(429, "Rate limit reached. Please try again in 42s.")
    assert cooldown == 42.0

    p2 = BaseProvider(name="test2", api_key="k", base_cooldown=10.0)
    cooldown2 = p2.record_failure(429, "Daily limit reached for requests per day (RPD).")
    assert cooldown2 == 3600.0


@pytest.mark.asyncio
async def test_mid_stream_disconnection_yields_error_chunk():
    class BrokenStreamProvider(BaseProvider):
        def __init__(self):
            super().__init__(name="broken", api_key="k")

        async def stream_chat_completion(self, request, model_override=None):
            yield ChatCompletionChunk(
                model=request.model,
                choices=[ChunkChoice(index=0, delta=Delta(content="First chunk"))],
            )
            raise ConnectionResetError("Connection dropped mid-stream")

    pool = ProviderPool()
    pool.providers["google"] = BrokenStreamProvider()

    req = ChatCompletionRequest(
        model="gemini-flash",
        messages=[ChatMessage(role="user", content="Hi")],
        stream=True,
    )

    chunks = []
    async for chunk in pool.route_stream_chat_completion(req):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "First chunk"
    assert chunks[1].choices[0].finish_reason == "error"


# ==============================================================================
# 9. Real HTTP Transport Mocking (Testing BaseProvider network layer)
# ==============================================================================

@pytest.mark.asyncio
async def test_base_provider_http_200_and_streaming():
    import httpx

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        if b'"stream":true' in body or b'"stream": true' in body:
            sse_content = (
                b'data: {"id":"c-123","object":"chat.completion.chunk","created":1730000000,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":"Live token"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            return httpx.Response(200, content=sse_content, headers={"Content-Type": "text/event-stream"})
        
        resp_data = {
            "id": "chatcmpl-live-123",
            "object": "chat.completion",
            "created": 1730000000,
            "model": "gemini-2.5-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Real HTTP response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    provider = GoogleProvider(api_key="mock-key", client=client)

    # 1. Non-streaming
    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=False,
    )
    resp = await provider.chat_completion(req)
    assert resp.choices[0].message.content == "Real HTTP response"
    assert provider.state == ProviderState.HEALTHY

    # 2. Streaming
    req_stream = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[ChatMessage(role="user", content="Stream me")],
        stream=True,
    )
    chunks = [c async for c in provider.stream_chat_completion(req_stream)]
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "Live token"

    await client.aclose()


@pytest.mark.asyncio
async def test_base_provider_http_error_handling():
    import httpx

    # Test 401 Unauthorized
    def transport_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    c_401 = httpx.AsyncClient(transport=httpx.MockTransport(transport_401))
    p_401 = GoogleProvider(api_key="bad-key", client=c_401)
    req = ChatCompletionRequest(model="gemini-flash", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderAuthError):
        await p_401.chat_completion(req)
    assert p_401.state == ProviderState.DISABLED
    await c_401.aclose()

    # Test 429 Quota Exhausted with Retry-After header
    def transport_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}},
            headers={"Retry-After": "25"},
        )

    c_429 = httpx.AsyncClient(transport=httpx.MockTransport(transport_429))
    p_429 = GoogleProvider(api_key="valid-key", client=c_429)

    with pytest.raises(ProviderQuotaExhaustedError) as exc_info:
        await p_429.chat_completion(req)
    assert p_429.state == ProviderState.COOLING_DOWN
    assert exc_info.value.retry_after == 25.0
    await c_429.aclose()


def test_all_providers_format_payload_defaults():
    empty_req = ChatCompletionRequest(model="", messages=[ChatMessage(role="user", content="Hi")])

    assert GoogleProvider(api_key="k").format_payload(empty_req)["model"] == "gemini-2.5-flash"
    assert SiliconFlowProvider(api_key="k").format_payload(empty_req)["model"] == "deepseek-ai/DeepSeek-V3"
    assert MoonshotProvider(api_key="k").format_payload(empty_req)["model"] == "moonshot-v1-32k"
    assert ZhipuProvider(api_key="k").format_payload(empty_req)["model"] == "glm-4-flash"
    assert DashScopeProvider(api_key="k").format_payload(empty_req)["model"] == "qwen-plus"
    assert OpenRouterProvider(api_key="k").format_payload(empty_req)["model"] == "meta-llama/llama-3.3-70b-instruct:free"
    assert GroqProvider(api_key="k").format_payload(empty_req)["model"] == "llama-3.3-70b-versatile"
    assert CerebrasProvider(api_key="k").format_payload(empty_req)["model"] == "llama3.1-70b"
