"""Pytest fixtures, configuration, and mock provider harness for Swarm Agent test suites."""

import warnings
import pytest
from typing import Any, Dict, List
from starlette.testclient import TestClient

from swarm.config import GatewayConfig, load_config
from swarm.core.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)
from swarm.main import create_app
from swarm.router.pool import ProviderPool

# Suppress Starlette testclient deprecation warning about httpx
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.testclient")


@pytest.fixture
def test_config() -> GatewayConfig:
    """Fixture providing loaded GatewayConfig with all 8 providers enabled and active."""
    cfg = load_config()
    for p in cfg.providers.values():
        p.api_key = "test-key-mock"
        p.enabled = True
    return cfg


@pytest.fixture
def mock_pool(test_config: GatewayConfig) -> ProviderPool:
    """Fixture providing initialized ProviderPool bound to test configuration."""
    return ProviderPool(config=test_config)


@pytest.fixture
def test_client(test_config: GatewayConfig) -> TestClient:
    """Fixture providing TestClient bound to active FastAPI application."""
    app = create_app(test_config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_chat_messages() -> List[ChatMessage]:
    """Fixture providing standard multi-turn conversation messages."""
    return [
        ChatMessage(role="system", content="You are a helpful coding assistant."),
        ChatMessage(role="user", content="Write a python function that adds two numbers."),
        ChatMessage(role="assistant", content="Certainly! Here is the function:\n```python\ndef add(a, b):\n    return a + b\n```\nHope this helps!"),
    ]


@pytest.fixture
def sample_long_trajectory() -> List[ChatMessage]:
    """Fixture providing a long agent trajectory with tool calls and tracebacks for compression tests."""
    messages = [
        ChatMessage(role="system", content="System Anchor: Swarm autonomous agent runtime."),
        ChatMessage(role="user", content="User Anchor: Fix broken unit tests in C:\\repo\\calc.py."),
    ]
    for i in range(6):
        messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    f"Certainly! Checking test step {i}.\n"
                    f"<thought>Need to inspect traceback for failure {i}.</thought>\n"
                    f"Executing test run on C:\\repo\\calc.py."
                ),
            )
        )
        messages.append(
            ChatMessage(
                role="tool",
                content=(
                    f"FAILED tests/test_calc.py::test_case_{i}\n"
                    f"AssertionError: Expected {i}, got {i * 2}\n"
                    f"Traceback in C:\\repo\\calc.py line {10 + i}"
                ),
            )
        )
    messages.extend([
        ChatMessage(role="assistant", content="I resolved the issue in C:\\repo\\calc.py."),
        ChatMessage(role="tool", content="6 passed in 0.04s"),
        ChatMessage(role="assistant", content="Gladly! All 6 tests are passing now.\nHope this helps!"),
        ChatMessage(role="user", content="Please summarize the final status."),
    ])
    return messages
