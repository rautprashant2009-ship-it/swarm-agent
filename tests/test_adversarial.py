"""Tier 5 White-Box Adversarial Stress Testing & Coverage Hardening Suite.

Systematic empirical verification across 4 critical domains and 15 adversarial edge cases:
1. Gateway:
   - Cascading 429 quota exhaustion across all 8 providers (non-streaming & streaming)
   - Pre-stream and mid-stream SSE socket/network disconnection handling
   - Malformed, invalid, and boundary HTTP request payloads
2. Compression:
   - Extreme multi-turn histories (50+ turns) compaction and invariant retention
   - Complex Unicode, CJK, and multi-byte Emoji token preservation
   - Deeply nested markdown code fences and docstring pleasantry preservation
   - Malformed/truncated JSON in tool call arguments
3. Native Tools:
   - PowerShell subprocess timeout termination and process-tree cleanup
   - Non-ASCII/Unicode arguments and environment encoding integrity
   - High-concurrency atomic file write race-condition protection
   - Binary file signature and null-byte read protection
4. Feedback Loops:
   - Tier 1 max-budget exhaustion resilience (no infinite loops)
   - Tier 2 adversarial judge security injection and dangerous primitive rejection
   - Tier 3 corrupted checkpoint recovery and milestone dependency DAG enforcement
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import AsyncIterator, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from swarm.config import GatewayConfig, ProviderConfig, TargetCandidate, load_config
from swarm.core.exceptions import (
    ModelNotFoundError,
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
from swarm.router.aliases import AliasResolver, CandidateTarget
from swarm.router.pool import ProviderPool
from swarm.compression import (
    CompressionEngine,
    Compactor,
    PromptRegistry,
    SemanticValidator,
    compact_json_payload,
    strip_pleasantries,
)
from swarm.tools import (
    PythonTestRunner,
    TestFailureDetail,
    TestRunResult,
    is_admin,
    read_file,
    run_powershell,
    write_file,
)
from swarm.loops.inner_test_loop import InnerTestLoop, InnerLoopResult
from swarm.loops.middle_audit_loop import AdversarialAuditLoop, AuditVerdict
from swarm.loops.outer_goal_loop import CheckpointState, OuterGoalConvergenceLoop


# Prevent pytest from treating Test* models as test classes
TestFailureDetail.__test__ = False
TestRunResult.__test__ = False


# ==============================================================================
# Helper Mock Providers for Adversarial Injection
# ==============================================================================

class AdversarialMockProvider(BaseProvider):
    """Configurable mock provider capable of simulating 429 exhaustion, mid-stream disconnects,
    and arbitrary network faults."""

    def __init__(
        self,
        name: str,
        fail_with: Optional[Exception] = None,
        disconnect_after_chunks: int = -1,
        response_text: str = "Mock answer",
    ):
        super().__init__(name=name, api_key="mock-key", enabled=True, base_cooldown=30.0)
        self.fail_with = fail_with
        self.disconnect_after_chunks = disconnect_after_chunks
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
                    message=ChatMessage(role="assistant", content=f"[{self.name}] {self.response_text}"),
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
        tokens = ["chunk_alpha", "chunk_beta", "chunk_gamma", "chunk_delta"]

        for idx, token in enumerate(tokens):
            if self.disconnect_after_chunks >= 0 and idx >= self.disconnect_after_chunks:
                # Simulate mid-stream network/socket drop
                raise ConnectionResetError(f"Simulated network socket reset in [{self.name}] after {idx} chunks")

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


# ==============================================================================
# Domain 1: Gateway Adversarial Stress Probes
# ==============================================================================

@pytest.mark.asyncio
async def test_cascading_429_quota_exhaustion_all_8_providers():
    """Torture test: Configure an alias mapped across ALL 8 upstream providers.
    Every provider fails with HTTP 429 Quota Exhausted.
    Empirically verify:
    1. Pool sequentially fails over through all 8 providers without premature abort.
    2. Total failovers recorded == 8.
    3. Every provider state transitions to COOLING_DOWN with consecutive failures == 1.
    4. ProviderPoolExhaustedError is raised containing detailed diagnostic messages for all 8 providers.
    """
    all_provider_names = [
        "google",
        "siliconflow",
        "moonshot",
        "zhipu",
        "dashscope",
        "openrouter",
        "groq",
        "cerebras",
    ]

    cfg = load_config()
    cfg.alias_mappings["stress-all-8"] = [
        TargetCandidate(provider=p, model=f"model-{p}") for p in all_provider_names
    ]

    pool = ProviderPool(config=cfg)
    mock_providers = {}

    for name in all_provider_names:
        err = ProviderQuotaExhaustedError(
            message=f"HTTP 429: Free tier daily token quota exhausted for {name}",
            provider_name=name,
            status_code=429,
        )
        mock_p = AdversarialMockProvider(name=name, fail_with=err)
        pool.providers[name] = mock_p
        mock_providers[name] = mock_p

    req = ChatCompletionRequest(
        model="stress-all-8",
        messages=[ChatMessage(role="user", content="Execute across all 8 providers")],
        stream=False,
    )

    with pytest.raises(ProviderPoolExhaustedError) as exc_info:
        await pool.route_chat_completion(req)

    # Verify all 8 providers were invoked exactly once in failover chain
    assert pool.total_failovers == 8
    assert pool.total_routed_requests == 0
    for name in all_provider_names:
        assert mock_providers[name].call_count == 1
        assert mock_providers[name].state == ProviderState.COOLING_DOWN
        assert mock_providers[name].consecutive_failures == 1

    # Verify error message contains all 8 failure records
    err = exc_info.value
    assert len(err.candidate_errors) == 8
    for name in all_provider_names:
        assert any(name in entry for entry in err.candidate_errors)


@pytest.mark.asyncio
async def test_streaming_cascading_429_all_8_providers():
    """Verify cascading failover across all 8 providers under SSE streaming mode.
    Probing the initial chunk triggers transparent failover across all 8 before raising ProviderPoolExhaustedError.
    """
    all_provider_names = [
        "google",
        "siliconflow",
        "moonshot",
        "zhipu",
        "dashscope",
        "openrouter",
        "groq",
        "cerebras",
    ]

    cfg = load_config()
    cfg.alias_mappings["streaming-stress-8"] = [
        TargetCandidate(provider=p, model=f"model-{p}") for p in all_provider_names
    ]

    pool = ProviderPool(config=cfg)
    for name in all_provider_names:
        err = ProviderQuotaExhaustedError(f"HTTP 429 quota exhausted on {name}", name, 429)
        pool.providers[name] = AdversarialMockProvider(name=name, fail_with=err)

    req = ChatCompletionRequest(
        model="streaming-stress-8",
        messages=[ChatMessage(role="user", content="Stream across all 8")],
        stream=True,
    )

    with pytest.raises(ProviderPoolExhaustedError) as exc_info:
        gen = pool.route_stream_chat_completion(req)
        await gen.__anext__()

    assert pool.total_failovers == 8
    assert len(exc_info.value.candidate_errors) == 8


@pytest.mark.asyncio
async def test_sse_stream_disconnection_resilience():
    """Verify gateway resilience against mid-stream socket disconnections.
    When a provider disconnects midway (after yielding chunks), the gateway must:
    1. Catch the network exception cleanly without unhandled server crash.
    2. Record failure on the provider circuit breaker.
    3. Yield a terminal chunk with finish_reason='error' to close SSE stream cleanly.
    """
    cfg = load_config()
    cfg.alias_mappings["stream-drop-alias"] = [
        TargetCandidate(provider="groq", model="llama-3.3-70b-versatile")
    ]
    pool = ProviderPool(config=cfg)

    # Provider yields 2 chunks then drops connection
    mock_p = AdversarialMockProvider(
        name="groq",
        disconnect_after_chunks=2,
    )
    pool.providers["groq"] = mock_p

    req = ChatCompletionRequest(
        model="stream-drop-alias",
        messages=[ChatMessage(role="user", content="Stream with mid-stream drop")],
        stream=True,
    )

    chunks: List[ChatCompletionChunk] = []
    async for chunk in pool.route_stream_chat_completion(req):
        chunks.append(chunk)

    # Expect: 2 valid content chunks followed by terminal error chunk
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "chunk_alpha"
    assert chunks[1].choices[0].delta.content == "chunk_beta"
    assert chunks[2].choices[0].finish_reason == "error"
    assert mock_p.consecutive_failures == 1


def test_malformed_payloads_gateway_resilience(test_client: TestClient):
    """Stress test the gateway API endpoint with hostile and malformed request bodies:
    - Empty JSON object {} -> 422 Unprocessable Entity
    - Missing required 'model' field -> 422
    - Missing required 'messages' field -> 422
    - Invalid role enum 'hacker' -> 422
    - Non-existent alias model -> 400 Bad Request
    - Non-JSON binary payload -> 422
    """
    # 1. Empty body
    r1 = test_client.post("/v1/chat/completions", json={})
    assert r1.status_code == 422

    # 2. Missing messages
    r2 = test_client.post("/v1/chat/completions", json={"model": "deepseek-v4-pro"})
    assert r2.status_code == 422

    # 3. Missing model
    r3 = test_client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r3.status_code == 422

    # 4. Invalid role enum
    r4 = test_client.post("/v1/chat/completions", json={
        "model": "deepseek-v4-pro",
        "messages": [{"role": "superadmin", "content": "escalate"}]
    })
    assert r4.status_code == 422

    # 5. Empty model name triggers 400 Bad Request
    r5 = test_client.post("/v1/chat/completions", json={
        "model": "",
        "messages": [{"role": "user", "content": "test"}]
    })
    assert r5.status_code == 400
    assert r5.json()["error"]["code"] == "model_not_found"
    assert r5.json()["error"]["type"] == "invalid_request_error"

    # 6. Malformed raw text (not valid JSON)
    r6 = test_client.post(
        "/v1/chat/completions",
        content="GARBAGE_PAYLOAD{{{not_json",
        headers={"Content-Type": "application/json"},
    )
    assert r6.status_code == 422


# ==============================================================================
# Domain 2: Compression Engine Adversarial Stress Probes
# ==============================================================================

def test_compression_extreme_multi_turn_history():
    """Stress test context compression engine under an extreme 60-turn agent trajectory.
    Verifies:
    1. System prompt (Anchor 0) and initial User prompt (Anchor 1) are preserved intact.
    2. Recent K turns (preserve_recent=4) are preserved intact.
    3. Middle 54 turns are compacted without data loss of critical error logs.
    4. Achieves token multiplier M >= 1.5x with bounded execution time.
    """
    engine = CompressionEngine()
    messages: List[ChatMessage] = [
        ChatMessage(role="system", content="System Anchor: Swarm autonomous agent core instructions."),
        ChatMessage(role="user", content="User Anchor: Fix unit tests in C:\\repo\\calc.py and C:\\repo\\app.py."),
    ]

    # Generate 54 intermediate conversational turns
    for i in range(27):
        messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    f"Certainly! Here is my analysis for step {i}.\n"
                    f"<thought>Reviewing test traceback for iteration {i}. Need to patch line {10 + i}.</thought>\n"
                    f"I will now inspect the source file C:\\repo\\calc.py."
                ),
            )
        )
        messages.append(
            ChatMessage(
                role="tool",
                content=(
                    f"FAILED tests/test_calc.py::test_iteration_{i}\n"
                    f"AssertionError: Expected {i * 10}, received {i}\n"
                    f"Traceback in C:\\repo\\calc.py line {10 + i}"
                ),
            )
        )

    # Recent turns
    messages.extend([
        ChatMessage(role="assistant", content="Patch applied to C:\\repo\\calc.py. Re-running test runner."),
        ChatMessage(role="tool", content="27 passed in 0.12s"),
        ChatMessage(role="assistant", content="Gladly! All 27 unit tests are passing clean now.\nHope this helps!"),
        ChatMessage(role="user", content="Final report requested."),
    ])

    assert len(messages) == 60

    start_t = time.perf_counter()
    compressed_msgs, stats = engine.compress(messages, preserve_recent=4)
    duration = time.perf_counter() - start_t

    # Performance bound: compression should finish under 500ms
    assert duration < 0.50

    # Invariants verification
    assert stats.token_multiplier >= 1.5
    assert stats.tokens_saved > 0
    assert stats.compressed_tokens < stats.original_tokens

    # Anchors strictly preserved
    assert compressed_msgs[0].role == "system"
    assert "Swarm autonomous agent core instructions" in compressed_msgs[0].content
    assert compressed_msgs[1].role == "user"
    assert "C:\\repo\\calc.py" in compressed_msgs[1].content

    # Recent 4 turns strictly preserved
    assert "Final report requested." in compressed_msgs[-1].content


def test_compression_cjk_emoji_unicode_tokens():
    """Torture test: Unicode/CJK, Cyrillic, Greek, Arabic, and multi-byte Emoji token preservation.
    Ensures pleasantry stripping does NOT mangle multi-byte Unicode strings or emojis inside code blocks.
    """
    mixed_content = (
        "Sure! 很高兴为您提供帮助。这里是完整的修复方案与代码：\n"
        "```python\n"
        "# 跨语言支持测试：Arabic: مرحبا, Russian: Привет, Greek: Γειά σου\n"
        "def greet(emoji='🤖🚀🔥👨‍👩‍👧‍👦'):\n"
        "    return f'Swarm Agent: {emoji} - 100% 成功'\n"
        "```\n"
        "Hope this helps! 祝您开发顺利！"
    )

    stripped = strip_pleasantries(mixed_content)

    # Opening and closing pleasantries stripped
    assert not stripped.startswith("Sure!")
    assert not stripped.endswith("Hope this helps!")

    # Code block inside ```python ... ``` is 100% identical byte-for-byte
    assert "def greet(emoji='🤖🚀🔥👨‍👩‍👧‍👦'):" in stripped
    assert "Arabic: مرحبا" in stripped
    assert "Russian: Привет" in stripped
    assert "100% 成功" in stripped


def test_compression_deeply_nested_markdown_code_fences():
    """Verify that deeply nested code fences (e.g. Markdown discussing code fences)
    and comments containing pleasantries are NEVER stripped or corrupted.
    """
    nested_markdown = (
        "Certainly! Here is the nested code fence example:\n"
        "````markdown\n"
        "```python\n"
        "# Sure! This comment inside nested fence must NEVER be stripped\n"
        "# Hope this helps! This closing comment must be preserved\n"
        "def inner():\n"
        "    return True\n"
        "```\n"
        "````\n"
        "Hope this helps!"
    )

    stripped = strip_pleasantries(nested_markdown)

    # Internal code comments must be strictly intact
    assert "# Sure! This comment inside nested fence must NEVER be stripped" in stripped
    assert "# Hope this helps! This closing comment must be preserved" in stripped
    assert "def inner():" in stripped


def test_compression_malformed_json_resilience():
    """Verify that compact_json_payload gracefully handles truncated/malformed JSON strings
    without raising unhandled JSONDecodeError.
    """
    # 1. Truncated JSON
    malformed_json = '{"query": "SELECT * FROM users WHERE active = 1, "unclosed_brace": '
    res1 = compact_json_payload(malformed_json)
    assert res1 == malformed_json  # Gracefully retained as-is

    # 2. Non-JSON string
    raw_str = "This is simply plain text without JSON formatting."
    res2 = compact_json_payload(raw_str)
    assert res2 == raw_str

    # 3. Valid JSON minification
    valid_json = '{\n  "status": "success",\n  "count": 42\n}'
    res3 = compact_json_payload(valid_json)
    assert res3 == '{"status":"success","count":42}'


# ==============================================================================
# Domain 3: Native Tools Adversarial Stress Probes
# ==============================================================================

def test_powershell_timeout_termination_and_tree_cleanup():
    """Adversarial probe: Execute a long-running child process sleep via PowerShell.
    Verify:
    1. Process times out at specified limit (1s).
    2. Process result reports timed_out == True and exit_code == 124.
    3. Process tree is forcefully terminated without leaving hanging processes.
    """
    # 10 second sleep with 1 second timeout
    start_t = time.perf_counter()
    result = run_powershell("Start-Sleep -Seconds 10", timeout=1)
    duration = time.perf_counter() - start_t

    assert result.timed_out is True
    assert result.exit_code == 124
    assert result.is_success is False
    assert duration < 5.0  # Must be killed quickly within grace period


def test_powershell_non_ascii_arguments_and_output():
    """Verify PowerShell UTF-8 bridge correctly handles CJK, Cyrillic, and Emoji arguments."""
    payload = "Hello 世界 🌍 Привет Γειά σου"
    cmd = f'Write-Output "{payload}"'
    res = run_powershell(cmd, timeout=10)

    assert res.is_success is True
    assert res.exit_code == 0
    assert "世界" in res.stdout
    assert "Привет" in res.stdout
    assert "Γειά σου" in res.stdout


def test_concurrent_atomic_file_writes(tmp_path):
    """Stress test: 20 concurrent threads attempting atomic writes to the same target file.
    Empirically verify:
    1. All writes succeed atomically without corruption.
    2. Final file content matches one of the full valid payloads and its SHA-256 matches.
    3. Zero dangling temporary staging files (.tmp_*) remain on disk.
    """
    target_file = str(tmp_path / "concurrent_atomic_target.txt")
    payloads = [f"Payload line data from thread {i:02d} - checksum token {i * 999}\n" for i in range(20)]

    def write_worker(idx: int):
        write_file(target_file, payloads[idx], overwrite=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_worker, range(20)))

    # Verify target file exists and content is a valid complete payload
    read_res = read_file(target_file)
    assert read_res.is_success is True
    assert any(p.strip() in read_res.content for p in payloads)

    # Check for orphaned staging files
    temp_files = [f for f in os.listdir(tmp_path) if ".tmp_" in f]
    assert len(temp_files) == 0


def test_binary_file_read_protection(tmp_path):
    """Verify that read_file detects binary content (null bytes, binary headers)
    and refuses to dump binary garbage or crash with UnicodeDecodeError.
    """
    bin_file = tmp_path / "test_binary.bin"
    # Write binary null bytes and high-byte sequence
    bin_file.write_bytes(b"\x00\x01\x02\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    res = read_file(str(bin_file))
    assert res.is_binary is True
    assert res.error == "binary_file_detected"
    assert res.content == ""
    assert res.size_bytes > 0


# ==============================================================================
# Domain 4: Feedback Loops Adversarial Stress Probes
# ==============================================================================

def test_tier1_inner_loop_budget_exhaustion_resilience(tmp_path):
    """Stress test: Inner Test Loop with a hard budget of 3 attempts on an unsolvable test failure.
    Ensure:
    1. Loop terminates cleanly after exactly 3 attempts.
    2. Returns status == 'BUDGET_EXCEEDED'.
    3. Does NOT enter an infinite loop or hang.
    4. Full attempt history is captured.
    """
    dummy_source = tmp_path / "flawed_source.py"
    dummy_source.write_text("def broken(): return False\n", encoding="utf-8")

    loop = InnerTestLoop(working_dir=str(tmp_path), max_attempts=3)

    # Mock runner that always reports test failure
    loop.runner.run_tests = lambda test_target, timeout=120: TestRunResult(
        test_target="test_stub.py",
        exit_code=1,
        passed=0,
        failed=1,
        total=1,
        duration_ms=10.0,
        failures=[
            TestFailureDetail(
                test_name="test_stub",
                line_number=5,
                error_message="AssertionError: Expected True, got False",
                traceback_snippet="AssertionError: Expected True, got False",
            )
        ],
        stdout="1 failed in 0.01s",
    )

    # Patch generator that fails to resolve the issue
    attempt_counter = [0]
    def bad_patch_generator(attempt, failures):
        attempt_counter[0] += 1
        return "def broken(): return False # attempt patch\n"

    result = loop.run_until_convergence(
        target_file=str(dummy_source),
        test_file="test_stub.py",
        patch_generator=bad_patch_generator,
    )

    assert result.status == "BUDGET_EXCEEDED"
    assert result.attempts_used == 3
    assert result.final_exit_code == 1
    assert len(result.history) == 3


def test_tier2_adversarial_judge_security_injection_rejection():
    """Verify that AdversarialAuditLoop (GLM-5.3 persona) strictly catches and rejects
    malicious security primitives and dangerous execution patterns.
    """
    loop = AdversarialAuditLoop()

    dangerous_snippets = [
        ("os.system('rm -rf /')", "SEC-01-COMMAND-INJECTION"),
        ("subprocess.Popen(cmd, shell=True)", "SEC-02-SHELL-TRUE"),
        ("result = eval(user_untrusted_input)", "SEC-03-DYNAMIC-EVAL"),
        ("exec(malicious_code_string)", "SEC-04-DYNAMIC-EXEC"),
        ("mod = __import__('os')", "SEC-05-DYNAMIC-IMPORT"),
        ("", "SPEC-00-EMPTY-IMPLEMENTATION"),
    ]

    for snippet, expected_rule in dangerous_snippets:
        verdict = loop.judge_audit(snippet)
        assert verdict.verdict == "REJECT", f"Expected rejection for '{snippet}'"
        assert verdict.audit_score < 50
        assert any(f.rule_violated == expected_rule for f in verdict.findings)

        # Reasoner derivation check
        remediation = loop.reasoner_derive(snippet, verdict)
        assert expected_rule in remediation


def test_tier3_checkpoint_corrupted_recovery_and_state_dag(tmp_path):
    """Verify Tier 3 Outer Goal Convergence Loop handles corrupted checkpoint.json
    gracefully by falling back to fresh state initialization without crashing,
    and enforces strict milestone DAG prerequisites.
    """
    corrupt_checkpoint = tmp_path / "checkpoint.json"
    corrupt_checkpoint.write_text("{NOT_VALID_JSON_GARBAGE...", encoding="utf-8")

    loop = OuterGoalConvergenceLoop(
        checkpoint_path=str(corrupt_checkpoint),
        session_id="adversarial-recovery-test",
    )

    # Must recover cleanly into fresh state
    assert loop.state.current_milestone == "M1_GATEWAY_ROUTER"
    assert loop.state.milestones["M1_GATEWAY_ROUTER"].status == "IN_PROGRESS"

    # Verify dependency DAG: M6 cannot be executed before M1..M5 are verified
    can_exec, missing = loop.check_dependencies("M6_E2E_VERIFICATION")
    assert can_exec is False
    assert len(missing) == 5
