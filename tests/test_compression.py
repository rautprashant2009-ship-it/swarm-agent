"""Unit and Integration Tests for R2 Context Compression & Token Multiplication Engine.

Verifies:
1. Deduplication: SHA-256 prompt hashing, LRU registry, and static prefix detection.
2. Compactor: Pleasantry stripping, code fence preservation, reasoning pruning, JSON compaction.
3. Sliding Window: Anchors 0 and 1 preservation, middle-turn condensation, recent K turns buffer.
4. Semantic Invariants: Code blocks, file paths, and test assertion failures retention.
5. Metrics: Token counting, multiplier >= 1.0x, tokens saved, headers, and usage augmentation.
6. End-to-End CompressionEngine integration.
"""

import json
import math
import pytest

from swarm.compression import (
    CompressionEngine,
    Compactor,
    SemanticValidator,
    PromptRegistry,
    PromptCacheEntry,
    compute_prompt_hash,
    detect_static_prefix,
    strip_pleasantries,
    strip_reasoning_blocks,
    compact_json_payload,
    extract_critical_entities,
    EMPTY_PROMPT_HASH,
    CompressionStats,
    TokenCounter,
    count_text_tokens,
    count_message_tokens,
    count_messages_tokens,
)
from swarm.core.models import ChatMessage, ToolCall, FunctionCall


# ============================================================================
# 1. Deduplication & Prompt Hashing Tests
# ============================================================================

def test_system_prompt_hashing():
    """Verify SHA-256 prompt hashing handles normalization and empty values."""
    # Empty or None prompt returns EMPTY_PROMPT_HASH
    assert compute_prompt_hash(None) == EMPTY_PROMPT_HASH
    assert compute_prompt_hash("") == EMPTY_PROMPT_HASH
    assert compute_prompt_hash("   ") == EMPTY_PROMPT_HASH

    # Leading/trailing whitespace produces identical hash
    p1 = "You are a helpful programming assistant."
    p2 = "   You are a helpful programming assistant.  \n"
    assert compute_prompt_hash(p1) == compute_prompt_hash(p2)
    assert len(compute_prompt_hash(p1)) == 64


def test_prompt_registry_lru_and_hits():
    """Verify PromptRegistry stores entries, tracks hit counts, and enforces LRU limit."""
    registry = PromptRegistry(max_entries=3)

    entry1 = registry.register("System prompt 1", token_count=50)
    assert entry1.hit_count == 0
    assert entry1.raw_tokens == 50

    # Repeated registration increments hit count
    entry1_again = registry.register("System prompt 1")
    assert entry1_again.hit_count == 1

    # Register additional entries
    registry.register("System prompt 2", token_count=60)
    registry.register("System prompt 3", token_count=70)
    assert registry.size() == 3

    # Access entry 1 to make it recently used
    h1 = compute_prompt_hash("System prompt 1")
    assert registry.has(h1)
    retrieved = registry.get(h1)
    assert retrieved is not None
    assert retrieved.hit_count == 2

    # Register entry 4, which should evict entry 2 (oldest LRU)
    registry.register("System prompt 4", token_count=80)
    assert registry.size() == 3
    h2 = compute_prompt_hash("System prompt 2")
    assert not registry.has(h2)
    assert registry.has(h1)


def test_static_prefix_detection():
    """Verify longest common prefix detection across message sequences."""
    msgs_a = [
        {"role": "system", "content": "You are Swarm Agent."},
        {"role": "user", "content": "Task 1: Fix bug in server."},
        {"role": "assistant", "content": "Executing tool."},
    ]
    msgs_b = [
        {"role": "system", "content": "You are Swarm Agent."},
        {"role": "user", "content": "Task 1: Fix bug in server."},
        {"role": "assistant", "content": "Different response."},
    ]

    prefix_len, static_msgs = detect_static_prefix(msgs_a, msgs_b)
    assert prefix_len == 2
    assert len(static_msgs) == 2
    assert static_msgs[0]["content"] == "You are Swarm Agent."
    assert static_msgs[1]["content"] == "Task 1: Fix bug in server."

    # Completely disjoint
    msgs_c = [{"role": "system", "content": "Completely different"}]
    prefix_len, static_msgs = detect_static_prefix(msgs_a, msgs_c)
    assert prefix_len == 0
    assert len(static_msgs) == 0


# ============================================================================
# 2. Pleasantry & Reasoning Pruning Tests
# ============================================================================

def test_conversational_pleasantry_pruning():
    """Verify opening and closing pleasantries are pruned from assistant messages."""
    text = "Certainly! I'd be happy to help with that. Here is the updated function code.\nHope this helps!"
    stripped = strip_pleasantries(text)
    assert "Certainly!" not in stripped
    assert "I'd be happy to help" not in stripped
    assert "Hope this helps!" not in stripped
    assert "Here is the updated function code" in stripped


def test_code_fence_preservation_during_pleasantry_pruning():
    """Verify that pleasantry-like phrases inside markdown code blocks are NEVER stripped."""
    code_text = (
        "Certainly! Here is your code:\n"
        "```python\n"
        "# Certainly! This is inside the code fence.\n"
        "def greet():\n"
        '    print("Hope this helps!")\n'
        "```\n"
        "Hope this helps!"
    )
    stripped = strip_pleasantries(code_text)

    # Outside pleasantries stripped
    assert not stripped.startswith("Certainly!")
    assert not stripped.endswith("Hope this helps!")

    # Inside code fence preserved 100% verbatim
    assert "# Certainly! This is inside the code fence." in stripped
    assert 'print("Hope this helps!")' in stripped
    assert "```python" in stripped


def test_degenerate_pleasantry_only_message():
    """Verify message consisting only of pleasantries does not become empty."""
    text = "You're welcome!"
    stripped = strip_pleasantries(text)
    assert stripped != ""
    assert "welcome" in stripped.lower()


def test_reasoning_block_pruning():
    """Verify <thought>, <reasoning>, and <thinking> blocks are stripped."""
    text = (
        "<thought>\n"
        "I need to inspect the directory and read the file first.\n"
        "Let's see what happens.\n"
        "</thought>\n"
        "Executing directory inspection at C:\\repo\\src."
    )
    stripped = strip_reasoning_blocks(text)
    assert "<thought>" not in stripped
    assert "I need to inspect the directory" not in stripped
    assert "Executing directory inspection at C:\\repo\\src." in stripped


def test_json_payload_compaction():
    """Verify pretty-printed JSON is minified and invalid JSON is safely handled."""
    pretty_json = '{\n  "path": "C:\\\\repo\\\\file.py",\n  "offset": 1,\n  "limit": 500\n}'
    minified = compact_json_payload(pretty_json)
    assert minified == '{"path":"C:\\\\repo\\\\file.py","offset":1,"limit":500}'

    # Invalid JSON handled safely without crash
    invalid_json = '{"path": "unclosed string'
    assert compact_json_payload(invalid_json) == invalid_json


# ============================================================================
# 3. Semantic Invariant & Critical Entity Preservation Tests
# ============================================================================

def test_critical_entity_extraction():
    """Verify extraction of file paths, assertions, and code fences."""
    content = (
        "Encountered test failure in C:\\Users\\Admin\\swarm\\test_file.py at /var/log/test.log.\n"
        "Traceback (most recent call last):\n"
        "AssertionError: Expected 200 OK, got 429\n"
        "```python\ndef test(): pass\n```"
    )
    entities = extract_critical_entities(content)
    assert "C:\\Users\\Admin\\swarm\\test_file.py" in entities["paths"]
    assert "/var/log/test.log" in entities["paths"]
    assert any("AssertionError" in a for a in entities["assertions"])
    assert len(entities["code_blocks"]) == 1


def test_semantic_validator_detects_violations():
    """Verify SemanticValidator flags missing file paths, errors, or unclosed code fences."""
    original = "Error occurred in C:\\swarm\\agent.py: AssertionError: x != y\n```python\nx = 1\n```"

    # Compressed with missing path and error
    corrupted = "Something went wrong in the code.\n```python\nx = 1\n```"
    is_valid, violations = SemanticValidator.validate(original, corrupted)
    assert not is_valid
    assert any("file paths" in v for v in violations)
    assert any("assertion failure" in v for v in violations)

    # Compressed with unclosed code fence
    broken_fence = "Error occurred in C:\\swarm\\agent.py: AssertionError: x != y\n```python\nx = 1"
    is_valid_fences, violations_fences = SemanticValidator.validate(original, broken_fence)
    assert not is_valid_fences
    assert any("code fences" in v for v in violations_fences)


# ============================================================================
# 4. Sliding Window & Hierarchical Compaction Tests
# ============================================================================

def test_sliding_window_compactor_preserves_anchors_and_recent():
    """Verify Compactor preserves Turn 0 (system), Turn 1 (user goal), and recent K turns."""
    compactor = Compactor(preserve_recent=4)

    messages = [
        ChatMessage(role="system", content="System Anchor 0: Follow strict instructions."),
        ChatMessage(role="user", content="User Anchor 1: Build the feature in C:\\project\\app.py."),
        # Middle turns to condense:
        ChatMessage(role="assistant", content="Certainly! Running pytest first.<thought>check tests</thought>"),
        ChatMessage(role="tool", content="FAILED test_app.py - AssertionError: Expected True, got False"),
        ChatMessage(role="assistant", content="Fixing the bug now."),
        ChatMessage(role="tool", content="File written successfully at C:\\project\\app.py."),
        # Recent buffer (4 turns):
        ChatMessage(role="assistant", content="Running pytest again."),
        ChatMessage(role="tool", content="1 passed in 0.05s"),
        ChatMessage(role="assistant", content="Sure! All tests are passing now. Hope this helps!"),
        ChatMessage(role="user", content="Excellent, please commit the changes."),
    ]

    compacted = compactor.compact(messages, preserve_recent=4)

    # 1. Turn 0 must match system anchor
    assert compacted[0].role == "system"
    assert "System Anchor 0" in compacted[0].content

    # 2. Turn 1 must match user goal anchor
    assert compacted[1].role == "user"
    assert "User Anchor 1" in compacted[1].content

    # 3. Middle turns were condensed into a synthetic trajectory summary
    assert len(compacted) < len(messages)
    summary_turn = compacted[2]
    assert "summary" in summary_turn.content.lower()

    # Invariant: File path and assertion failure preserved in summary
    assert "C:\\project\\app.py" in summary_turn.content
    assert "AssertionError" in summary_turn.content

    # 4. Recent turns preserved with pleasantries stripped
    last_assistant_turn = compacted[-2]
    assert "All tests are passing now." in last_assistant_turn.content
    assert "Hope this helps!" not in last_assistant_turn.content


def test_short_conversation_not_condensed():
    """Verify conversations shorter than 2 + K turns are preserved without middle condensation."""
    compactor = Compactor(preserve_recent=4)
    short_messages = [
        ChatMessage(role="system", content="System instruction."),
        ChatMessage(role="user", content="Hello, write a script."),
        ChatMessage(role="assistant", content="Certainly! Here is your script.\n```python\nprint(1)\n```"),
    ]

    compacted = compactor.compact(short_messages, preserve_recent=4)
    assert len(compacted) == 3
    assert compacted[0].content == "System instruction."
    assert compacted[1].content == "Hello, write a script."
    assert "print(1)" in compacted[2].content


# ============================================================================
# 5. Token Metrics & Telemetry Tests
# ============================================================================

def test_token_counting_and_stats_calculation():
    """Verify token counting, multiplier bounds, headers, and usage augmentation."""
    counter = TokenCounter()

    # Heuristic character count: ceil(len / 3.85)
    text = "Hello world, this is a test prompt."
    tokens = counter.count_text(text)
    assert tokens > 0

    stats = CompressionStats.calculate(
        original_tokens=1000,
        compressed_tokens=400,
        prefix_cache_hit=True,
        duration_ms=12.5,
    )

    assert stats.tokens_saved == 600
    assert stats.token_multiplier == 2.50
    assert stats.compression_ratio == 0.4
    assert stats.prefix_cache_hit is True

    # Headers format
    headers = stats.to_headers()
    assert headers["X-Swarm-Original-Tokens"] == "1000"
    assert headers["X-Swarm-Compressed-Tokens"] == "400"
    assert headers["X-Swarm-Token-Multiplier"] == "2.50x"
    assert headers["X-Swarm-Tokens-Saved"] == "600"
    assert headers["X-Swarm-Prefix-Cache-Hit"] == "true"

    # Multiplier is clamped to minimum 1.0x even if compressed > original (anomaly guard)
    anomaly_stats = CompressionStats.calculate(original_tokens=100, compressed_tokens=200)
    assert anomaly_stats.token_multiplier >= 1.0
    assert anomaly_stats.tokens_saved == 0

    # Usage augmentation
    usage_dict = stats.augment_usage(completion_tokens=50)
    assert usage_dict["prompt_tokens"] == 400
    assert usage_dict["completion_tokens"] == 50
    assert usage_dict["total_tokens"] == 450
    assert usage_dict["prompt_tokens_original"] == 1000
    assert usage_dict["tokens_saved"] == 600
    assert usage_dict["token_multiplier"] == 2.5


# ============================================================================
# 6. End-to-End CompressionEngine Integration Tests
# ============================================================================

def test_compression_engine_end_to_end():
    """Verify CompressionEngine compresses long multi-turn trajectories with multiplier in range."""
    engine = CompressionEngine(preserve_recent=4)

    # Construct long multi-turn agent trajectory
    messages = [
        ChatMessage(role="system", content="You are a swarm engineering agent operating on Windows."),
        ChatMessage(role="user", content="Investigate and repair test failures in C:\\repo\\core.py."),
    ]

    # Add 10 middle turns of verbose tool calls, pleasantries, thoughts, and error messages
    verbose_traceback = (
        "FAILED tests/test_core.py::test_case - AssertionError: Expected 200, got 500\n"
        "Traceback (most recent call last):\n"
        "  File C:\\repo\\core.py line 10, in test_case\n"
        "    assert response.status_code == 200\n"
        "AssertionError: Expected 200, got 500\n"
        + ("Detailed verbose execution trace and log statements from subprocess execution\n" * 15)
    )

    for i in range(5):
        messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    f"Certainly! Let me check step {i}.\n"
                    f"<thought>Evaluating next action and inspecting workspace files in detail for step {i}.</thought>\n"
                    f"Running test iteration {i} on C:\\repo\\core.py."
                ),
            )
        )
        messages.append(
            ChatMessage(
                role="tool",
                content=verbose_traceback,
            )
        )

    # Add recent 4 turns
    messages.extend([
        ChatMessage(role="assistant", content="I found the bug and patched it at C:\\repo\\core.py."),
        ChatMessage(role="tool", content="10 passed, 0 failed in 0.12s"),
        ChatMessage(role="assistant", content="Gladly! Everything is now passing and verified. Hope this helps!"),
        ChatMessage(role="user", content="Please summarize the results."),
    ])

    # First pass: cache miss on system prompt
    compacted_1, stats_1 = engine.compress(messages, preserve_recent=4)

    assert len(compacted_1) < len(messages)
    assert stats_1.original_tokens > stats_1.compressed_tokens
    assert stats_1.tokens_saved > 0
    # Multiplier should be comfortably between 1.8x and 8.0x
    assert 1.8 <= stats_1.token_multiplier <= 8.0
    assert stats_1.prefix_cache_hit is False

    # Second pass with same system prompt: prefix cache hit!
    compacted_2, stats_2 = engine.compress(messages, preserve_recent=4)
    assert stats_2.prefix_cache_hit is True
    assert stats_2.to_headers()["X-Swarm-Prefix-Cache-Hit"] == "true"

    # Verify invariants in compressed representation
    full_compressed_text = "\n".join(str(m.content) for m in compacted_1 if m.content)
    assert "C:\\repo\\core.py" in full_compressed_text
    assert "AssertionError" in full_compressed_text


def test_dict_messages_interoperability():
    """Verify CompressionEngine works seamlessly with raw dictionary message lists."""
    engine = CompressionEngine(preserve_recent=2)
    dict_messages = [
        {"role": "system", "content": "You are a swarm agent."},
        {"role": "user", "content": "Task goal."},
        {"role": "assistant", "content": "Sure! Step 1."},
        {"role": "tool", "content": "Tool 1 output."},
        {"role": "assistant", "content": "Certainly! Step 2."},
        {"role": "tool", "content": "Tool 2 output."},
        {"role": "assistant", "content": "Step 3 final."},
        {"role": "user", "content": "Check status."},
    ]

    compacted, stats = engine.compress(dict_messages, preserve_recent=2)
    assert len(compacted) < len(dict_messages)
    assert isinstance(compacted[0], dict)
    assert compacted[0]["role"] == "system"
    assert stats.token_multiplier >= 1.0


def test_concurrent_prompt_registry_access():
    """Verify PromptRegistry thread safety under concurrent registrations."""
    import concurrent.futures

    registry = PromptRegistry(max_entries=50)

    def worker(worker_id: int):
        for i in range(20):
            prompt = f"System prompt variant {i % 10}"
            registry.register(prompt, token_count=100 + i)
            h = compute_prompt_hash(prompt)
            registry.get(h)
            registry.record_hit(h)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert registry.size() <= 50
    stats = registry.get_stats()
    assert stats["total_lookups"] > 0
    assert stats["total_hits"] > 0


def test_tool_calls_json_compaction_in_messages():
    """Verify tool_calls arguments are minified in assistant messages without losing IDs."""
    compactor = Compactor(preserve_recent=2)
    tc = ToolCall(
        id="call_test_12345",
        function=FunctionCall(
            name="run_command",
            arguments='{\n  "command": "git status",\n  "timeout": 30\n}',
        ),
    )
    msg = ChatMessage(role="assistant", content="Running git command.", tool_calls=[tc])
    compacted = compactor._compact_tool_calls(msg)
    assert compacted.tool_calls[0].id == "call_test_12345"
    assert compacted.tool_calls[0].function.name == "run_command"
    assert compacted.tool_calls[0].function.arguments == '{"command":"git status","timeout":30}'

