"""Token Measurement, Multiplier Calculation, and Telemetry Engine.

Provides dual token counting (exact via tiktoken or robust 3.85 char/token heuristic fallback),
effective token multiplier calculation, tokens saved metric, response headers injection,
and OpenAI usage object augmentation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import time
from typing import Any, Dict, List, Optional, Union

# Attempt to import tiktoken; fallback to character-based heuristic if unavailable
_TIKTOKEN_ENCODER = None
try:
    import tiktoken
    try:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        pass
except ImportError:
    pass

CHARS_PER_TOKEN = 3.85


def count_text_tokens(text: Optional[str]) -> int:
    """Calculate token count for text using tiktoken or heuristic fallback.
    
    Fallback formula from specification:
        tokens = ceil(len(text) / 3.85)
    """
    if not text:
        return 0

    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass

    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def count_message_tokens(message: Any) -> int:
    """Calculate total token count for a single message turn.
    
    Includes role, content, name, tool_calls, and tool_call_id, plus message framing.
    Supports ChatMessage Pydantic models and standard dicts.
    """
    if message is None:
        return 0

    tokens = 4  # OpenAI protocol framing overhead per message (<|im_start|>role\n ... <|im_end|>)

    # Extract fields
    if isinstance(message, dict):
        role = message.get("role", "")
        content = message.get("content", "")
        name = message.get("name")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")
    else:
        role = getattr(message, "role", "")
        content = getattr(message, "content", "")
        name = getattr(message, "name", None)
        tool_calls = getattr(message, "tool_calls", None)
        tool_call_id = getattr(message, "tool_call_id", None)

    tokens += count_text_tokens(str(role) if role else "")

    if content is not None:
        if isinstance(content, str):
            tokens += count_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        tokens += count_text_tokens(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        tokens += 65  # Approximate token cost for image metadata
                else:
                    tokens += count_text_tokens(str(part))
        else:
            tokens += count_text_tokens(str(content))

    if name:
        tokens += count_text_tokens(str(name))

    if tool_call_id:
        tokens += count_text_tokens(str(tool_call_id))

    if tool_calls:
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    fn_args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                    tokens += count_text_tokens(str(fn_name))
                    tokens += count_text_tokens(str(fn_args))
                else:
                    fn = getattr(tc, "function", None)
                    if fn:
                        tokens += count_text_tokens(getattr(fn, "name", ""))
                        tokens += count_text_tokens(getattr(fn, "arguments", ""))

    return tokens


def count_messages_tokens(messages: List[Any]) -> int:
    """Calculate total token count across a list of message turns."""
    if not messages:
        return 0
    total = sum(count_message_tokens(m) for m in messages)
    total += 3  # Conversation primer tokens (<|im_start|>assistant)
    return total


@dataclass
class CompressionStats:
    """Statistics and metrics computed from a compression pass."""
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    token_multiplier: float
    compression_ratio: float
    prefix_cache_hit: bool = False
    duration_ms: float = 0.0

    @classmethod
    def calculate(
        cls,
        original_tokens: int,
        compressed_tokens: int,
        prefix_cache_hit: bool = False,
        duration_ms: float = 0.0,
    ) -> CompressionStats:
        """Construct CompressionStats applying mathematical bounds and guards.
        
        Multiplier M = original_tokens / max(compressed_tokens, 1), clamped to >= 1.0.
        Tokens saved S = max(original_tokens - compressed_tokens, 0).
        Compression ratio R = compressed_tokens / max(original_tokens, 1).
        """
        orig = max(original_tokens, 0)
        comp = max(compressed_tokens, 0)

        # Saved tokens cannot be negative
        tokens_saved = max(orig - comp, 0)

        # Multiplier clamped to a minimum of 1.0
        if comp > 0:
            raw_multiplier = orig / comp
            multiplier = max(1.0, round(raw_multiplier, 2))
        else:
            multiplier = 1.0

        if orig > 0:
            ratio = round(comp / orig, 4)
        else:
            ratio = 1.0

        return cls(
            original_tokens=orig,
            compressed_tokens=comp,
            tokens_saved=tokens_saved,
            token_multiplier=multiplier,
            compression_ratio=ratio,
            prefix_cache_hit=prefix_cache_hit,
            duration_ms=round(duration_ms, 2),
        )

    def to_headers(self) -> Dict[str, str]:
        """Convert statistics to HTTP response headers."""
        return {
            "X-Swarm-Original-Tokens": str(self.original_tokens),
            "X-Swarm-Compressed-Tokens": str(self.compressed_tokens),
            "X-Swarm-Token-Multiplier": f"{self.token_multiplier:.2f}x",
            "X-Swarm-Tokens-Saved": str(self.tokens_saved),
            "X-Swarm-Prefix-Cache-Hit": "true" if self.prefix_cache_hit else "false",
        }

    def augment_usage(
        self,
        usage: Optional[Union[Dict[str, Any], Any]] = None,
        completion_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Augment or generate an OpenAI-compatible usage dictionary."""
        if usage is not None:
            if isinstance(usage, dict):
                comp_tokens = usage.get("completion_tokens", completion_tokens)
            else:
                comp_tokens = getattr(usage, "completion_tokens", completion_tokens)
        else:
            comp_tokens = completion_tokens

        prompt_tokens = self.compressed_tokens
        total_tokens = prompt_tokens + comp_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_original": self.original_tokens,
            "tokens_saved": self.tokens_saved,
            "token_multiplier": self.token_multiplier,
            "prefix_cache_hit": self.prefix_cache_hit,
        }


class TokenCounter:
    """Stateful token counting utility with session accumulation."""

    def __init__(self) -> None:
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.total_tokens_saved = 0

    def count_text(self, text: Optional[str]) -> int:
        return count_text_tokens(text)

    def count_message(self, message: Any) -> int:
        return count_message_tokens(message)

    def count_messages(self, messages: List[Any]) -> int:
        return count_messages_tokens(messages)

    def record_pass(self, stats: CompressionStats) -> None:
        """Accumulate token statistics across requests."""
        self.total_original_tokens += stats.original_tokens
        self.total_compressed_tokens += stats.compressed_tokens
        self.total_tokens_saved += stats.tokens_saved

    @property
    def cumulative_multiplier(self) -> float:
        """Effective token multiplier across all recorded passes."""
        if self.total_compressed_tokens > 0:
            return max(1.0, round(self.total_original_tokens / self.total_compressed_tokens, 2))
        return 1.0
