"""Context Compression & Token Multiplication Engine (R2 Subsystem).

Coordinates deduplication, history stripping, sliding window compaction,
semantic invariant validation, and token multiplier metrics.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from swarm.compression.compactor import (
    Compactor,
    SemanticValidator,
    compact_json_payload,
    extract_critical_entities,
    strip_pleasantries,
    strip_reasoning_blocks,
)
from swarm.compression.deduplication import (
    EMPTY_PROMPT_HASH,
    PromptCacheEntry,
    PromptRegistry,
    compute_prompt_hash,
    detect_static_prefix,
    inject_cache_markers,
)
from swarm.compression.metrics import (
    CompressionStats,
    TokenCounter,
    count_message_tokens,
    count_messages_tokens,
    count_text_tokens,
)


class CompressionEngine:
    """Inline Context-Compression and Prompt-Caching Layer.
    
    Transforms multi-turn prompt history by removing conversational pleasantries,
    pruning reasoning tags, minifying JSON payloads, deduplicating system prompts,
    and compacting middle turns while preserving code blocks, paths, and assertions.
    """

    def __init__(
        self,
        preserve_recent: int = 4,
        max_prompt_cache_entries: int = 1000,
    ) -> None:
        self.preserve_recent = preserve_recent
        self.prompt_registry = PromptRegistry(max_entries=max_prompt_cache_entries)
        self.compactor = Compactor(preserve_recent=preserve_recent)
        self.token_counter = TokenCounter()

    def compress(
        self,
        messages: List[Any],
        preserve_recent: Optional[int] = None,
    ) -> Tuple[List[Any], CompressionStats]:
        """Compress message history and return compacted messages with CompressionStats.
        
        Args:
            messages: List of ChatMessage objects or message dictionaries.
            preserve_recent: Optional override for number of recent turns to preserve verbatim.
            
        Returns:
            Tuple of (compacted_messages, compression_stats)
        """
        start_time = time.perf_counter()

        if not messages:
            empty_stats = CompressionStats.calculate(0, 0, False, 0.0)
            return [], empty_stats

        # 1. Count original raw tokens
        original_tokens = count_messages_tokens(messages)

        # 2. Check for system prompt deduplication / prefix cache hit
        prefix_cache_hit = False
        system_content = None

        first_msg = messages[0]
        role = first_msg.get("role") if isinstance(first_msg, dict) else getattr(first_msg, "role", "")
        if role == "system":
            system_content = (
                first_msg.get("content")
                if isinstance(first_msg, dict)
                else getattr(first_msg, "content", "")
            )
            if isinstance(system_content, str) and system_content.strip():
                prompt_hash = compute_prompt_hash(system_content)
                if self.prompt_registry.has(prompt_hash):
                    prefix_cache_hit = True
                    self.prompt_registry.record_hit(prompt_hash)
                else:
                    self.prompt_registry.register(system_content, count_text_tokens(system_content))

        # 3. Compact message trajectory
        k = preserve_recent if preserve_recent is not None else self.preserve_recent
        compacted_messages = self.compactor.compact(messages, preserve_recent=k)

        # 4. Count compressed tokens
        compressed_tokens = count_messages_tokens(compacted_messages)

        # 5. Compute metrics & duration
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        stats = CompressionStats.calculate(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            prefix_cache_hit=prefix_cache_hit,
            duration_ms=duration_ms,
        )

        # Record metrics pass
        self.token_counter.record_pass(stats)

        return compacted_messages, stats


__all__ = [
    "CompressionEngine",
    "Compactor",
    "SemanticValidator",
    "strip_pleasantries",
    "strip_reasoning_blocks",
    "compact_json_payload",
    "extract_critical_entities",
    "PromptRegistry",
    "PromptCacheEntry",
    "compute_prompt_hash",
    "detect_static_prefix",
    "inject_cache_markers",
    "EMPTY_PROMPT_HASH",
    "CompressionStats",
    "TokenCounter",
    "count_text_tokens",
    "count_message_tokens",
    "count_messages_tokens",
]
