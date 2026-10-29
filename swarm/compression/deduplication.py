"""System Prompt Deduplication & Static Prefix Caching Engine.

Implements SHA-256 prompt hashing, static prefix detection, and an LRU-backed
prompt registry for multi-provider prompt caching.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

EMPTY_PROMPT_HASH = hashlib.sha256(b"").hexdigest()


def compute_prompt_hash(system_prompt: Optional[str]) -> str:
    """Compute SHA-256 hex digest of normalized system prompt.
    
    Normalizes leading and trailing whitespace. If prompt is None or empty,
    returns EMPTY_PROMPT_HASH.
    """
    if system_prompt is None:
        return EMPTY_PROMPT_HASH
    normalized = system_prompt.strip().encode("utf-8")
    if not normalized:
        return EMPTY_PROMPT_HASH
    return hashlib.sha256(normalized).hexdigest()


@dataclass
class PromptCacheEntry:
    """Cache entry representing a registered prompt in the prefix store."""
    prompt_hash: str
    raw_tokens: int = 0
    token_ids: Optional[List[int]] = None
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    content: Optional[str] = None


class PromptRegistry:
    """Thread-safe LRU in-memory store for prompt hashes and static prefixes."""

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[str, PromptCacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_lookups = 0
        self._total_hits = 0

    def __len__(self) -> int:
        """Return the number of cached prompt entries."""
        with self._lock:
            return len(self._cache)

    def register(
        self,
        prompt: str,
        token_count: Optional[int] = None,
        token_ids: Optional[List[int]] = None,
    ) -> PromptCacheEntry:
        """Register or update a system prompt in the cache.
        
        If already registered, updates access order and increments hit_count.
        """
        prompt_hash = compute_prompt_hash(prompt)
        with self._lock:
            if prompt_hash in self._cache:
                entry = self._cache[prompt_hash]
                entry.hit_count += 1
                if token_count is not None:
                    entry.raw_tokens = token_count
                if token_ids is not None:
                    entry.token_ids = token_ids
                self._cache.move_to_end(prompt_hash)
                return entry

            # Create new entry
            entry = PromptCacheEntry(
                prompt_hash=prompt_hash,
                raw_tokens=token_count or 0,
                token_ids=token_ids,
                created_at=time.time(),
                hit_count=0,
                content=prompt.strip(),
            )

            # Evict LRU entry if capacity exceeded
            if len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[prompt_hash] = entry
            return entry

    def get(self, prompt_hash: str) -> Optional[PromptCacheEntry]:
        """Retrieve a cached entry by its SHA-256 hash, updating LRU order."""
        with self._lock:
            self._total_lookups += 1
            if prompt_hash in self._cache:
                self._total_hits += 1
                entry = self._cache[prompt_hash]
                entry.hit_count += 1
                self._cache.move_to_end(prompt_hash)
                return entry
            return None

    def has(self, prompt_hash: str) -> bool:
        """Check if a prompt hash exists in the cache without updating LRU order."""
        with self._lock:
            return prompt_hash in self._cache

    def record_hit(self, prompt_hash: str) -> bool:
        """Increment hit count if present, returning whether it existed."""
        with self._lock:
            self._total_lookups += 1
            if prompt_hash in self._cache:
                self._total_hits += 1
                entry = self._cache[prompt_hash]
                entry.hit_count += 1
                self._cache.move_to_end(prompt_hash)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached prompt entries."""
        with self._lock:
            self._cache.clear()
            self._total_lookups = 0
            self._total_hits = 0

    def size(self) -> int:
        """Return the current number of cached prompt entries."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational telemetry for the prompt registry."""
        with self._lock:
            hit_ratio = (
                (self._total_hits / self._total_lookups)
                if self._total_lookups > 0
                else 0.0
            )
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "hit_ratio": round(hit_ratio, 4),
            }


def _extract_role_and_content(msg: Any) -> Tuple[str, Any]:
    """Helper to extract role and content from ChatMessage or dict."""
    if isinstance(msg, dict):
        return msg.get("role", ""), msg.get("content", "")
    role = getattr(msg, "role", "")
    content = getattr(msg, "content", "")
    return role, content


def detect_static_prefix(
    messages: List[Any],
    cached_messages: List[Any],
) -> Tuple[int, List[Any]]:
    """Identify the longest common message prefix between incoming and cached messages.
    
    Returns:
        (prefix_length, static_messages) where prefix_length is the number of matching
        turns from the beginning of the conversation.
    """
    if not messages or not cached_messages:
        return 0, []

    max_len = min(len(messages), len(cached_messages))
    prefix_length = 0

    for i in range(max_len):
        role_a, content_a = _extract_role_and_content(messages[i])
        role_b, content_b = _extract_role_and_content(cached_messages[i])

        if role_a == role_b and content_a == content_b:
            prefix_length += 1
        else:
            break

    return prefix_length, messages[:prefix_length]


def inject_cache_markers(
    messages: List[Any],
    provider: str = "generic",
) -> List[Any]:
    """Inject provider-specific prefix cache markers onto eligible static turns."""
    # Anthropic ephemeral caching marker
    if provider.lower() in ("anthropic", "claude"):
        marked: List[Any] = []
        for i, msg in enumerate(messages):
            if i == 0:  # Inject cache_control on system instruction
                if isinstance(msg, dict):
                    msg_copy = dict(msg)
                    msg_copy["cache_control"] = {"type": "ephemeral"}
                    marked.append(msg_copy)
                elif hasattr(msg, "model_copy"):
                    marked.append(msg.model_copy(update={"cache_control": {"type": "ephemeral"}}))
                else:
                    marked.append(msg)
            else:
                marked.append(msg)
        return marked
    return messages
