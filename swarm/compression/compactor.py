"""Context Compactor & Semantic Invariant Preservation Engine.

Implements:
1. Conversational pleasantry stripping (exempting code blocks).
2. Intermediate reasoning block pruning.
3. Tool JSON compaction.
4. Sliding window & anchor history compaction.
5. Critical entity preservation & Semantic Invariant Validator.
"""

from __future__ import annotations
import copy
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from swarm.core.models import ChatMessage, ToolCall, FunctionCall
except ImportError:
    ChatMessage = None  # type: ignore

# Opening pleasantry regex patterns
PLEASANTRY_OPENING_PATTERNS = [
    re.compile(
        r"^(?:Sure|Certainly|Of course|Gladly|Hello|Hi there|Understood|No problem)\b[\s,!.]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:I(?:'d|\swould)\sbe\shappy\sto\shelp(?:\swith\sthat)?)\b[!.]*\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Here(?:'s|\sis)\sthe\s(?:solution|code|implementation|update|file)\b)[:\s]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Let\sme\s(?:know\sif|help\syou|take\sa\slook))\b[\w\s,!.]*?:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Okay|Ok|Great|Awesome|Alright)\b[\s,!.]*",
        re.IGNORECASE,
    ),
]

# Closing pleasantry regex patterns
PLEASANTRY_CLOSING_PATTERNS = [
    re.compile(
        r"(?:(?:\n|^|\s+))(?:Let\sme\sknow\sif\syou\s(?:need|have)\sany(?:thing\selse|other\squestions)[\w\s.!?]*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:\n|^|\s+))(?:Hope\sthis\shelps[!.]*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:\n|^|\s+))(?:Feel\sfree\sto\sask\sif\syou\srun\sinto\sany\sissues[!.]*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:\n|^|\s+))(?:You(?:'re|\sare)\swelcome[!.]*)$",
        re.IGNORECASE,
    ),
]

# Intermediate thought / reasoning block tags
REASONING_BLOCK_PATTERNS = [
    re.compile(r"<thought>[\s\S]*?</thought>", re.IGNORECASE),
    re.compile(r"<reasoning>[\s\S]*?</reasoning>", re.IGNORECASE),
    re.compile(r"<thinking>[\s\S]*?</thinking>", re.IGNORECASE),
    re.compile(r"^Thinking Process:[\s\S]*?(?=\n\n|\Z)", re.IGNORECASE | re.MULTILINE),
]

# Critical entity patterns
WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[A-Za-z0-9_.\-\\]+")
POSIX_PATH_PATTERN = re.compile(r"(?:/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+)")
ASSERTION_ERROR_PATTERN = re.compile(
    r"(?:AssertionError:.*?$|FAILED\s+.*?$|FAIL:\s+.*?$|Traceback \(most recent call last\):.*?$|Error:\s+.*?$|Exception:\s+.*?$)",
    re.MULTILINE,
)
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")


def split_text_and_code_spans(text: str) -> List[Tuple[str, bool]]:
    """Split text into alternating segments of (content, is_code).
    
    Protects both markdown code fences (```...```) and inline code (`...`).
    """
    if not text:
        return []

    # Combined fence pattern matching ``` blocks first, then ` inline blocks
    combined_pattern = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
    spans: List[Tuple[str, bool]] = []
    last_idx = 0

    for match in combined_pattern.finditer(text):
        start, end = match.span()
        if start > last_idx:
            spans.append((text[last_idx:start], False))
        spans.append((text[start:end], True))
        last_idx = end

    if last_idx < len(text):
        spans.append((text[last_idx:], False))

    return spans


def strip_pleasantries(text: Optional[str]) -> str:
    """Remove opening and closing conversational pleasantries from text.
    
    Strict Invariant: Content inside Markdown code fences or inline backticks
    is strictly protected and never modified.
    If stripping would reduce the entire message to empty, preserves original.
    """
    if not text or not text.strip():
        return text or ""

    spans = split_text_and_code_spans(text)
    if not spans:
        return text

    # Process opening pleasantry on the first text span
    first_non_code_idx = next((i for i, (_, is_code) in enumerate(spans) if not is_code), None)
    if first_non_code_idx is not None:
        seg_content, _ = spans[first_non_code_idx]
        modified_seg = seg_content.strip()
        changed = True
        while changed:
            changed = False
            for pattern in PLEASANTRY_OPENING_PATTERNS:
                new_seg = pattern.sub("", modified_seg).strip()
                if new_seg != modified_seg:
                    modified_seg = new_seg
                    changed = True
        spans[first_non_code_idx] = (modified_seg, False)

    # Process closing pleasantry on the last text span
    last_non_code_idx = next(
        (i for i in range(len(spans) - 1, -1, -1) if not spans[i][1]), None
    )
    if last_non_code_idx is not None:
        seg_content, _ = spans[last_non_code_idx]
        modified_seg = seg_content.strip()
        changed = True
        while changed:
            changed = False
            for pattern in PLEASANTRY_CLOSING_PATTERNS:
                new_seg = pattern.sub("", modified_seg).strip()
                if new_seg != modified_seg:
                    modified_seg = new_seg
                    changed = True
        spans[last_non_code_idx] = (modified_seg, False)

    # Reconstruct text
    reconstructed = "".join(content for content, _ in spans).strip()

    # Degenerate case guard: if message became completely empty, retain original
    if not reconstructed:
        return text.strip()

    return reconstructed


def strip_reasoning_blocks(text: Optional[str]) -> str:
    """Remove thought and reasoning tags (<thought>, <reasoning>, etc.) from text."""
    if not text:
        return text or ""

    result = text
    for pattern in REASONING_BLOCK_PATTERNS:
        result = pattern.sub("", result)

    # Clean up excess blank lines left behind
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def compact_json_payload(raw_json: str) -> str:
    """Minify JSON payload by removing whitespace and indentations.
    
    If raw_json is not valid JSON, returns the original string safely.
    """
    if not raw_json or not raw_json.strip():
        return raw_json

    try:
        parsed = json.loads(raw_json)
        return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError, TypeError):
        return raw_json


def extract_critical_entities(text: Optional[str]) -> Dict[str, Set[str]]:
    """Extract paths, code blocks, and assertion errors from text."""
    if not text:
        return {"paths": set(), "assertions": set(), "code_blocks": set()}

    raw_paths = WINDOWS_PATH_PATTERN.findall(text) + POSIX_PATH_PATTERN.findall(text)
    # Strip trailing punctuation from extracted paths so sentence periods don't corrupt the path
    paths = {re.sub(r"[.,;:!?]+$", "", p) for p in raw_paths if p.strip()}
    paths = {p for p in paths if len(p) > 2}

    assertions = {a.strip() for a in ASSERTION_ERROR_PATTERN.findall(text) if a.strip()}
    code_blocks = set(CODE_FENCE_PATTERN.findall(text))

    return {
        "paths": paths,
        "assertions": assertions,
        "code_blocks": code_blocks,
    }


class SemanticValidator:
    """Validates that compression preserved critical semantic invariants."""

    @staticmethod
    def validate(
        original_text: str,
        compressed_text: str,
    ) -> Tuple[bool, List[str]]:
        """Validate invariant preservation between original and compressed text.
        
        Checks:
        1. Code fence balance and non-corruption.
        2. File path retention.
        3. Test assertion failure retention.
        """
        violations: List[str] = []

        # 1. Code Fence Check: all opened code fences in compressed must be balanced
        comp_fences = compressed_text.count("```")
        if comp_fences % 2 != 0:
            violations.append(
                f"Unbalanced code fences in compressed text (found {comp_fences} fence markers)"
            )

        # 2. File Path Invariant
        orig_entities = extract_critical_entities(original_text)
        comp_entities = extract_critical_entities(compressed_text)

        missing_paths = orig_entities["paths"] - comp_entities["paths"]
        if missing_paths:
            violations.append(f"Missing file paths after compression: {missing_paths}")

        # 3. Test Assertion Invariant
        # Check that error/assertion snippets from original are present in compressed
        for assertion in orig_entities["assertions"]:
            clean_assertion = assertion.strip()
            if clean_assertion and clean_assertion not in compressed_text:
                violations.append(
                    f"Missing assertion failure snippet: '{clean_assertion[:80]}...'"
                )

        is_valid = len(violations) == 0
        return is_valid, violations


class Compactor:
    """Hierarchical message history compactor with sliding window and anchor preservation."""

    def __init__(self, preserve_recent: int = 4) -> None:
        self.preserve_recent = preserve_recent
        self.validator = SemanticValidator()

    def _get_role(self, msg: Any) -> str:
        if isinstance(msg, dict):
            return msg.get("role", "")
        return getattr(msg, "role", "")

    def _get_content(self, msg: Any) -> Optional[str]:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if content is None:
            return None
        return str(content)

    def _set_content(self, msg: Any, new_content: Optional[str]) -> Any:
        """Return a copy of message with updated content."""
        if isinstance(msg, dict):
            copy_msg = dict(msg)
            copy_msg["content"] = new_content
            return copy_msg
        elif hasattr(msg, "model_copy"):
            return msg.model_copy(update={"content": new_content})
        else:
            copy_msg = copy.copy(msg)
            setattr(copy_msg, "content", new_content)
            return copy_msg

    def _compact_tool_calls(self, msg: Any) -> Any:
        """Minify JSON arguments inside tool_calls."""
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg
            copy_msg = dict(msg)
            new_tool_calls = []
            for tc in tool_calls:
                tc_copy = dict(tc)
                if "function" in tc_copy and isinstance(tc_copy["function"], dict):
                    fn_copy = dict(tc_copy["function"])
                    args = fn_copy.get("arguments", "")
                    if args:
                        fn_copy["arguments"] = compact_json_payload(args)
                    tc_copy["function"] = fn_copy
                new_tool_calls.append(tc_copy)
            copy_msg["tool_calls"] = new_tool_calls
            return copy_msg
        else:
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg
            new_tool_calls = []
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    fn = getattr(tc, "function")
                    args = getattr(fn, "arguments", "")
                    if args:
                        compacted_args = compact_json_payload(args)
                        if hasattr(fn, "model_copy"):
                            new_fn = fn.model_copy(update={"arguments": compacted_args})
                            new_tc = tc.model_copy(update={"function": new_fn})
                            new_tool_calls.append(new_tc)
                            continue
                new_tool_calls.append(tc)
            if hasattr(msg, "model_copy"):
                return msg.model_copy(update={"tool_calls": new_tool_calls})
            return msg

    def _summarize_middle_turns(self, middle_turns: List[Any]) -> str:
        """Condense historical middle turns into a concise structured summary.
        
        Preserves critical entities: code blocks, file paths, and assertion failures.
        """
        summary_lines = ["[Previous execution trajectory summary:]"]
        all_paths: Set[str] = set()
        all_assertions: Set[str] = set()
        all_code_blocks: Set[str] = set()

        for i, turn in enumerate(middle_turns):
            role = self._get_role(turn)
            content = self._get_content(turn) or ""

            # Extract entities
            entities = extract_critical_entities(content)
            all_paths.update(entities["paths"])
            all_assertions.update(entities["assertions"])
            all_code_blocks.update(entities["code_blocks"])

            # Check for tool calls
            tool_calls = None
            if isinstance(turn, dict):
                tool_calls = turn.get("tool_calls")
            else:
                tool_calls = getattr(turn, "tool_calls", None)

            if role == "user":
                clean_content = content.strip().replace("\n", " ")
                if len(clean_content) > 100:
                    clean_content = clean_content[:100] + "..."
                summary_lines.append(f"- User: {clean_content}")

            elif role == "assistant":
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn_name = tc.get("function", {}).get("name", "tool")
                            args = tc.get("function", {}).get("arguments", "")
                        else:
                            fn = getattr(tc, "function", None)
                            fn_name = getattr(fn, "name", "tool") if fn else "tool"
                            args = getattr(fn, "arguments", "") if fn else ""
                        compact_args = compact_json_payload(args)
                        if len(compact_args) > 60:
                            compact_args = compact_args[:60] + "..."
                        summary_lines.append(f"- Ran {fn_name}({compact_args})")
                else:
                    clean_content = strip_pleasantries(strip_reasoning_blocks(content))
                    clean_content = clean_content.replace("\n", " ")
                    if len(clean_content) > 100:
                        clean_content = clean_content[:100] + "..."
                    summary_lines.append(f"- Action: {clean_content}")

            elif role == "tool":
                first_line = content.strip().split("\n")[0] if content else ""
                if len(first_line) > 80:
                    first_line = first_line[:80] + "..."
                summary_lines.append(f"- Result: {first_line}")

        # INVARIANT GUARANTEE:
        # Re-inject any missing paths
        current_text = "\n".join(summary_lines)
        current_entities = extract_critical_entities(current_text)

        missing_paths = all_paths - current_entities["paths"]
        for path in sorted(missing_paths):
            summary_lines.append(f"  [Path: {path}]")

        # Re-inject any missing assertion errors
        for assertion in sorted(all_assertions):
            clean_assertion = assertion.strip()
            if clean_assertion and clean_assertion not in "\n".join(summary_lines):
                summary_lines.append(f"  [Failure: {clean_assertion[:200]}]")

        # Re-inject any missing code blocks
        for code_block in sorted(all_code_blocks):
            if code_block not in "\n".join(summary_lines):
                summary_lines.append(f"  {code_block}")

        return "\n".join(summary_lines)

    def compact(
        self,
        messages: List[Any],
        preserve_recent: Optional[int] = None,
    ) -> List[Any]:
        """Compact messages list applying sliding window, pruning, and summarization.
        
        Guarantees:
        1. Turn 0 (Anchor 1: System prompt) is preserved.
        2. Turn 1 (Anchor 2: Initial user prompt) is preserved.
        3. Recent K turns (default 4) are preserved verbatim.
        4. Middle turns are condensed into a structured summary turn.
        5. Invariant validation ensures zero loss of code fences, paths, or errors.
        """
        if not messages:
            return []

        k = preserve_recent if preserve_recent is not None else self.preserve_recent
        total_turns = len(messages)

        # Base case: conversation too short to compact middle turns
        if total_turns <= (2 + k):
            compacted: List[Any] = []
            for i, msg in enumerate(messages):
                role = self._get_role(msg)
                content = self._get_content(msg)

                # Strip completed assistant reasoning blocks and pleasantries
                if role == "assistant" and content:
                    # Strip reasoning tags if not the very last message
                    if i < total_turns - 1:
                        content = strip_reasoning_blocks(content)
                    content = strip_pleasantries(content)
                    msg = self._set_content(msg, content)

                msg = self._compact_tool_calls(msg)
                compacted.append(msg)
            return compacted

        # Anchors
        anchor_system = messages[0]
        anchor_user = messages[1]

        # Middle turns to condense
        middle_turns = messages[2:-k]
        # Recent buffer to preserve verbatim
        recent_buffer = messages[-k:]

        # Process middle turns
        orig_middle_text = "\n".join(
            self._get_content(m) or "" for m in middle_turns
        )
        summary_text = self._summarize_middle_turns(middle_turns)

        # Validate semantic invariants of the middle turn summary
        is_valid, violations = self.validator.validate(orig_middle_text, summary_text)

        if not is_valid:
            # Fall back to uncompacted middle turns with light stripping
            processed_middle = []
            for turn in middle_turns:
                t = self._compact_tool_calls(turn)
                role = self._get_role(t)
                content = self._get_content(t)
                if role == "assistant" and content:
                    content = strip_reasoning_blocks(strip_pleasantries(content))
                    t = self._set_content(t, content)
                processed_middle.append(t)
            middle_result = processed_middle
        else:
            # Create synthetic summary turn
            if ChatMessage is not None and isinstance(anchor_system, ChatMessage):
                summary_msg = ChatMessage(role="system", content=summary_text)
            elif isinstance(anchor_system, dict):
                summary_msg = {"role": "system", "content": summary_text}
            else:
                summary_msg = self._set_content(copy.copy(anchor_system), summary_text)
            middle_result = [summary_msg]

        # Process recent buffer: light pleasantry & reasoning cleanup on completed turns
        processed_recent: List[Any] = []
        recent_count = len(recent_buffer)
        for i, turn in enumerate(recent_buffer):
            t = self._compact_tool_calls(turn)
            role = self._get_role(t)
            content = self._get_content(t)

            if role == "assistant" and content:
                # If not the active/final turn, strip thought blocks
                if i < recent_count - 1:
                    content = strip_reasoning_blocks(content)
                content = strip_pleasantries(content)
                t = self._set_content(t, content)

            processed_recent.append(t)

        result: List[Any] = [anchor_system, anchor_user]
        if isinstance(middle_result, list):
            result.extend(middle_result)
        else:
            result.append(middle_result)
        result.extend(processed_recent)

        return result
