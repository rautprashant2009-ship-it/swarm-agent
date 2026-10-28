"""Local File Management Suite for Swarm Agent.

Provides safe, atomic, structured file operations:
- `read_file`: Line pagination, 1-based indexing, binary file detection
- `write_file`: Atomic replacement staging (temp file + os.replace), auto directory creation
- `edit_file`: Precise single-block replacement with occurrence count validation
- `list_directory`: Directory traversal with metadata, cycle guards, and safety caps
- `search_files`: Regex and substring text search with binary and ignore filters
"""

from __future__ import annotations

import datetime
import fnmatch
import hashlib
import mimetypes
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

# Default directories and file patterns to ignore during traversal
DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Max entries returned in list_directory to prevent context explosion
MAX_DIRECTORY_ENTRIES = 500

# Max matches returned in search_files
DEFAULT_MAX_MATCHES = 100


class FileReadResult(BaseModel):
    """Result of read_file operation."""

    path: str
    content: str = ""
    lines: List[str] = Field(default_factory=list)
    offset: int = 1
    limit: int = 500
    total_lines: int = 0
    has_more: bool = False
    is_binary: bool = False
    size_bytes: int = 0
    mime: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "offset": self.offset,
            "limit": self.limit,
            "total_lines": self.total_lines,
            "has_more": self.has_more,
            "is_binary": self.is_binary,
            "size_bytes": self.size_bytes,
            "mime": self.mime,
            "error": self.error,
        }


class FileWriteResult(BaseModel):
    """Result of write_file operation."""

    path: str
    bytes_written: int = 0
    sha256: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "bytes_written": self.bytes_written,
            "sha256": self.sha256,
            "success": self.success,
            "error": self.error,
        }


class FileEditResult(BaseModel):
    """Result of edit_file operation."""

    path: str
    success: bool = True
    occurrences_replaced: int = 0
    total_lines: int = 0
    diff_summary: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "success": self.success,
            "occurrences_replaced": self.occurrences_replaced,
            "total_lines": self.total_lines,
            "diff_summary": self.diff_summary,
            "error": self.error,
        }


class DirectoryEntry(BaseModel):
    """Metadata record for a directory item."""

    name: str
    path: str
    is_dir: bool
    size_bytes: int = 0
    modified_time: str = ""
    extension: str = ""


class DirectoryListResult(BaseModel):
    """Result of list_directory operation."""

    path: str
    entries: List[DirectoryEntry] = Field(default_factory=list)
    total_count: int = 0
    truncated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "entries": [entry.model_dump() for entry in self.entries],
            "total_count": self.total_count,
            "truncated": self.truncated,
            "error": self.error,
        }


class FileMatch(BaseModel):
    """Single matching line in a searched file."""

    file: str
    line: int
    text: str


class FileSearchResult(BaseModel):
    """Result of search_files operation."""

    directory: str
    query: str
    matches: List[FileMatch] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "query": self.query,
            "matches": [match.model_dump() for match in self.matches],
            "total_matches": self.total_matches,
            "truncated": self.truncated,
            "error": self.error,
        }


def is_binary_file(filepath: str) -> bool:
    """Detect whether a file is binary by scanning for NULL bytes in its first 8KB."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return False


def read_file(
    path: str,
    offset: int = 1,
    limit: int = 500,
    encoding: str = "utf-8",
) -> FileReadResult:
    """Read text content from a file with line-based pagination and binary detection.

    Args:
        path: Path to the target file (relative or absolute).
        offset: 1-based starting line number. Defaults to 1.
        limit: Maximum number of lines to return. Defaults to 500.
        encoding: File character encoding. Defaults to 'utf-8'.

    Returns:
        FileReadResult containing sliced content, line numbers, and pagination metadata.
    """
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return FileReadResult(
            path=abs_path,
            offset=offset,
            limit=limit,
            error=f"File not found: {path}",
        )

    if os.path.isdir(abs_path):
        return FileReadResult(
            path=abs_path,
            offset=offset,
            limit=limit,
            error=f"Path is a directory, not a file: {path}",
        )

    # Check for binary file
    if is_binary_file(abs_path):
        file_size = os.path.getsize(abs_path)
        mime, _ = mimetypes.guess_type(abs_path)
        return FileReadResult(
            path=abs_path,
            offset=offset,
            limit=limit,
            is_binary=True,
            size_bytes=file_size,
            mime=mime or "application/octet-stream",
            error="binary_file_detected",
        )

    try:
        with open(abs_path, "r", encoding=encoding, errors="replace") as f:
            raw_lines = f.readlines()
    except PermissionError as exc:
        return FileReadResult(
            path=abs_path,
            offset=offset,
            limit=limit,
            error=f"Permission denied: {exc}",
        )
    except Exception as exc:
        return FileReadResult(
            path=abs_path,
            offset=offset,
            limit=limit,
            error=f"Failed to read file: {exc}",
        )

    total_lines = len(raw_lines)
    effective_offset = max(1, offset)

    if effective_offset > total_lines:
        return FileReadResult(
            path=abs_path,
            content="",
            lines=[],
            offset=effective_offset,
            limit=limit,
            total_lines=total_lines,
            has_more=False,
            size_bytes=os.path.getsize(abs_path),
        )

    start_idx = effective_offset - 1
    end_idx = min(start_idx + limit, total_lines)
    sliced_raw = raw_lines[start_idx:end_idx]

    clean_lines = [line.rstrip("\r\n") for line in sliced_raw]
    formatted_lines = [
        f"{start_idx + idx + 1}: {line}" for idx, line in enumerate(clean_lines)
    ]
    content_str = "\n".join(formatted_lines)
    has_more = end_idx < total_lines

    return FileReadResult(
        path=abs_path,
        content=content_str,
        lines=clean_lines,
        offset=effective_offset,
        limit=limit,
        total_lines=total_lines,
        has_more=has_more,
        size_bytes=os.path.getsize(abs_path),
    )


def write_file(
    path: str,
    content: str,
    overwrite: bool = True,
) -> FileWriteResult:
    """Atomically write text content to a destination file.

    Follows atomic staging:
    1. Ensures parent directories exist.
    2. Writes content to temporary sibling file.
    3. Flushes and syncs to disk.
    4. Atomically replaces destination with `os.replace`.

    Args:
        path: Path to target file.
        content: String content to write.
        overwrite: If False and destination exists, writing is rejected.

    Returns:
        FileWriteResult with bytes written, sha256 checksum, and status.
    """
    abs_path = os.path.abspath(path)

    if not overwrite and os.path.exists(abs_path):
        raise FileExistsError(f"File already exists and overwrite is False: {path}")

    # Auto-create intermediate parent directories
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = f"{abs_path}.tmp_{uuid.uuid4().hex}"
    content_bytes = content.encode("utf-8")
    sha256_hash = hashlib.sha256(content_bytes).hexdigest()

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, abs_path)
    except PermissionError as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return FileWriteResult(
            path=abs_path,
            bytes_written=0,
            sha256="",
            success=False,
            error=f"Permission denied / file locked: {exc}",
        )
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return FileWriteResult(
            path=abs_path,
            bytes_written=0,
            sha256="",
            success=False,
            error=f"Failed to write file: {exc}",
        )

    return FileWriteResult(
        path=abs_path,
        bytes_written=len(content_bytes),
        sha256=sha256_hash,
        success=True,
    )


def edit_file(
    path: str,
    target_content: str,
    replacement_content: str,
    allow_multiple: bool = False,
) -> FileEditResult:
    """Perform precise single-block replacement within a file with occurrence validation.

    Args:
        path: Path to the target file.
        target_content: The exact string block to replace.
        replacement_content: The new text to insert.
        allow_multiple: If True, replaces all occurrences. If False, requires exactly 1 match.

    Returns:
        FileEditResult detailing modified lines and occurrence count.

    Raises:
        FileNotFoundError: If target file does not exist.
        ValueError: If target_content is not found or matches multiple times without flag.
    """
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        return FileEditResult(
            path=abs_path,
            success=False,
            error=f"Failed to read file for editing: {exc}",
        )

    count = content.count(target_content)

    if count == 0:
        raise ValueError(f"target_content not found in file: {path}")

    if count > 1 and not allow_multiple:
        raise ValueError(
            f"target_content matches {count} occurrences in {path}. "
            "Specific replacement requires unique match or allow_multiple=True."
        )

    # Apply replacement
    if allow_multiple:
        new_content = content.replace(target_content, replacement_content)
        occurrences = count
    else:
        new_content = content.replace(target_content, replacement_content, 1)
        occurrences = 1

    # Atomically write back
    write_res = write_file(abs_path, new_content, overwrite=True)
    if not write_res.success:
        return FileEditResult(
            path=abs_path,
            success=False,
            error=write_res.error,
        )

    total_lines = len(new_content.splitlines())
    diff_summary = f"Replaced {occurrences} occurrence(s) in {os.path.basename(abs_path)}"

    return FileEditResult(
        path=abs_path,
        success=True,
        occurrences_replaced=occurrences,
        total_lines=total_lines,
        diff_summary=diff_summary,
    )


def list_directory(
    path: str,
    recursive: bool = False,
    max_depth: int = 2,
    show_hidden: bool = False,
) -> DirectoryListResult:
    """List directory entries with metadata, cycle guards, and safety caps.

    Args:
        path: Path to directory to scan.
        recursive: Whether to traverse subdirectories. Default: False.
        max_depth: Maximum recursive depth. Default: 2.
        show_hidden: Whether to include hidden files (prefixed with '.'). Default: False.

    Returns:
        DirectoryListResult with file metadata entries.
    """
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return DirectoryListResult(
            path=abs_path,
            entries=[],
            total_count=0,
            error=f"Directory not found: {path}",
        )

    if not os.path.isdir(abs_path):
        return DirectoryListResult(
            path=abs_path,
            entries=[],
            total_count=0,
            error=f"Path is not a directory: {path}",
        )

    entries: List[DirectoryEntry] = []
    visited_inodes: Set[Tuple[int, int]] = set()
    truncated = False

    effective_max_depth = max_depth if recursive else 1

    # Breadth-first traversal with depth tracking
    queue: List[Tuple[str, int]] = [(abs_path, 0)]

    while queue and len(entries) < MAX_DIRECTORY_ENTRIES:
        curr_dir, curr_depth = queue.pop(0)

        try:
            stat_info = os.stat(curr_dir)
            inode_key = (stat_info.st_dev, stat_info.st_ino)
            if inode_key in visited_inodes:
                continue
            visited_inodes.add(inode_key)
        except Exception:
            pass

        try:
            with os.scandir(curr_dir) as scanner:
                dir_entries = list(scanner)
        except PermissionError:
            continue
        except Exception:
            continue

        for item in sorted(dir_entries, key=lambda x: (not x.is_dir(), x.name.lower())):
            name = item.name

            # Skip hidden files if show_hidden is False
            if not show_hidden and name.startswith("."):
                continue

            # Skip ignored directories
            if item.is_dir() and name in DEFAULT_IGNORED_DIRS:
                continue

            item_path = os.path.abspath(item.path)
            is_dir = False
            size_bytes = 0
            mod_time_str = ""
            ext = ""

            try:
                st = item.stat()
                is_dir = item.is_dir()
                size_bytes = st.st_size if not is_dir else 0
                mod_time_str = datetime.datetime.fromtimestamp(
                    st.st_mtime, tz=datetime.timezone.utc
                ).isoformat()
                if not is_dir:
                    ext = os.path.splitext(name)[1].lower()
            except Exception:
                pass

            entries.append(
                DirectoryEntry(
                    name=name,
                    path=item_path,
                    is_dir=is_dir,
                    size_bytes=size_bytes,
                    modified_time=mod_time_str,
                    extension=ext,
                )
            )

            if len(entries) >= MAX_DIRECTORY_ENTRIES:
                truncated = True
                break

            # Queue subdirectory for deeper recursion if within depth
            if is_dir and recursive and curr_depth + 1 < effective_max_depth:
                queue.append((item_path, curr_depth + 1))

    return DirectoryListResult(
        path=abs_path,
        entries=entries,
        total_count=len(entries),
        truncated=truncated,
    )


def search_files(
    directory: str,
    query: str,
    is_regex: bool = False,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    max_matches: int = DEFAULT_MAX_MATCHES,
) -> FileSearchResult:
    """Search for text or regular expression patterns across files in a directory tree.

    Args:
        directory: Root directory to search.
        query: Search string or regex pattern.
        is_regex: If True, compiles query as a regular expression. Default: False.
        file_pattern: Glob pattern for filtering filenames (e.g. '*.py'). Default: '*'.
        case_sensitive: If False, ignores case. Default: False.
        max_matches: Maximum number of matches to return. Default: 100.

    Returns:
        FileSearchResult with file, line number, and text snippet for each match.
    """
    abs_dir = os.path.abspath(directory)

    if not os.path.exists(abs_dir):
        return FileSearchResult(
            directory=abs_dir,
            query=query,
            matches=[],
            total_matches=0,
            error=f"Directory not found: {directory}",
        )

    if not os.path.isdir(abs_dir):
        return FileSearchResult(
            directory=abs_dir,
            query=query,
            matches=[],
            total_matches=0,
            error=f"Path is not a directory: {directory}",
        )

    # Compile regex if requested
    compiled_re: Optional[re.Pattern] = None
    if is_regex:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled_re = re.compile(query, flags)
        except re.error as exc:
            return FileSearchResult(
                directory=abs_dir,
                query=query,
                matches=[],
                total_matches=0,
                error=f"Invalid regular expression: {exc}",
            )
    else:
        needle = query if case_sensitive else query.lower()

    matches: List[FileMatch] = []
    truncated = False

    for root, dirs, files in os.walk(abs_dir):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORED_DIRS and not d.startswith(".")]

        for fname in files:
            if not fnmatch.fnmatch(fname, file_pattern):
                continue

            file_path = os.path.join(root, fname)

            # Skip binary files
            if is_binary_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, start=1):
                        match_found = False
                        if compiled_re:
                            match_found = bool(compiled_re.search(line))
                        else:
                            haystack = line if case_sensitive else line.lower()
                            match_found = needle in haystack

                        if match_found:
                            matches.append(
                                FileMatch(
                                    file=file_path,
                                    line=line_num,
                                    text=line.strip(),
                                )
                            )
                            if len(matches) >= max_matches:
                                truncated = True
                                break
            except PermissionError:
                continue
            except Exception:
                continue

            if truncated:
                break
        if truncated:
            break

    return FileSearchResult(
        directory=abs_dir,
        query=query,
        matches=matches,
        total_matches=len(matches),
        truncated=truncated,
    )
