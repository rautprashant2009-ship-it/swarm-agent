"""Automated verification test suite for Milestone 3 (R3):
Native System Execution & Agentic Tool Suite.

Tests cover Tiers 1-4:
- Tier 1: Sanity & Happy Paths (PowerShell, File Ops, Runner, Schemas, ToolExecutor)
- Tier 2: Boundaries & Edge Cases (UTF-8/Unicode, Pagination, Binary detection, Multiple edits)
- Tier 3: Fault Injection & Timeouts (Timeout guard 124, Missing target edit, Invalid cwd)
- Tier 4: Integration & Tool Dispatch (OpenAI tool schema mapping, Full lifecycle)
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from swarm.tools import (
    DEFAULT_IGNORED_DIRS,
    DirectoryEntry,
    DirectoryListResult,
    FileEditResult,
    FileReadResult,
    FileSearchResult,
    FileWriteResult,
    PowerShellBridge,
    ProcessResult,
    PythonTestRunner,
    TestFailureDetail,
    TestRunResult,
    ToolExecutor,
    TOOL_SCHEMAS,
    discover_python,
    edit_file,
    execute_tool_call,
    get_tool_map,
    get_tool_schemas,
    is_admin,
    is_binary_file,
    kill_process_tree,
    list_directory,
    read_file,
    run_command,
    run_powershell,
    run_python_tests,
    search_files,
    strip_ansi,
    write_file,
)

# Prevent pytest from treating dataclasses/models starting with Test* as test cases
TestFailureDetail.__test__ = False
TestRunResult.__test__ = False


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """Provides a fresh isolated directory for filesystem testing."""
    sb = tmp_path / "sandbox"
    sb.mkdir(parents=True, exist_ok=True)
    return sb


# ==============================================================================
# Tier 1: Sanity & Happy Paths
# ==============================================================================

def test_powershell_clean_command_execution():
    """Tier 1: Execute a clean PowerShell command and verify exit code 0 and stdout."""
    res = run_powershell('Write-Output "SWARM_ACTIVE"')
    assert res.is_success is True
    assert res.exit_code == 0
    assert "SWARM_ACTIVE" in res.stdout.strip()
    assert not res.timed_out
    assert res.duration_ms >= 0


def test_run_command_wrapper():
    """Tier 1: Verify run_command tool interface produces expected ProcessResult."""
    res = run_command('Write-Output "TOOL_RUN"')
    assert res.exit_code == 0
    assert "TOOL_RUN" in res.stdout


def test_admin_privilege_detection():
    """Tier 1: Verify is_admin returns a boolean without throwing exceptions."""
    result = is_admin()
    assert isinstance(result, bool)

    bridge = PowerShellBridge()
    assert isinstance(bridge.is_admin, bool)


def test_file_write_and_read_basic(sandbox_dir: Path):
    """Tier 1: Basic write_file followed by read_file."""
    target_file = str(sandbox_dir / "sample.txt")
    write_res = write_file(target_file, "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\n")
    assert write_res.success is True
    assert write_res.bytes_written > 0
    assert len(write_res.sha256) == 64

    read_res = read_file(target_file)
    assert read_res.error is None
    assert read_res.total_lines == 3
    assert "Line 1: Alpha" in read_res.content
    assert len(read_res.lines) == 3
    assert not read_res.is_binary


def test_file_edit_basic(sandbox_dir: Path):
    """Tier 1: Basic edit_file single block replacement."""
    target_file = str(sandbox_dir / "code.py")
    write_file(target_file, "def value():\n    return 10\n")

    edit_res = edit_file(
        path=target_file,
        target_content="    return 10",
        replacement_content="    return 42",
    )
    assert edit_res.success is True
    assert edit_res.occurrences_replaced == 1

    read_res = read_file(target_file)
    assert "return 42" in read_res.content
    assert "return 10" not in read_res.content


def test_list_directory_basic(sandbox_dir: Path):
    """Tier 1: list_directory returns entries with metadata."""
    f1 = str(sandbox_dir / "file1.txt")
    f2 = str(sandbox_dir / "file2.py")
    write_file(f1, "hello")
    write_file(f2, "print('hi')")

    list_res = list_directory(str(sandbox_dir))
    assert list_res.error is None
    assert list_res.total_count >= 2
    names = [e.name for e in list_res.entries]
    assert "file1.txt" in names
    assert "file2.py" in names


def test_search_files_basic(sandbox_dir: Path):
    """Tier 1: search_files finds matching text in files."""
    f1 = str(sandbox_dir / "mod1.py")
    f2 = str(sandbox_dir / "mod2.py")
    write_file(f1, "SECRET_TOKEN = 'xyz123'")
    write_file(f2, "PUBLIC_DATA = 'abc456'")

    search_res = search_files(str(sandbox_dir), query="SECRET_TOKEN")
    assert search_res.total_matches == 1
    assert search_res.matches[0].line == 1
    assert "SECRET_TOKEN" in search_res.matches[0].text


def test_discover_python_interpreter():
    """Tier 1: discover_python returns a valid path to an executable."""
    py_bin = discover_python()
    assert py_bin != ""
    assert os.path.exists(py_bin) or py_bin in ("python", "python3")


def test_tool_schemas_and_map():
    """Tier 1: Verify all 7 OpenAI tool schemas are registered and valid."""
    schemas = get_tool_schemas()
    assert len(schemas) == 7
    expected_names = {
        "run_command",
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "run_python_tests",
    }
    tool_names = {s["function"]["name"] for s in schemas}
    assert tool_names == expected_names

    tool_map = get_tool_map()
    assert set(tool_map.keys()) == expected_names


def test_tool_executor_facade(sandbox_dir: Path):
    """Tier 1: Verify ToolExecutor interface contract matches PROJECT.md."""
    # Test class-level methods
    ps_res = ToolExecutor.run_powershell('Write-Output "FACADE_OK"')
    assert ps_res.exit_code == 0
    assert "FACADE_OK" in ps_res.stdout

    test_file = str(sandbox_dir / "facade_test.txt")
    ToolExecutor.atomic_write_file(test_file, "Original text")
    assert os.path.exists(test_file)

    edit_ok = ToolExecutor.edit_file(test_file, "Original", "Modified")
    assert edit_ok is True

    read_back = read_file(test_file)
    assert "Modified text" in read_back.content


# ==============================================================================
# Tier 2: Boundaries & Edge Cases
# ==============================================================================

def test_powershell_utf8_and_unicode_handling():
    """Tier 2: PowerShell UTF-8 bridge correctly handles non-ASCII and Emoji glyphs."""
    unicode_str = "你好世界_Swarm_🚀_123"
    cmd = f'Write-Output "{unicode_str}"'
    res = run_powershell(cmd)
    assert res.exit_code == 0
    assert unicode_str in res.stdout


def test_read_file_line_pagination(sandbox_dir: Path):
    """Tier 2: read_file handles 1-based offset, limits, and has_more flag."""
    target_file = str(sandbox_dir / "multiline.txt")
    lines = [f"Line {i:02d}" for i in range(1, 11)]
    write_file(target_file, "\n".join(lines) + "\n")

    # Read page 1: lines 1-4
    page1 = read_file(target_file, offset=1, limit=4)
    assert page1.total_lines == 10
    assert len(page1.lines) == 4
    assert page1.has_more is True
    assert "Line 01" in page1.lines[0]
    assert "Line 04" in page1.lines[3]

    # Read page 2: lines 5-10
    page2 = read_file(target_file, offset=5, limit=10)
    assert len(page2.lines) == 6
    assert page2.has_more is False
    assert "Line 05" in page2.lines[0]
    assert "Line 10" in page2.lines[5]

    # Offset beyond EOF
    page_empty = read_file(target_file, offset=50, limit=10)
    assert len(page_empty.lines) == 0
    assert page_empty.has_more is False


def test_binary_file_protection(sandbox_dir: Path):
    """Tier 2: is_binary_file and read_file detect binary files and protect context."""
    bin_file = str(sandbox_dir / "binary.bin")
    with open(bin_file, "wb") as f:
        f.write(b"PNG\x00\x00\x00\rIHDR\x00\x00\x01\x00")

    assert is_binary_file(bin_file) is True

    res = read_file(bin_file)
    assert res.is_binary is True
    assert res.content == ""
    assert res.size_bytes > 0
    assert res.error == "binary_file_detected"


def test_write_file_auto_creates_nested_directories(sandbox_dir: Path):
    """Tier 2: write_file automatically creates deeply nested parent directories."""
    nested_file = str(sandbox_dir / "a" / "b" / "c" / "d" / "target.py")
    res = write_file(nested_file, "print('deeply nested')")
    assert res.success is True
    assert os.path.exists(nested_file)


def test_write_file_overwrite_refusal(sandbox_dir: Path):
    """Tier 2: write_file raises FileExistsError when overwrite=False and file exists."""
    fpath = str(sandbox_dir / "keep_me.txt")
    write_file(fpath, "Initial content")

    with pytest.raises(FileExistsError) as exc_info:
        write_file(fpath, "New content", overwrite=False)
    assert "already exists" in str(exc_info.value)

    # Content unchanged
    check = read_file(fpath)
    assert check.lines[0] == "Initial content"


def test_edit_file_multiple_occurrences(sandbox_dir: Path):
    """Tier 2: edit_file requires allow_multiple=True when target matches > 1 time."""
    target_file = str(sandbox_dir / "multi_match.txt")
    write_file(target_file, "repeat\nrepeat\nrepeat\n")

    # Failing case: multiple occurrences without flag raises ValueError
    with pytest.raises(ValueError) as exc_info:
        edit_file(
            path=target_file,
            target_content="repeat",
            replacement_content="once",
            allow_multiple=False,
        )
    assert "matches 3 occurrences" in str(exc_info.value)

    # Success case: allow_multiple=True
    res_ok = edit_file(
        path=target_file,
        target_content="repeat",
        replacement_content="once",
        allow_multiple=True,
    )
    assert res_ok.success is True
    assert res_ok.occurrences_replaced == 3

    check = read_file(target_file)
    assert check.lines == ["once", "once", "once"]


def test_list_directory_recursive_and_depth(sandbox_dir: Path):
    """Tier 2: list_directory respects max_depth and ignores default directories."""
    # Create structure: sandbox/sub1/sub2/sub3
    d1 = sandbox_dir / "sub1"
    d2 = d1 / "sub2"
    d3 = d2 / "sub3"
    git_dir = sandbox_dir / ".git"

    d3.mkdir(parents=True)
    git_dir.mkdir()

    write_file(str(d1 / "f1.txt"), "f1")
    write_file(str(d2 / "f2.txt"), "f2")
    write_file(str(d3 / "f3.txt"), "f3")
    write_file(str(git_dir / "config"), "git config")

    # Depth 2: should see sub1 and f1.txt, but not sub2/f2.txt
    res_d2 = list_directory(str(sandbox_dir), recursive=True, max_depth=2)
    paths_d2 = [e.path for e in res_d2.entries]
    assert any("f1.txt" in p for p in paths_d2)
    assert not any("f2.txt" in p for p in paths_d2)
    assert not any(".git" in p for p in paths_d2)


def test_search_files_regex(sandbox_dir: Path):
    """Tier 2: search_files with regex pattern and file pattern filter."""
    f1 = str(sandbox_dir / "service.py")
    f2 = str(sandbox_dir / "service.log")
    write_file(f1, "port = 8080\nhost = '127.0.0.1'")
    write_file(f2, "port = 9090 in log")

    # Regex search for port assignment matching only .py files
    res = search_files(
        directory=str(sandbox_dir),
        query=r"port\s*=\s*\d+",
        is_regex=True,
        file_pattern="*.py",
    )
    assert res.total_matches == 1
    assert "8080" in res.matches[0].text


# ==============================================================================
# Tier 3: Fault Injection & Timeouts
# ==============================================================================

def test_powershell_timeout_handling():
    """Tier 3: Long-running command terminates at timeout and returns exit code 124."""
    # Sleep 10s with timeout=1s
    res = run_powershell("Start-Sleep -Seconds 10", timeout=1)
    assert res.timed_out is True
    assert res.exit_code == 124
    assert not res.is_success


def test_powershell_invalid_cwd():
    """Tier 3: Invalid working directory returns exit code 1 with clean error."""
    res = run_powershell('Write-Output "test"', cwd="C:\\invalid_dir_does_not_exist_xyz123")
    assert res.exit_code == 1
    assert "Directory does not exist" in res.stderr


def test_powershell_command_syntax_error():
    """Tier 3: Syntax error in command produces non-zero exit code and captures stderr."""
    res = run_powershell("throw 'FORCED_CRITICAL_ERROR'")
    assert res.exit_code != 0
    assert "FORCED_CRITICAL_ERROR" in res.stderr or "FORCED_CRITICAL_ERROR" in res.stdout


def test_edit_file_target_not_found(sandbox_dir: Path):
    """Tier 3: edit_file raises ValueError when target_content does not exist."""
    fpath = str(sandbox_dir / "immutable.txt")
    write_file(fpath, "Original data")

    with pytest.raises(ValueError) as exc_info:
        edit_file(fpath, target_content="Missing String", replacement_content="New")
    assert "not found" in str(exc_info.value)


def test_python_test_runner_structured_failure_parsing(sandbox_dir: Path):
    """Tier 3: Subprocess test runner executes failing tests and parses failures into structured detail."""
    test_code = (
        "def test_broken_math():\n"
        "    expected = 42\n"
        "    actual = 10\n"
        "    assert actual == expected, 'Math logic failure'\n"
    )
    test_file = str(sandbox_dir / "test_sample_broken.py")
    write_file(test_file, test_code)

    runner = PythonTestRunner(workspace_root=str(sandbox_dir))
    res = runner.run_tests(test_target=test_file)

    assert res.exit_code != 0
    assert not res.is_success
    assert res.failed >= 1
    assert len(res.failures) >= 1

    failure = res.failures[0]
    assert "test_broken_math" in failure.test_name
    assert "AssertionError" in failure.failure_type
    assert "Math logic failure" in failure.message or "10 == 42" in failure.message


# ==============================================================================
# Tier 4: Integration & Tool Dispatch Lifecycle
# ==============================================================================

def test_execute_tool_call_lifecycle(sandbox_dir: Path):
    """Tier 4: Test full tool lifecycle through OpenAI function calling dispatch interface."""
    target_file = str(sandbox_dir / "lifecycle.txt")

    # 1. write_file via dispatch
    res_w = execute_tool_call("write_file", {"path": target_file, "content": "Lifecycle turn 1"})
    assert res_w["success"] is True

    # 2. read_file via dispatch
    res_r = execute_tool_call("read_file", {"path": target_file})
    assert "Lifecycle turn 1" in res_r["content"]

    # 3. edit_file via dispatch
    res_e = execute_tool_call(
        "edit_file",
        {
            "path": target_file,
            "target_content": "turn 1",
            "replacement_content": "turn 2 completed",
        },
    )
    assert res_e["success"] is True

    # 4. search_files via dispatch
    res_s = execute_tool_call(
        "search_files",
        {"directory": str(sandbox_dir), "query": "turn 2"},
    )
    assert res_s["total_matches"] == 1

    # 5. run_command via dispatch
    res_c = execute_tool_call("run_command", {"command": 'Write-Output "DISPATCH_PROBE"'})
    assert "DISPATCH_PROBE" in res_c["stdout"]
