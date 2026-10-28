"""Native System Execution & Agentic Tool Suite (Milestone 3 / R3).

Provides:
- PowerShell Execution Bridge (`run_powershell`, `run_command`, `is_admin`, `PowerShellBridge`)
- Local File Management (`read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `is_binary_file`)
- Python Test Runner Subsystem (`run_python_tests`, `discover_python`, `PythonTestRunner`)
- OpenAI Tool Calling Schemas (`TOOL_SCHEMAS`, `get_tool_schemas`, `get_tool_map`, `execute_tool_call`)
- Unified `ToolExecutor` facade matching swarm interface contracts
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from swarm.tools.executor import (
    POWERSHELL_UTF8_PREAMBLE,
    PowerShellBridge,
    ProcessResult,
    find_powershell,
    is_admin,
    kill_process_tree,
    run_command,
    run_powershell,
    strip_ansi,
)
from swarm.tools.files import (
    DEFAULT_IGNORED_DIRS,
    DirectoryEntry,
    DirectoryListResult,
    FileEditResult,
    FileMatch,
    FileReadResult,
    FileSearchResult,
    FileWriteResult,
    edit_file,
    is_binary_file,
    list_directory,
    read_file,
    search_files,
    write_file,
)
from swarm.tools.runner import (
    PythonTestRunner,
    TestFailureDetail,
    TestRunResult,
    discover_python,
    parse_pytest_failures,
    parse_test_counts,
    parse_unittest_failures,
    run_python_tests,
)
from swarm.tools.schemas import (
    TOOL_SCHEMAS,
    execute_tool_call,
    get_tool_map,
    get_tool_schemas,
)

# Alias for interface contract compatibility with PROJECT.md
TestFailure = TestFailureDetail


class ToolExecutor:
    """Unified execution facade matching PROJECT.md interface contracts.

    Supports both class-level invocation:
        ToolExecutor.run_powershell(...)
    and instance-level invocation:
        executor = ToolExecutor(cwd="...")
        executor.run_powershell(...)
    """

    def __init__(self, cwd: Optional[str] = None) -> None:
        self.cwd = os.path.abspath(cwd) if cwd else os.getcwd()
        self.ps_bridge = PowerShellBridge(default_cwd=self.cwd)
        self.test_runner = PythonTestRunner(workspace_root=self.cwd)

    @classmethod
    def run_powershell(
        cls,
        command: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
    ) -> ProcessResult:
        """Execute a PowerShell command with UTF-8 normalization and timeout protection."""
        return run_powershell(command=command, timeout=timeout, cwd=cwd)

    @classmethod
    def run_pytest(
        cls,
        test_path: str = "tests",
        timeout: int = 120,
        test_name: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> TestRunResult:
        """Run pytest test suite and return structured failure diagnostics."""
        return run_python_tests(
            test_target=test_path,
            test_name=test_name,
            timeout=timeout,
            cwd=cwd,
        )

    @classmethod
    def atomic_write_file(cls, path: str, content: str) -> None:
        """Atomically write content to file, raising on error."""
        result = write_file(path=path, content=content, overwrite=True)
        if not result.success:
            raise IOError(f"atomic_write_file failed for {path}: {result.error}")

    @classmethod
    def edit_file(
        cls,
        path: str,
        target: str,
        replacement: str,
        allow_multiple: bool = False,
    ) -> bool:
        """Perform single-block replacement and return True on success."""
        result = edit_file(
            path=path,
            target_content=target,
            replacement_content=replacement,
            allow_multiple=allow_multiple,
        )
        return result.success

    @classmethod
    def read_file(
        cls,
        path: str,
        offset: int = 1,
        limit: int = 500,
    ) -> FileReadResult:
        """Read lines from file with pagination."""
        return read_file(path=path, offset=offset, limit=limit)

    @classmethod
    def list_directory(
        cls,
        path: str,
        recursive: bool = False,
        max_depth: int = 2,
    ) -> DirectoryListResult:
        """List files and subdirectories with metadata."""
        return list_directory(path=path, recursive=recursive, max_depth=max_depth)

    @classmethod
    def search_files(
        cls,
        directory: str,
        query: str,
        is_regex: bool = False,
        file_pattern: str = "*",
    ) -> FileSearchResult:
        """Search files in directory matching pattern."""
        return search_files(
            directory=directory,
            query=query,
            is_regex=is_regex,
            file_pattern=file_pattern,
        )

    @classmethod
    def is_admin(cls) -> bool:
        """Check whether process has elevated administrative rights."""
        return is_admin()


__all__ = [
    # PowerShell bridge
    "run_powershell",
    "run_command",
    "is_admin",
    "find_powershell",
    "kill_process_tree",
    "strip_ansi",
    "ProcessResult",
    "PowerShellBridge",
    "POWERSHELL_UTF8_PREAMBLE",
    # File operations
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "is_binary_file",
    "FileReadResult",
    "FileWriteResult",
    "FileEditResult",
    "DirectoryEntry",
    "DirectoryListResult",
    "FileMatch",
    "FileSearchResult",
    "DEFAULT_IGNORED_DIRS",
    # Test runner
    "run_python_tests",
    "discover_python",
    "parse_pytest_failures",
    "parse_unittest_failures",
    "parse_test_counts",
    "TestFailureDetail",
    "TestFailure",
    "TestRunResult",
    "PythonTestRunner",
    # Schemas
    "TOOL_SCHEMAS",
    "get_tool_schemas",
    "get_tool_map",
    "execute_tool_call",
    # Facade
    "ToolExecutor",
]
