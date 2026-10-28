"""Python Test Runner Subsystem for Swarm Agent (Tier 1 Inner Test Loop).

Provides automated test execution and structured diagnostic extraction:
- Runtime discovery (workspace .venv, active sys.executable, PATH lookup)
- Pytest and Unittest subprocess invocation with timeout protection
- Structured failure extraction: isolating test names, offending file paths,
  line numbers, exception types, assertion messages, and traceback snippets.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from swarm.tools.executor import kill_process_tree, strip_ansi


class TestFailureDetail(BaseModel):
    """Structured information extracted from a failed unit test."""

    __test__ = False  # Prevent pytest from collecting this class as a test suite

    test_name: str
    file_path: str = ""
    line_number: int = 0
    failure_type: str = "AssertionError"
    message: str = ""
    traceback_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "failure_type": self.failure_type,
            "message": self.message,
            "traceback_snippet": self.traceback_snippet,
        }


class TestRunResult(BaseModel):
    """Aggregate result from a test suite execution."""

    __test__ = False  # Prevent pytest from collecting this class as a test suite

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    failures: List[TestFailureDetail] = Field(default_factory=list)
    timed_out: bool = False
    runner: str = "pytest"

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.failed == 0

    @property
    def success(self) -> bool:
        return self.is_success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 2),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total,
            "failures": [f.to_dict() for f in self.failures],
            "timed_out": self.timed_out,
            "runner": self.runner,
            "is_success": self.is_success,
        }


def discover_python(workspace_root: Optional[str] = None) -> str:
    """Discover the most appropriate Python interpreter.

    Search priority:
    1. Workspace virtual environments (.venv or venv)
    2. Active running interpreter (`sys.executable`)
    3. System PATH lookup (`python.exe` or `python3.exe`)
    """
    candidates: List[str] = []

    if workspace_root:
        abs_root = os.path.abspath(workspace_root)
        if sys.platform == "win32":
            candidates.extend(
                [
                    os.path.join(abs_root, ".venv", "Scripts", "python.exe"),
                    os.path.join(abs_root, "venv", "Scripts", "python.exe"),
                ]
            )
        else:
            candidates.extend(
                [
                    os.path.join(abs_root, ".venv", "bin", "python"),
                    os.path.join(abs_root, "venv", "bin", "python"),
                ]
            )

    # Current running interpreter
    if sys.executable:
        candidates.append(sys.executable)

    # PATH lookups
    if sys.platform == "win32":
        path_py = shutil.which("python.exe") or shutil.which("python")
        if path_py:
            candidates.append(path_py)
    else:
        path_py3 = shutil.which("python3") or shutil.which("python")
        if path_py3:
            candidates.append(path_py3)

    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)

    return sys.executable


def parse_pytest_failures(raw_text: str) -> List[TestFailureDetail]:
    """Parse structured failure details from pytest stdout/stderr."""
    failures: List[TestFailureDetail] = []
    text = strip_ansi(raw_text)

    # 1. Parse individual failure sections
    # Format:
    # _________________________________ test_name __________________________________
    # path/to/file.py:42: in test_name
    #     assert x == y
    # E   AssertionError: assert 1 == 2
    section_pattern = re.compile(
        r"^_{3,}\s+(?P<test_name>[A-Za-z0-9_\[\]\-\.]+)\s+_{3,}\n(?P<body>.*?)(?=(?:^_{3,}|\n=+\s+FAILURES|\n=+\s+short test summary|\Z))",
        re.MULTILINE | re.DOTALL,
    )

    for match in section_pattern.finditer(text):
        test_name = match.group("test_name").strip()
        body = match.group("body")

        file_path = ""
        line_num = 0
        error_type = "AssertionError"
        error_msg = ""

        # Extract file and line number
        fl_match = re.search(r"(?P<file>[A-Za-z0-9_./\\-]+\.py):(?P<line>\d+):", body)
        if fl_match:
            file_path = fl_match.group("file").strip()
            try:
                line_num = int(fl_match.group("line"))
            except (ValueError, TypeError):
                line_num = 0

        # Extract error type and message (lines starting with 'E   ')
        e_lines = [
            line[1:].strip()
            for line in body.splitlines()
            if line.startswith("E ") or line.startswith("E\t")
        ]
        if e_lines:
            err_line = e_lines[0]
            err_match = re.match(r"(?P<type>[A-Za-z0-9_]+(?:Error|Exception)):\s*(?P<msg>.*)", err_line)
            if err_match:
                error_type = err_match.group("type").strip()
                error_msg = err_match.group("msg").strip()
            else:
                error_msg = err_line
                if len(e_lines) > 1 and not error_msg:
                    error_msg = e_lines[1]
        else:
            # Fallback search for Exception in body
            err_match = re.search(r"(?P<type>[A-Za-z0-9_]+(?:Error|Exception)):\s*(?P<msg>.*)", body)
            if err_match:
                error_type = err_match.group("type").strip()
                error_msg = err_match.group("msg").strip()

        # Extract concise traceback snippet
        body_lines = [l for l in body.splitlines() if l.strip()]
        snippet = "\n".join(body_lines[-10:]) if body_lines else body.strip()

        failures.append(
            TestFailureDetail(
                test_name=test_name,
                file_path=file_path,
                line_number=line_num,
                failure_type=error_type,
                message=error_msg,
                traceback_snippet=snippet,
            )
        )

    # 2. Also cross-check with short test summary info:
    # FAILED test_sample_fail.py::test_bad - AssertionError: Boom
    summary_failed_pattern = re.compile(
        r"^FAILED\s+(?P<file>[^:\n]+)::(?P<test>[A-Za-z0-9_\[\]\-\.]+)(?:\s+-\s+(?:(?P<type>[A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*)?(?P<msg>.*))?",
        re.MULTILINE,
    )
    for m in summary_failed_pattern.finditer(text):
        t_name = m.group("test").strip()
        f_path = m.group("file").strip()
        t_type = m.group("type") or "AssertionError"
        t_msg = (m.group("msg") or "").strip()

        existing = next((f for f in failures if f.test_name == t_name or t_name in f.test_name), None)
        if existing:
            if not existing.file_path and f_path:
                existing.file_path = f_path
            if not existing.message and t_msg:
                existing.message = t_msg
        else:
            failures.append(
                TestFailureDetail(
                    test_name=t_name,
                    file_path=f_path,
                    line_number=0,
                    failure_type=t_type,
                    message=t_msg,
                    traceback_snippet=m.group(0),
                )
            )

    # 3. Check for early SyntaxError or CollectionError
    if not failures:
        syntax_pattern = re.compile(
            r'File "(?P<file>[^"]+)", line (?P<line>\d+).*?(?P<type>SyntaxError|IndentationError):\s*(?P<msg>.*)',
            re.DOTALL,
        )
        sm = syntax_pattern.search(text)
        if sm:
            failures.append(
                TestFailureDetail(
                    test_name="<collection_failure>",
                    file_path=sm.group("file").strip(),
                    line_number=int(sm.group("line")),
                    failure_type=sm.group("type").strip(),
                    message=sm.group("msg").strip().splitlines()[0],
                    traceback_snippet=sm.group(0),
                )
            )

    return failures


def parse_unittest_failures(raw_text: str) -> List[TestFailureDetail]:
    """Parse failure details from standard library unittest output."""
    failures: List[TestFailureDetail] = []
    text = strip_ansi(raw_text)

    ut_pattern = re.compile(
        r"^(?:FAIL|ERROR):\s+(?P<test>[A-Za-z0-9_]+)\s+\((?P<suite>[^)]+)\)\n(?P<body>.*?)(?=(?:^(?:FAIL|ERROR):|\n-{5,}|\n={5,}|\Z))",
        re.MULTILINE | re.DOTALL,
    )

    for match in ut_pattern.finditer(text):
        test_name = match.group("test").strip()
        body = match.group("body")

        file_path = ""
        line_num = 0
        error_type = "AssertionError"
        error_msg = ""

        file_line_matches = re.findall(
            r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\w+)', body
        )
        if file_line_matches:
            last_fl = file_line_matches[-1]
            file_path = last_fl[0]
            try:
                line_num = int(last_fl[1])
            except ValueError:
                line_num = 0

        err_match = re.search(
            r"(?P<type>[A-Za-z0-9_]+(?:Error|Exception)):\s*(?P<msg>.*)$",
            body,
            re.MULTILINE,
        )
        if err_match:
            error_type = err_match.group("type").strip()
            error_msg = err_match.group("msg").strip()

        failures.append(
            TestFailureDetail(
                test_name=test_name,
                file_path=file_path,
                line_number=line_num,
                failure_type=error_type,
                message=error_msg,
                traceback_snippet=body.strip(),
            )
        )

    return failures


def parse_test_counts(output: str) -> Dict[str, int]:
    """Extract passed, failed, skipped, and total test counts from test output."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
    clean = strip_ansi(output)

    # 1. Check pytest summary line
    summary_match = re.search(r"=+\s+(.*?)\s+in\s+[\d.]+s\s+=+", clean)
    if summary_match:
        summary_text = summary_match.group(1)
        fm = re.search(r"(\d+)\s+failed", summary_text)
        em = re.search(r"(\d+)\s+error", summary_text)
        pm = re.search(r"(\d+)\s+passed", summary_text)
        sm = re.search(r"(\d+)\s+skipped", summary_text)

        if fm:
            counts["failed"] += int(fm.group(1))
        if em:
            counts["failed"] += int(em.group(1))
        if pm:
            counts["passed"] = int(pm.group(1))
        if sm:
            counts["skipped"] = int(sm.group(1))

        counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
        return counts

    # Fallback search anywhere in clean text
    fm = re.search(r"(\d+)\s+failed", clean)
    em = re.search(r"(\d+)\s+error", clean)
    pm = re.search(r"(\d+)\s+passed", clean)
    sm = re.search(r"(\d+)\s+skipped", clean)
    if fm or em or pm or sm:
        if fm:
            counts["failed"] += int(fm.group(1))
        if em:
            counts["failed"] += int(em.group(1))
        if pm:
            counts["passed"] = int(pm.group(1))
        if sm:
            counts["skipped"] = int(sm.group(1))
        counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
        return counts

    # 2. Check unittest summary
    ut_ran_match = re.search(r"Ran (\d+) tests?", clean)
    if ut_ran_match:
        counts["total"] = int(ut_ran_match.group(1))
        if "OK" in clean:
            counts["passed"] = counts["total"]
        else:
            fm = re.search(r"failures=(\d+)", clean)
            em = re.search(r"errors=(\d+)", clean)
            failed = (int(fm.group(1)) if fm else 0) + (int(em.group(1)) if em else 0)
            counts["failed"] = failed
            counts["passed"] = max(0, counts["total"] - failed)

    return counts


def run_python_tests(
    test_target: str = "tests",
    test_name: Optional[str] = None,
    timeout: int = 120,
    cwd: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> TestRunResult:
    """Execute unit tests and return structured pass/fail metrics and failure traces.

    Args:
        test_target: Target test directory or file path. Default: "tests".
        test_name: Optional test function or class filter.
        timeout: Subprocess timeout in seconds. Default: 120.
        cwd: Working directory to run tests in. Defaults to workspace_root or os.getcwd().
        workspace_root: Project root for python interpreter resolution.

    Returns:
        TestRunResult with exit code, counts, and structured failures.
    """
    effective_cwd = cwd or workspace_root or os.getcwd()
    python_bin = discover_python(workspace_root=effective_cwd)

    # Determine whether pytest is installed in target python environment
    has_pytest = False
    try:
        check_proc = subprocess.run(
            [python_bin, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check_proc.returncode == 0:
            has_pytest = True
    except Exception:
        has_pytest = False

    runner_name = "pytest" if has_pytest else "unittest"

    if has_pytest:
        cmd = [python_bin, "-m", "pytest", "-v", "--tb=short", test_target]
        if test_name:
            cmd.extend(["-k", test_name])
    else:
        # Fallback to unittest
        cmd = [python_bin, "-m", "unittest", "discover", "-s", test_target, "-p", "test_*.py"]

    start_time = time.perf_counter()
    proc: Optional[subprocess.Popen] = None
    stdout_text = ""
    stderr_text = ""
    timed_out = False
    exit_code = 0

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=effective_cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout_text, stderr_text = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            kill_process_tree(proc.pid)
            try:
                extra_out, extra_err = proc.communicate(timeout=2)
                if extra_out:
                    stdout_text += extra_out
                if extra_err:
                    stderr_text += extra_err
            except Exception:
                pass
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return TestRunResult(
            exit_code=1,
            stdout="",
            stderr=f"Test runner invocation error: {exc}",
            duration_ms=duration_ms,
            timed_out=False,
            runner=runner_name,
        )

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    combined_output = f"{stdout_text}\n{stderr_text}"

    # Parse counts
    counts = parse_test_counts(combined_output)

    # Parse failure details
    if has_pytest:
        failures = parse_pytest_failures(combined_output)
    else:
        failures = parse_unittest_failures(combined_output)

    # Sync failed count if parser found failures
    if failures and counts["failed"] == 0:
        counts["failed"] = len(failures)
        counts["total"] = max(counts["total"], counts["passed"] + counts["failed"])

    return TestRunResult(
        exit_code=exit_code,
        stdout=stdout_text,
        stderr=stderr_text,
        duration_ms=duration_ms,
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        total=counts["total"],
        failures=failures,
        timed_out=timed_out,
        runner=runner_name,
    )


class PythonTestRunner:
    """Stateful Python test runner holding working directory and environment context."""

    def __init__(self, workspace_root: Optional[str] = None, timeout: int = 120) -> None:
        self.workspace_root = os.path.abspath(workspace_root) if workspace_root else os.getcwd()
        self.timeout = timeout
        self.python_bin = discover_python(workspace_root=self.workspace_root)

    def run_tests(
        self,
        test_target: str = "tests",
        test_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> TestRunResult:
        """Run tests using configured environment."""
        effective_timeout = timeout or self.timeout
        return run_python_tests(
            test_target=test_target,
            test_name=test_name,
            timeout=effective_timeout,
            cwd=self.workspace_root,
            workspace_root=self.workspace_root,
        )

    def run_pytest(
        self,
        test_path: str,
        timeout: Optional[int] = None,
        test_name: Optional[str] = None,
    ) -> TestRunResult:
        """Interface method matching PROJECT.md interface contract."""
        return self.run_tests(test_target=test_path, test_name=test_name, timeout=timeout)
