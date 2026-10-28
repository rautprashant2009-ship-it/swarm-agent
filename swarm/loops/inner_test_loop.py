"""Tier 1: Inner Local Test Loop (Physical Test Execution & Fast Self-Repair).

Physical code execution loop that runs real local tests on the host machine via ToolExecutor /
PythonTestRunner, deterministically parses failures and tracebacks, packages targeted diagnostics,
and autonomously loops with the Coder model (Qwen 3.8 / swarm-coder) to iteratively patch bugs
until clean exit code 0 is achieved, protected by a hard 5-attempt budget against infinite loops.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from swarm.tools.runner import (
    PythonTestRunner,
    TestFailureDetail,
    TestRunResult,
    run_python_tests,
)
from swarm.tools.files import read_file, write_file

# Mark test models to prevent pytest test collector warnings
TestFailureDetail.__test__ = False
TestRunResult.__test__ = False

logger = logging.getLogger("swarm.loops.inner_test_loop")


class InnerLoopResult(BaseModel):
    """Result of an Inner Test Loop convergence session.

    Supports both attribute access (result.status) and dictionary access (result['status'])
    for backward compatibility with test harnesses and caller protocols.
    """

    status: Literal["CONVERGED", "BUDGET_EXCEEDED", "FAILED"]
    final_exit_code: int
    attempts_used: int
    last_result: Optional[TestRunResult] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class InnerLoopState(BaseModel):
    """Persistent execution state for an inner test loop task."""

    task_id: str
    current_attempt: int = 1
    max_attempts: int = 5
    target_files: List[str] = Field(default_factory=list)
    test_command: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    status: Literal["IN_PROGRESS", "CONVERGED", "BUDGET_EXCEEDED"] = "IN_PROGRESS"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class InnerTestLoop:
    """Tier 1 Inner Local Test Loop execution engine.

    Executes physical test suites on the host machine, captures structured failure diagnostics,
    re-prompts the coder persona, and loops iteratively until clean exit code 0 or budget exhaustion.
    """

    DEFAULT_MAX_ATTEMPTS = 5

    def __init__(
        self,
        working_dir: str = ".",
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        router: Optional[Any] = None,
    ) -> None:
        self.working_dir = os.path.abspath(working_dir)
        self.max_attempts = max_attempts
        self.router = router
        self.runner = PythonTestRunner(workspace_root=self.working_dir)

    def format_self_repair_prompt(
        self,
        target_file: str,
        failing_test_name: str,
        failure_line: Optional[int],
        exit_code: int,
        attempt: int,
        max_attempts: int,
        traceback_snippet: str,
        stdout_stderr: str,
        current_code: str,
    ) -> str:
        """Construct the standardized self-repair prompt for the Coder persona."""
        line_str = f":{failure_line}" if failure_line else ""
        return (
            f"[TIER 1 LOCAL TEST RUNNER FAILURE ALERT]\n"
            f"Target Test: {failing_test_name}\n"
            f"Target Source File: {target_file}{line_str}\n"
            f"Exit Code: {exit_code}\n"
            f"Attempt: {attempt} of {max_attempts}\n\n"
            f"Captured Traceback:\n"
            f"```{traceback_snippet}```\n\n"
            f"Physical Pytest Output:\n"
            f"```{stdout_stderr}```\n\n"
            f"Current Source Code:\n"
            f"```{current_code}```\n\n"
            f"INSTRUCTION FOR CODER:\n"
            f"The local test suite failed physically on the host machine.\n"
            f"Analyze the exact assertion failure and traceback above.\n"
            f"Provide a minimal, targeted patch that fixes the bug without introducing regressions.\n"
            f"Output ONLY the replacement code or diff block."
        )

    def _resolve_test_target(
        self,
        test_file: Optional[str] = None,
        test_command: Optional[str] = None,
        target_file: Optional[str] = None,
    ) -> str:
        """Resolve the appropriate test target path from test_file or test_command."""
        if test_file:
            return test_file
        if test_command:
            # Parse test file from command like "pytest tests/test_calc.py -v"
            parts = test_command.strip().split()
            for part in parts:
                if (part.endswith(".py") or "::" in part or "/" in part or "\\" in part) and not part.startswith("-") and not part.lower() in ("pytest", "python", "-m"):
                    return part
            # Fallback to test_command if no path extracted
            return test_command
        if target_file:
            # If target_file itself looks like a test file, run it
            if "test_" in os.path.basename(target_file):
                return target_file
        return "tests"

    def _call_patch_generator(
        self,
        patch_generator: Callable[..., str],
        attempt: int,
        run_res: TestRunResult,
        target_file: str,
        current_code: str,
    ) -> str:
        """Invokes the patch generator with adaptive argument signature inspection."""
        failures = run_res.failures
        stdout_stderr = (run_res.stdout + "\n" + run_res.stderr).strip()

        # Try inspect signature
        try:
            sig = inspect.signature(patch_generator)
            param_count = len(sig.parameters)
        except Exception:
            param_count = 2

        if param_count == 1:
            return patch_generator(attempt)
        elif param_count == 2:
            try:
                # Try passing failures list first (test_convergence_loop.py pattern)
                return patch_generator(attempt, failures)
            except TypeError:
                # Fallback to passing raw error trace string (analysis.md pattern)
                return patch_generator(attempt, stdout_stderr)
        else:
            # 3 or more parameters: pass attempt, failures, current_code
            try:
                return patch_generator(attempt, failures, current_code)
            except TypeError:
                return patch_generator(attempt, stdout_stderr)

    def run_until_convergence(
        self,
        target_file: str,
        test_file: Optional[str] = None,
        test_command: Optional[str] = None,
        patch_generator: Optional[Callable[..., str]] = None,
        timeout: int = 120,
    ) -> InnerLoopResult:
        """Run the test loop iteratively until clean exit code 0 or budget exhaustion.

        Args:
            target_file: Path to the source file being tested and patched.
            test_file: Path to the test file to execute.
            test_command: Shell test command string (used if test_file is not given).
            patch_generator: Callback providing patched source code for each iteration.
            timeout: Subprocess timeout in seconds.

        Returns:
            InnerLoopResult with status ('CONVERGED' or 'BUDGET_EXCEEDED'), attempts_used,
            final_exit_code, and last_result.
        """
        test_target = self._resolve_test_target(test_file, test_command, target_file)
        history: List[Dict[str, Any]] = []

        logger.info(
            f"[Tier 1] Starting inner test loop on target='{target_file}', "
            f"test='{test_target}', max_attempts={self.max_attempts}"
        )

        for attempt in range(1, self.max_attempts + 1):
            run_res = self.runner.run_tests(test_target=test_target, timeout=timeout)

            attempt_record = {
                "attempt": attempt,
                "exit_code": run_res.exit_code,
                "passed": run_res.passed,
                "failed": run_res.failed,
                "total": run_res.total,
                "failure_count": len(run_res.failures),
                "duration_ms": run_res.duration_ms,
            }
            history.append(attempt_record)

            logger.info(
                f"[Tier 1 Attempt {attempt}/{self.max_attempts}] Exit code: {run_res.exit_code}, "
                f"Passed: {run_res.passed}, Failed: {run_res.failed}"
            )

            # Check if clean convergence achieved
            if run_res.is_success and run_res.exit_code == 0:
                logger.info(f"[Tier 1] Converged successfully on attempt {attempt}.")
                return InnerLoopResult(
                    status="CONVERGED",
                    final_exit_code=0,
                    attempts_used=attempt,
                    last_result=run_res,
                    history=history,
                )

            # If budget exhausted, do not attempt to generate another patch
            if attempt >= self.max_attempts:
                break

            # Read current source code
            current_code = ""
            if os.path.exists(target_file):
                read_res = read_file(target_file, limit=10000)
                current_code = read_res.content if read_res.is_success else ""

            # Synthesize patch
            new_code: Optional[str] = None
            if patch_generator is not None:
                new_code = self._call_patch_generator(
                    patch_generator=patch_generator,
                    attempt=attempt,
                    run_res=run_res,
                    target_file=target_file,
                    current_code=current_code,
                )
            elif self.router is not None:
                # Dispatch to swarm-coder (Qwen 3.8 persona) via router
                failing_name = run_res.failures[0].test_name if run_res.failures else "unknown_test"
                failing_line = run_res.failures[0].line_number if run_res.failures else None
                traceback_snip = run_res.failures[0].traceback_snippet if run_res.failures else ""
                prompt = self.format_self_repair_prompt(
                    target_file=target_file,
                    failing_test_name=failing_name,
                    failure_line=failing_line,
                    exit_code=run_res.exit_code,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    traceback_snippet=traceback_snip,
                    stdout_stderr=(run_res.stdout + "\n" + run_res.stderr).strip(),
                    current_code=current_code,
                )
                try:
                    from swarm.core.models import ChatMessage
                    response = self.router.dispatch_role(
                        role="swarm-coder",
                        messages=[ChatMessage(role="user", content=prompt)],
                    )
                    content = response.choices[0].message.content or ""
                    # Strip markdown code blocks if wrapped
                    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
                    new_code = code_match.group(1) if code_match else content
                except Exception as e:
                    logger.warning(f"[Tier 1] Router coder dispatch failed: {e}")

            if new_code is not None:
                write_file(target_file, new_code, overwrite=True)
            else:
                logger.warning(f"[Tier 1] No patch generated on attempt {attempt}.")

        # Final verification attempt
        final_res = self.runner.run_tests(test_target=test_target, timeout=timeout)
        if final_res.is_success and final_res.exit_code == 0:
            logger.info(f"[Tier 1] Converged on final check with {self.max_attempts} attempts.")
            return InnerLoopResult(
                status="CONVERGED",
                final_exit_code=0,
                attempts_used=self.max_attempts,
                last_result=final_res,
                history=history,
            )

        logger.warning(
            f"[Tier 1] Attempt budget ({self.max_attempts}) exhausted. Final exit code: {final_res.exit_code}"
        )
        return InnerLoopResult(
            status="BUDGET_EXCEEDED",
            final_exit_code=final_res.exit_code,
            attempts_used=self.max_attempts,
            last_result=final_res,
            history=history,
        )


# Class alias for interface contracts and reference harness compatibility
InnerLoopRunner = InnerTestLoop

__all__ = [
    "InnerTestLoop",
    "InnerLoopRunner",
    "InnerLoopResult",
    "InnerLoopState",
]
