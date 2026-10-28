"""PowerShell Execution Bridge for Swarm Agent (Windows 10/11 & PowerShell 5.1/7+).

Provides robust, timeout-guarded native shell execution with:
- Forced UTF-8 input/output stream normalization
- Elevated Windows Administrative privilege detection
- Process tree termination via taskkill on timeout (exit code 124)
- ANSI terminal escape sequence stripping
- Execution time tracking and detailed stream separation
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ANSI Escape sequence regex for terminal output cleaning
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")

# Standard UTF-8 preamble for PowerShell sessions
POWERSHELL_UTF8_PREAMBLE = (
    "$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8;"
)


class ProcessResult(BaseModel):
    """Execution result container for native shell commands."""

    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False

    @property
    def is_success(self) -> bool:
        """Returns True if the process completed with returncode 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def success(self) -> bool:
        """Alias for is_success."""
        return self.is_success

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 2),
            "timed_out": self.timed_out,
            "is_success": self.is_success,
        }


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    if not text:
        return ""
    return ANSI_ESCAPE_RE.sub("", text)


def is_admin() -> bool:
    """Detect whether the current process is running with elevated administrative rights.

    Uses Win32 Shell32 API token query on Windows, with fallback to PowerShell role probe.
    Returns False safely on access denied, non-Windows systems, or query failure.
    """
    if sys.platform != "win32":
        try:
            return os.geteuid() == 0  # type: ignore[attr-defined]
        except AttributeError:
            return False

    # Primary check: Win32 IsUserAnAdmin
    try:
        if ctypes.windll.shell32.IsUserAnAdmin() != 0:
            return True
    except Exception:
        pass

    # Secondary check: PowerShell role probe
    ps_bin = find_powershell()
    if ps_bin:
        cmd = (
            "([Security.Principal.WindowsPrincipal]"
            "[Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("
            "[Security.Principal.WindowsBuiltInRole]::Administrator)"
        )
        try:
            proc = subprocess.run(
                [
                    ps_bin,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "True" in proc.stdout
        except Exception:
            pass

    return False


def find_powershell() -> str:
    """Locate the preferred PowerShell executable on the system.

    Prioritizes PowerShell 7+ (`pwsh.exe`), falling back to Windows PowerShell 5.1 (`powershell.exe`).
    On non-Windows systems, checks for `pwsh` or falls back to `/bin/sh`.
    """
    if sys.platform == "win32":
        # 1. Check for PowerShell 7 (pwsh.exe) in PATH
        pwsh = shutil.which("pwsh.exe") or shutil.which("pwsh")
        if pwsh:
            return pwsh

        # 2. Check for Windows PowerShell (powershell.exe) in PATH
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell:
            return powershell

        # 3. Default Windows system directory fallback
        system32_ps = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"),
            r"System32\WindowsPowerShell\v1.0\powershell.exe",
        )
        if os.path.isfile(system32_ps):
            return system32_ps

        return "powershell.exe"
    else:
        # Non-Windows fallback
        pwsh = shutil.which("pwsh")
        if pwsh:
            return pwsh
        sh = shutil.which("bash") or shutil.which("sh")
        return sh or "/bin/sh"


def kill_process_tree(pid: int) -> None:
    """Forcefully terminate a process and all of its spawned child processes.

    On Windows, uses `taskkill /F /T /PID <pid>`.
    Falls back to `os.kill` if taskkill fails or on non-Windows platforms.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            try:
                import signal

                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    else:
        try:
            import signal

            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                import signal

                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def run_powershell(
    command: str,
    timeout: int = 60,
    cwd: Optional[str] = None,
) -> ProcessResult:
    """Execute a command via PowerShell with UTF-8 normalization and timeout protection.

    Args:
        command: PowerShell command line string to execute.
        timeout: Maximum duration in seconds before forceful termination (default: 60).
        cwd: Optional working directory for the command. Defaults to current directory.

    Returns:
        ProcessResult containing stdout, stderr, exit code, duration, and timeout status.
    """
    # Validate cwd if provided
    if cwd:
        abs_cwd = os.path.abspath(cwd)
        if not os.path.isdir(abs_cwd):
            return ProcessResult(
                command=command,
                stdout="",
                stderr=f"Directory does not exist: {cwd}",
                exit_code=1,
                duration_ms=0.0,
                timed_out=False,
            )
    else:
        abs_cwd = os.getcwd()

    # Clamp timeout to sane bounds (5s to 600s)
    timeout = max(1, min(timeout, 600))

    ps_bin = find_powershell()
    is_ps = "pwsh" in ps_bin.lower() or "powershell" in ps_bin.lower()

    if is_ps:
        full_command = f"{POWERSHELL_UTF8_PREAMBLE} {command}"
        args = [
            ps_bin,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            full_command,
        ]
    else:
        # Fallback for non-Windows bash/sh testing
        args = [ps_bin, "-c", command]

    start_time = time.perf_counter()
    proc: Optional[subprocess.Popen] = None
    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    exit_code = 0

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=abs_cwd,
            text=False,  # Raw bytes for clean UTF-8 decoding
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124  # Standard timeout exit code
            kill_process_tree(proc.pid)
            try:
                # Capture any output flushed before termination
                extra_out, extra_err = proc.communicate(timeout=2)
                if extra_out:
                    stdout_bytes += extra_out
                if extra_err:
                    stderr_bytes += extra_err
            except Exception:
                pass
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return ProcessResult(
            command=command,
            stdout="",
            stderr=f"Execution error: {exc}",
            exit_code=1,
            duration_ms=duration_ms,
            timed_out=False,
        )

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # UTF-8 decoding with safe replacement for unmappable bytes
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    # Strip ANSI escape codes
    clean_stdout = strip_ansi(stdout_text)
    clean_stderr = strip_ansi(stderr_text)

    return ProcessResult(
        command=command,
        stdout=clean_stdout,
        stderr=clean_stderr,
        exit_code=exit_code,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout_seconds: int = 60,
) -> ProcessResult:
    """Tool-level entrypoint matching OpenAI function calling schema.

    Args:
        command: The exact PowerShell command line string to execute.
        cwd: Optional working directory path. Defaults to current directory if omitted.
        timeout_seconds: Maximum execution time in seconds. Default: 60.

    Returns:
        ProcessResult with command execution details.
    """
    return run_powershell(command=command, timeout=timeout_seconds, cwd=cwd)


class PowerShellBridge:
    """Stateful PowerShell execution bridge maintaining working directory and execution policies."""

    def __init__(
        self,
        default_cwd: Optional[str] = None,
        default_timeout: int = 60,
    ) -> None:
        self.default_cwd = os.path.abspath(default_cwd) if default_cwd else os.getcwd()
        self.default_timeout = default_timeout
        self._is_admin_cached: Optional[bool] = None

    @property
    def is_admin(self) -> bool:
        """Cached admin privilege check."""
        if self._is_admin_cached is None:
            self._is_admin_cached = is_admin()
        return self._is_admin_cached

    def run(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ProcessResult:
        """Execute command in bridge's directory or specified cwd."""
        effective_cwd = cwd or self.default_cwd
        effective_timeout = timeout or self.default_timeout
        return run_powershell(
            command=command,
            timeout=effective_timeout,
            cwd=effective_cwd,
        )
