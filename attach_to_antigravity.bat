@echo off
setlocal enabledelayedexpansion
title "Swarm Agent: Attach to Antigravity (Temporary)"

cd /d "%~dp0"

echo ======================================================================
echo    SWARM AGENT: Attaching to Antigravity (TEMPORARY INTEGRATION)
echo ======================================================================
echo.

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "!VENV_PYTHON!" (
    echo [ERROR] Virtual environment not found at !VENV_PYTHON!
    echo Please run run_swarm.bat first to initialize dependencies.
    exit /b 1
)

!VENV_PYTHON! -m swarm antigravity attach
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to attach Swarm Agent to Antigravity.
    exit /b !ERRORLEVEL!
)

echo.
echo [*] Verifying MCP Server communication...
!VENV_PYTHON! -m swarm antigravity test
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Verification reported issues. Check diagnostics above.
)

echo.
echo [INFO] Swarm Agent is now TEMPORARILY active in Antigravity!
echo [INFO] You can command the swarm across all 7 frontier models.
echo [INFO] To cleanly detach at any time, run: detach_from_antigravity.bat
echo ======================================================================
