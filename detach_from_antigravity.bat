@echo off
setlocal enabledelayedexpansion
title "Swarm Agent: Detach from Antigravity (Zero-Residue Reversion)"

cd /d "%~dp0"

echo ======================================================================
echo    SWARM AGENT: Detaching from Antigravity (RESTORE ORIGINAL STATE)
echo ======================================================================
echo.

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "!VENV_PYTHON!" (
    set "VENV_PYTHON=python"
)

!VENV_PYTHON! -m swarm antigravity detach
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to cleanly detach Swarm Agent.
    exit /b !ERRORLEVEL!
)

echo.
echo [SUCCESS] Detachment complete. Antigravity settings restored!
echo ======================================================================
