@echo off
setlocal enabledelayedexpansion
title "Swarm Agent Gateway & Autonomous Swarm Runtime Launcher"

:: Set current working directory to the batch script directory
cd /d "%~dp0"

echo ======================================================================
echo    SWARM AGENT: Multi-Provider Token Gateway ^& Autonomous Runtime
echo ======================================================================
echo.

:: 1. Detect Python 3.10+
set "PYTHON_CMD="
where py.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3"
    goto :check_version
)

where python.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    goto :check_version
)

echo [ERROR] Python was not found in your PATH.
echo Please install Python 3.10 or higher from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
exit /b 1

:check_version
%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10 or higher is required.
    for /f "tokens=*" %%v in ('%PYTHON_CMD% --version') do echo Current version: %%v
    exit /b 1
)

:: 2. Ensure Virtual Environment (.venv)
set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "!VENV_PYTHON!" goto :verify_venv

:create_venv
echo [*] Creating virtual environment .venv ...
%PYTHON_CMD% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)
echo [*] Virtual environment created successfully.
echo [*] Upgrading pip...
!VENV_PYTHON! -m pip install --upgrade pip --quiet
echo [*] Installing dependencies from requirements.txt...
!VENV_PYTHON! -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
)
echo [*] Dependencies installed successfully.
goto :dispatch_args

:verify_venv
!VENV_PYTHON! -c "import sys" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :dispatch_args

echo [WARNING] Existing .venv appears corrupted. Recreating...
rmdir /s /q .venv
goto :create_venv

:: 3. Dispatch CLI Arguments
:dispatch_args
set "ARG1=%~1"

if /i "%ARG1%"=="--help" goto :help
if /i "%ARG1%"=="-h" goto :help
if /i "%ARG1%"=="--test" goto :run_tests
if /i "%ARG1%"=="--wizard" goto :run_wizard
if /i "%ARG1%"=="--status" goto :run_status
if /i "%ARG1%"=="--daemon" goto :run_daemon

:: 4. Check for config.json; if missing, launch wizard
if not exist "config.json" (
    echo [!] No config.json found. Launching initial configuration wizard...
    echo.
    !VENV_PYTHON! -m swarm.cli.wizard %*
    if not exist "config.json" (
        echo [!] Setup was cancelled or config.json was not created.
        exit /b 1
    )
)

:: Default action: Launch the Gateway Server Daemon
:run_daemon
shift
echo [*] Starting Swarm Gateway Server on configured host and port...
!VENV_PYTHON! -m swarm.main %*
goto :end

:run_wizard
shift
echo [*] Launching Interactive Setup Wizard...
!VENV_PYTHON! -m swarm.cli.wizard %*
goto :end

:run_status
shift
echo [*] Checking Swarm Gateway Status...
!VENV_PYTHON! -m swarm.cli.client status %*
goto :end

:run_tests
shift
echo [*] Running Automated Verification Test Suite...
!VENV_PYTHON! -m pytest -v tests/ %*
goto :end


:help
echo Usage: run_swarm.bat [OPTION]
echo.
echo Options:
echo   --daemon   Start the gateway server daemon (default if config.json exists)
echo   --wizard   Run the interactive configuration wizard for 8 free-tier providers
echo   --status   Check the health and circuit breaker status of the local gateway
echo   --test     Execute the automated pytest verification test suite
echo   --help     Display this help message
echo.
goto :end

:end
exit /b %ERRORLEVEL%
