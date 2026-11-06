@echo off
REM ==============================================================================
REM AURA 2.0 — Quick Start Launcher (Windows 10/11)
REM ==============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo =================================================================
echo   AURA 2.0 - Autonomous Universal Reasoning Agent
echo   Enterprise Workspace Launcher (Windows)
echo =================================================================

REM Check virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [-] Virtual environment not found. Running setup.bat first...
    call setup.bat
)

call .venv\Scripts\activate.bat

REM Check .env configuration
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
    )
)

if not exist "logs" mkdir "logs"
if not exist "knowledge_data\documents" mkdir "knowledge_data\documents"

set HOST=127.0.0.1
set PORT=8000
set URL=http://%HOST%:%PORT%

echo [+] Starting AURA 2.0 Gateway and React Workspace...
echo [+] Access URL: %URL%
echo [+] Press Ctrl+C to stop the server.
echo =================================================================
echo.

REM Launch browser in 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start %URL%"

python server.py
pause

