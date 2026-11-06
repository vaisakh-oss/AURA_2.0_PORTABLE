@echo off
REM ==============================================================================
REM AURA 2.0 — Automated Environment Setup (Windows 10/11)
REM ==============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo =================================================================
echo   AURA 2.0 - Environment Setup (Windows)
echo =================================================================

REM 1. Check Python installation
set PYTHON_CMD=
for %%C in (python py python3) do (
    %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=%%C
        goto :FOUND_PYTHON
    )
)

:FOUND_PYTHON
if "%PYTHON_CMD%"=="" (
    echo [-] Error: Python 3.10 or newer was not found.
    echo     Please install Python 3.10+ from https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [+] Found compatible Python interpreter: %PYTHON_CMD%

REM 2. Create Virtual Environment
if not exist ".venv" (
    echo [+] Creating virtual environment in .venv...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [-] Error creating virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [+] Using existing virtual environment: .venv
)

call .venv\Scripts\activate.bat

REM 3. Install Python Dependencies
echo [+] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

echo [+] Installing required packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [-] Warning: Some packages may have failed to install.
)

REM 4. Optional Node.js & Frontend Build
where npm >nul 2>&1
if %errorlevel% equ 0 (
    if exist "web" (
        echo [+] Node.js detected. Installing frontend dependencies...
        cd web
        call npm install --no-audit --no-fund
        echo [+] Building frontend bundle...
        call npm run build
        cd ..
    )
) else (
    echo [!] Node.js not detected. Serving pre-compiled production bundle from web\dist\.
)

REM 5. Setup .env configuration
if not exist ".env" (
    if exist ".env.example" (
        echo [+] Creating .env from .env.example...
        copy .env.example .env >nul
        echo =================================================================
        echo   [!] ACTION REQUIRED: Configure your Gemini API key in .env
        echo       Get a free key from: https://aistudio.google.com/
        echo =================================================================
    )
) else (
    echo [+] Existing .env file found.
)

REM 6. Create directories
if not exist "logs" mkdir "logs"
if not exist "knowledge_data\documents" mkdir "knowledge_data\documents"

echo.
echo [v] Setup completed successfully!
echo     To start AURA 2.0, double-click start.bat
pause

