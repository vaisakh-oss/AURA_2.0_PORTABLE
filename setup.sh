#!/usr/bin/env bash
# ==============================================================================
# AURA 2.0 — Automated Environment Setup (macOS / Linux)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  AURA 2.0 — Environment Setup & Dependency Installation"
echo "================================================================="

# 1. Check Python version (requires >= 3.10)
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo "[+] Found compatible Python: $cmd ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[-] Error: Python 3.10 or newer is required."
    echo "    Please install Python 3.10+ from https://www.python.org/downloads/ or via package manager."
    exit 1
fi

# 2. Virtual Environment Setup
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creating Python virtual environment in .venv..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "[+] Using existing virtual environment: $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# 3. Upgrade pip and install dependencies
echo "[+] Upgrading pip and wheel..."
pip install --upgrade pip setuptools wheel >/dev/null

echo "[+] Installing backend Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Optional Node.js & Frontend dependencies
if command -v npm &>/dev/null && [ -d "web" ]; then
    echo "[+] Node.js detected: $(node -v). Installing frontend dependencies..."
    (cd web && npm install --no-audit --no-fund)
    echo "[+] Building fresh production frontend SPA bundle..."
    (cd web && npm run build)
else
    echo "[!] Node.js not detected or skipped. Serving pre-compiled production bundle from web/dist/."
fi

# 5. Environment configuration setup
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[+] Creating .env from .env.example..."
        cp .env.example .env
        echo "================================================================="
        echo "  [!] ACTION REQUIRED: Configure your Gemini API key in .env"
        echo "      Open .env in a text editor and set GEMINI_API_KEY"
        echo "      Get a free key from: https://aistudio.google.com/"
        echo "================================================================="
    fi
else
    echo "[+] Existing .env file found."
fi

# 6. Ensure runtime directories exist
mkdir -p logs knowledge_data/documents

echo ""
echo "[✓] Setup completed successfully!"
echo "    To launch AURA 2.0, run: ./start.sh"

