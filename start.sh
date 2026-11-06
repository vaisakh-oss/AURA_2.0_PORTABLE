#!/usr/bin/env bash
# ==============================================================================
# AURA 2.0 — Quick Start Launcher (macOS / Linux)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  AURA 2.0 — Autonomous Universal Reasoning Agent"
echo "  Enterprise Workspace Launcher"
echo "================================================================="

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "[-] Virtual environment not found. Running setup first..."
    ./setup.sh
fi

source .venv/bin/activate

# Check environment configuration
if [ ! -f ".env" ]; then
    echo "[-] .env file not found. Creating from .env.example..."
    cp .env.example .env
fi

# Check for API key configuration
API_KEY=$(grep -E "^GEMINI_API_KEY=" .env | cut -d= -f2- | tr -d "'" | tr -d '"' || true)
if [ -z "$API_KEY" ] || [ "$API_KEY" = "your_gemini_api_key_here" ]; then
    echo ""
    echo "  [!] WARNING: GEMINI_API_KEY is not configured in .env."
    echo "      AURA requires a Google Gemini API key for live reasoning."
    echo "      You can obtain a key for free at: https://aistudio.google.com/"
    echo "      Edit .env and add: GEMINI_API_KEY=your_key_here"
    echo ""
    read -p "Would you like to enter your Gemini API key now? (y/N): " -r PROMPT_KEY
    if [[ $PROMPT_KEY =~ ^[Yy]$ ]]; then
        read -s -p "Paste Gemini API Key: " USER_KEY
        echo ""
        if [ -n "$USER_KEY" ]; then
            python3 -c "
with open('.env', 'r') as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    if line.startswith('GEMINI_API_KEY='):
        new_lines.append(f'GEMINI_API_KEY={USER_KEY}\n')
    else:
        new_lines.append(line)
with open('.env', 'w') as f:
    f.writelines(new_lines)
print('[+] API key securely updated in .env')
"
        fi
    fi
fi

# Ensure runtime directories exist
mkdir -p logs knowledge_data/documents

HOST=$(grep -E "^AURA_HOST=" .env | cut -d= -f2- | tr -d "'" | tr -d '"' || echo "127.0.0.1")
PORT=$(grep -E "^AURA_PORT=" .env | cut -d= -f2- | tr -d "'" | tr -d '"' || echo "8000")
URL="http://${HOST}:${PORT}"

echo ""
echo "[+] Starting AURA 2.0 Gateway and React Workspace..."
echo "[+] Access URL: $URL"
echo "[+] Press Ctrl+C to stop the server."
echo "================================================================="
echo ""

# Attempt to open browser in background after 1.5s
(
    sleep 1.5
    if command -v open &>/dev/null; then
        open "$URL" >/dev/null 2>&1 || true
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL" >/dev/null 2>&1 || true
    fi
) &

exec python server.py

