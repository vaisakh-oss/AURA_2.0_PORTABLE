# ==============================================================================
# AURA 2.0 — PowerShell Quick Start Launcher (Windows / macOS / Linux)
# ==============================================================================
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  AURA 2.0 — Autonomous Universal Reasoning Agent" -ForegroundColor Cyan
Write-Host "  Enterprise Workspace Launcher (PowerShell)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Check for virtual environment
$VenvActivateWin = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
$VenvActivateUnix = Join-Path $ScriptDir ".venv/bin/Activate.ps1"

if (Test-Path $VenvActivateWin) {
    & $VenvActivateWin
} elseif (Test-Path $VenvActivateUnix) {
    & $VenvActivateUnix
} else {
    Write-Host "[-] Virtual environment not found. Please run setup script first." -ForegroundColor Yellow
    exit 1
}

# Ensure directories
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "knowledge_data\documents" | Out-Null

$Url = "http://127.0.0.1:8000"
Write-Host "[+] Starting AURA 2.0 Gateway and React Workspace..." -ForegroundColor Green
Write-Host "[+] Access URL: $Url" -ForegroundColor Green
Write-Host "[+] Press Ctrl+C to stop the server." -ForegroundColor Yellow

# Launch browser after delay
Start-Job -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 2
    Start-Process $u
} -ArgumentList $Url | Out-Null

python server.py

