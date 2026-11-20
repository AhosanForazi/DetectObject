# DetectObject — Windows Automated Setup Script (PowerShell)
# Usage: .\setup_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== DetectObject Windows Setup ===" -ForegroundColor Cyan

# 1. Verify Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python is not installed or not on PATH. Install Python 3.10+ from https://www.python.org/downloads/."
    exit 1
}
python --version

# 2. Create virtual environment if not already present
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Activating .venv..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
Write-Host "Upgrading pip and installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Prepare dataset if not present
if (-not (Test-Path "datasets\processed\dataset.yaml")) {
    Write-Host "Preparing validated COCO128 dataset..." -ForegroundColor Yellow
    python training\dataset_prepare.py
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Launch DetectObject dashboard:  python app.py" -ForegroundColor Cyan
Write-Host "Launch Desktop Webcam mode:     python app.py --webcam" -ForegroundColor Cyan
Write-Host "Run Automated Tests:            pytest -v" -ForegroundColor Cyan
