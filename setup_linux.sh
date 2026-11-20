#!/usr/bin/env bash
# DetectObject — Linux Setup Script
# Usage: bash setup_linux.sh

set -euo pipefail

echo "=== DetectObject Linux Setup ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed. Install Python 3.10+ first." >&2
  exit 1
fi

python3 --version

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f "datasets/processed/dataset.yaml" ]; then
  echo "Preparing validated COCO128 dataset..."
  python training/dataset_prepare.py
fi

echo ""
echo "Setup complete!"
echo "Launch DetectObject dashboard:  python app.py"
echo "Launch Streamlit directly:      streamlit run application/dashboard.py"
echo "Run Automated Tests:            pytest -v"
