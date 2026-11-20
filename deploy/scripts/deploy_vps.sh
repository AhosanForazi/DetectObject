#!/usr/bin/env bash
# ==============================================================================
# DetectObject — Ubuntu/Debian Automated Production Setup Script
# Run as root or with sudo on a clean Ubuntu 22.04 / 24.04 LTS VPS
# ==============================================================================

set -euo pipefail

echo ">>> [1/6] Updating APT repositories & installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    nginx \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    certbot \
    python3-certbot-nginx

APP_DIR="/opt/detectobject"
echo ">>> [2/6] Preparing application directory at ${APP_DIR}..."
sudo mkdir -p "${APP_DIR}"
sudo chown -R "$USER:$USER" "${APP_DIR}"

if [ ! -f "${APP_DIR}/app.py" ]; then
    echo "Copying repository files to ${APP_DIR}..."
    cp -r ./* "${APP_DIR}/"
fi

cd "${APP_DIR}"

echo ">>> [3/6] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt

echo ">>> [4/6] Creating application directories..."
mkdir -p outputs/images outputs/videos outputs/screenshots logs/training models/weights

echo ">>> [5/6] Installing and starting systemd service..."
sudo cp deploy/systemd/detectobject.service /etc/systemd/system/detectobject.service
# Update service user to current user
sudo sed -i "s/User=ubuntu/User=$USER/g" /etc/systemd/system/detectobject.service
sudo systemctl daemon-reload
sudo systemctl enable detectobject
sudo systemctl restart detectobject

echo ">>> [6/6] Configuring Nginx reverse proxy..."
sudo cp deploy/nginx/detectobject.conf /etc/nginx/sites-available/detectobject
sudo ln -sf /etc/nginx/sites-available/detectobject /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

PUBLIC_IP=$(curl -s ifconfig.me || echo "your-server-ip")
echo "=============================================================================="
echo " DetectObject Deployment Complete!"
echo " Access your application at: http://${PUBLIC_IP}"
echo ""
echo " To enable HTTPS (mandatory for live browser camera on remote servers):"
echo "   sudo certbot --nginx -d your-domain.com"
echo "=============================================================================="
