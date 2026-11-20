# DetectObject — Production Deployment Guide
**AI-Powered Real-Time Object Detection, Tracking & Counting System**

This guide provides end-to-end instructions for deploying **DetectObject** across primary target environments:

1. [Render Deployment (Recommended Cloud Web Service)](#1-render-deployment-recommended-cloud-web-service)
2. [Vercel Architecture & Configuration](#2-vercel-architecture--configuration)
3. [Streamlit Community Cloud (Free 1-Click Git Deploy)](#3-streamlit-community-cloud-free-1-click-git-deploy)
4. [Hugging Face Spaces (Free Cloud Hosting)](#4-hugging-face-spaces-free-cloud-hosting)
5. [Docker & Docker Compose (Containerized Deployment)](#5-docker--docker-compose-containerized-deployment)
6. [Linux Cloud VPS / VM (AWS, GCP, Azure, DigitalOcean)](#6-production-linux-vps-aws-gcp-digitalocean)
7. [Webcam & Browser Camera Security Architecture (HTTPS Requirements)](#7-webcam--browser-security-architecture)
8. [Production Healthcheck & Troubleshooting](#8-production-healthcheck--troubleshooting)

---

## Pre-Deployment Verification Checklist

Before pushing to production, verify that your repository contains the following deployment configurations:

- [x] **`render.yaml`**: Infrastructure-as-code Blueprint for 1-click Render web service deployment.
- [x] **`vercel.json` & `public/index.html`**: Configures Vercel to serve a static landing page, avoiding Python serverless execution errors.
- [x] **`.python-version`**: Pinned to Python 3.11.9 for stable precompiled PyTorch/OpenCV wheels.
- [x] **`packages.txt`**: Debian/Ubuntu APT packages (`libgl1`, `libglib2.0-0`, `ffmpeg`) for cloud runners.
- [x] **`requirements.txt`**: Production dependencies using `opencv-python-headless` and `lapx`.
- [x] **`.streamlit/config.toml`**: `address = "0.0.0.0"`, `headless = true`, `maxUploadSize = 200`.
- [x] **`Dockerfile`**: Python 3.11-slim base, system libraries, and dynamic `$PORT` handling.
- [x] **`docker-compose.yml`**: Volume mounts for models/weights and outputs, automatic restart.
- [x] **`deploy/nginx/detectobject.conf`**: Nginx reverse proxy with WebSocket upgrade headers.
- [x] **`deploy/systemd/detectobject.service`**: Systemd unit file for Linux service auto-restart.

---

## 1. Render Deployment (Recommended Cloud Web Service)

Render is the primary cloud hosting platform for the DetectObject Streamlit application.

### Method A: Blueprint Deployment (Fastest)
1. Sign in to [render.com](https://render.com).
2. Click **Blueprints** &rarr; **New Blueprint Instance**.
3. Select your repository: `https://github.com/AhosanForazi/DetectObject`.
4. Render automatically detects [render.yaml](file:///c:/Users/Ahosan%20Habib/OneDrive/Desktop/detectObject/render.yaml) and configures the service, build command, start command, and environment variables.
5. Click **Apply**.

### Method B: Manual Web Service
1. On the Render Dashboard, click **New +** &rarr; **Web Service**.
2. Connect your GitHub repository `AhosanForazi/DetectObject`.
3. Configure the service:
   - **Name**: `detectobject`
   - **Region**: Oregon (US West) or Frankfurt (EU)
   - **Branch**: `main`
   - **Root Directory**: leave blank (root)
   - **Runtime**: `Python`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run application/dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Plan**: Free (or Starter for higher CPU/RAM)
4. Under **Advanced &rarr; Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`
5. Click **Create Web Service**. Render will build and deploy the application.

---

## 2. Vercel Architecture & Configuration

### Why Vercel Produced `[PYTHON_ENTRYPOINT_NOT_FOUND]`
Vercel is a Serverless Function / Jamstack hosting platform. When Vercel detected `app.py` in the root folder, it attempted to load it as a serverless microservice (WSGI/ASGI like Flask/FastAPI) and threw:
`[PYTHON_ENTRYPOINT_NOT_FOUND] Found app.py but it does not export a top-level "app", "application", or "handler" variable.`

Moreover, **Vercel cannot run Streamlit computer vision applications**:
- Serverless functions have a **250 MB size limit** (PyTorch + TorchVision + Ultralytics YOLO requires over 1.2 GB).
- Serverless functions terminate after 10–60 seconds, which terminates Streamlit's WebSocket connection and model caching.

### How DetectObject Handles Vercel
1. The repository includes [vercel.json](file:///c:/Users/Ahosan%20Habib/OneDrive/Desktop/detectObject/vercel.json) configured with `"outputDirectory": "public"`.
2. When Vercel builds the repo, it deploys [public/index.html](file:///c:/Users/Ahosan%20Habib/OneDrive/Desktop/detectObject/public/index.html) as a fast, responsive static showcase page that links to your live Render application.
3. This completely bypasses the Python serverless runtime, allowing your Vercel deployment to succeed 100% without error.
4. **Alternative**: If you do not need Vercel, simply navigate to your Vercel Project Settings &rarr; **Delete Project** (or disconnect the GitHub repository), as Render hosts the complete application.

---

## 3. Streamlit Community Cloud (Free 1-Click Git Deploy)

Streamlit Community Cloud provides 100% free hosting, automated CI/CD directly from GitHub, and free **HTTPS** (which is essential for browser camera permissions).

### Step 1: Push Project to GitHub

Open your terminal or PowerShell inside `detectObject` and push the code to a new GitHub repository:

```bash
# 1. Initialize git repository if not already done
git init
git add .
git commit -m "Deploy DetectObject v1.0"

# 2. Set default branch to main
git branch -M main

# 3. Link your remote GitHub repository
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/detectObject.git

# 4. Push code to GitHub
git push -u origin main
```

### Step 2: Connect Streamlit Community Cloud

1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click the **"New app"** button.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/detectObject`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py` (or `application/dashboard.py`).
6. Click **"Deploy!"**.

### Step 3: Automatic Build & Launch

Streamlit Cloud will:
- Detect `packages.txt` and install `libgl1`, `libglib2.0-0`, and `ffmpeg`.
- Install all Python dependencies from `requirements.txt`.
- Launch the application on a public URL: `https://<your-app-name>.streamlit.app`.

> [!TIP]
> **Webcam on Streamlit Cloud:**
> Since Streamlit Cloud provides HTTPS out of the box, camera permissions work seamlessly.
> For maximum compatibility with cloud container firewalls, use the **Cloud Snapshot Camera** tab on the Live Webcam page, or use the 30 FPS stream if secondary WebSockets are allowed by your network.

---

## 2. Hugging Face Spaces (Free Cloud Hosting)

Hugging Face Spaces offers free CPU hardware (and optional GPU upgrades) with persistent storage and Git-based deployments.

### Step 1: Create a New Space

1. Log in to [Hugging Face](https://huggingface.co/).
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
3. Choose a Space name (e.g., `DetectObject`).
4. Select **Space SDK**: **Streamlit**.
5. Select **Space Hardware**: **CPU basic (2 vCPU · 16 GB RAM — Free)**.
6. Click **"Create Space"**.

### Step 2: Push Repository to Hugging Face

In your terminal, add Hugging Face as a remote and push:

```bash
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/DetectObject
git push -u hf main --force
```

Hugging Face reads `packages.txt` and `requirements.txt`, builds the environment, and launches the app with a public HTTPS link!

---

## 3. Docker & Docker Compose (Containerized Deployment)

Use Docker to deploy DetectObject on any Linux, Windows Server, or macOS host with guaranteed reproducibility.

### Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)

### Method A: Docker Compose (Single Command)

```bash
# 1. Build and start container in detached background mode
docker-compose up -d --build

# 2. View live runtime logs
docker-compose logs -f

# 3. Check container health status
docker-compose ps

# 4. Stop container when finished
docker-compose down
```

The application is immediately available at:
`http://localhost:8501`

### Method B: Standalone Docker CLI

```bash
# 1. Build Docker image
docker build -t detectobject:latest .

# 2. Run container with port forwarding and persistent volumes
docker run -d \
  --name detectobject \
  -p 8501:8501 \
  -p 8511:8511 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/models/weights:/app/models/weights \
  --restart unless-stopped \
  detectobject:latest

# 3. Verify logs
docker logs -f detectobject
```

---

## 4. Production Linux VPS (AWS, GCP, DigitalOcean)

For a dedicated production server running Ubuntu 22.04 or 24.04 LTS with automated service recovery, Nginx reverse proxy, and SSL certificates.

### Automated One-Click VPS Setup

We have bundled a complete setup script in `deploy/scripts/deploy_vps.sh`. On your clean server:

```bash
# Clone or copy your detectObject folder to the server, then run:
cd /opt/detectobject
chmod +x deploy/scripts/deploy_vps.sh
sudo bash deploy/scripts/deploy_vps.sh
```

### Manual VPS Step-by-Step Instructions

#### 1. Install System Packages & Nginx
```bash
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git nginx ffmpeg libgl1 libglib2.0-0 curl certbot python3-certbot-nginx
```

#### 2. Set Up Application Directory & Virtualenv
```bash
sudo mkdir -p /opt/detectobject
sudo chown -R $USER:$USER /opt/detectobject
cd /opt/detectobject

# Copy project files here, then:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Configure Systemd Service
Copy the included service unit to `/etc/systemd/system/`:

```bash
sudo cp deploy/systemd/detectobject.service /etc/systemd/system/detectobject.service
sudo systemctl daemon-reload
sudo systemctl enable detectobject
sudo systemctl start detectobject
sudo systemctl status detectobject
```

#### 4. Configure Nginx Reverse Proxy
Copy the included Nginx configuration:

```bash
sudo cp deploy/nginx/detectobject.conf /etc/nginx/sites-available/detectobject
sudo ln -sf /etc/nginx/sites-available/detectobject /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. Obtain Free Let's Encrypt SSL Certificate (CRITICAL for Camera)
```bash
sudo certbot --nginx -d your-domain.com
```

Certbot automatically configures HTTPS redirects and installs SSL certificates.

---

## 5. Local Network / LAN Deployment

To access DetectObject from your smartphone, tablet, or another computer on the same home or office Wi-Fi:

### Step 1: Start Server on `0.0.0.0`
```bash
streamlit run application/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

### Step 2: Find Your Host Computer's Local IP Address
- **Windows**: Run `ipconfig` in PowerShell (look for `IPv4 Address`, e.g. `192.168.1.100`).
- **Linux / Mac**: Run `hostname -I` or `ip a` (e.g. `192.168.1.100`).

### Step 3: Open on Any Device
On your phone or laptop browser, open:
`http://192.168.1.100:8501`

> [!NOTE]
> **Mobile Browser Camera over HTTP:**
> Modern mobile browsers (iOS Safari, Android Chrome) restrict camera access on non-`localhost` HTTP origins.
> To use the camera on your phone:
> - Either use the **Cloud Snapshot Camera** tab which works through standard photo capture,
> - Or in Android Chrome, visit `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enter `http://192.168.1.100:8501`, and enable it for testing!

---

## 6. Webcam & Browser Security Architecture

### Why Does HTTPS Matter for Camera Access?

The W3C Media Capture Specification (`navigator.mediaDevices.getUserMedia`) restricts camera hardware access strictly to **Secure Contexts**:
- `http://localhost:*` and `http://127.0.0.1:*` (Allowed automatically)
- `https://*` (Allowed with user permission)
- `http://<remote-ip>` (Blocked by browser security policies)

| Deployment Target | Protocol | Live Stream (30 FPS) | Cloud Snapshot Camera |
|---|---|---|---|
| **Local PC / Laptop** | `http://localhost:8501` | Supported | Supported |
| **Streamlit Community Cloud** | `https://*.streamlit.app` | Supported | Supported |
| **Hugging Face Spaces** | `https://*.hf.space` | Supported | Supported |
| **Docker (Local)** | `http://localhost:8501` | Supported | Supported |
| **Cloud VPS with SSL** | `https://your-domain.com` | Supported | Supported |
| **Cloud VPS (HTTP Only)** | `http://your-server-ip` | Blocked by browser | Blocked by browser (Requires SSL) |

---

## 7. Production Healthcheck & Troubleshooting

### Issue 1: `ImportError: libGL.so.1: cannot open shared object file`
- **Cause**: OpenCV requires standard OpenGL system libraries missing in minimal Linux / Docker images.
- **Fix**: Ensure `packages.txt` contains `libgl1` and `libglib2.0-0`. On Linux, run `sudo apt-get install -y libgl1 libglib2.0-0`.

### Issue 2: Streamlit File Upload Limit (Videos)
- **Cause**: Default Streamlit upload limit is 200MB.
- **Fix**: Configured in `.streamlit/config.toml` (`maxUploadSize = 200`) and Nginx `client_max_body_size 200M;`.

### Issue 3: High CPU Usage in Production
- In the dashboard Options menu:
  - Set **Resolution (px)** to `320px` (doubles FPS on CPU).
  - Increase **Frame Skip** to `2` or `3` for uploaded video processing.
  - Set `device: cuda` in `configuration/config.yaml` if an NVIDIA GPU is available.

### Issue 4: Docker Memory Allocation
- Give Docker at least 4 GB RAM in Docker Desktop Settings → Resources → Memory.
