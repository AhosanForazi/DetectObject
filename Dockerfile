# ==============================================================================
# DetectObject — Production Dockerfile (CPU-Optimized & Production-Ready)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system runtime dependencies for OpenCV, FFmpeg, and PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . /app

# Ensure output directories exist with proper write permissions
RUN mkdir -p /app/outputs/images \
             /app/outputs/videos \
             /app/outputs/screenshots \
             /app/logs/training \
             /app/models/weights

# Expose Streamlit dashboard port (8501) and Live Inference API port (8511)
EXPOSE 8501
EXPOSE 8511

# Healthcheck to monitor Streamlit availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the DetectObject dashboard
ENTRYPOINT ["streamlit", "run", "application/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
