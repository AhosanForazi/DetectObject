# DetectObject — Setup, Deployment & Troubleshooting Guide

## 1. System Requirements

| Component | Minimum (CPU Mode) | Recommended (GPU Mode) |
|---|---|---|
| **Operating System** | Windows 10/11 (64-bit) or Linux | Windows 11 / Ubuntu 22.04+ |
| **Python** | 3.10+ (Verified on Python 3.14.4) | 3.10 – 3.14 |
| **RAM** | 6 GB (4 GB usable) | 16 GB+ |
| **Disk Space** | 2 GB free (code, weights, COCO128) | 10 GB+ (for larger datasets) |
| **GPU** | Optional (runs on CPU automatically) | NVIDIA GPU with CUDA 12.x |
| **Camera** | Any USB or integrated webcam | 720p/1080p UVC webcam |

---

## 2. Windows Installation & Launch

### Automated Setup (PowerShell)
```powershell
cd C:\path\to\detectObject
.\setup_windows.ps1
```

### Manual Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Launch Application
```powershell
# Primary launch command (starts Streamlit dashboard on http://localhost:8501)
python app.py

# Custom port or headless server mode
python app.py --port 8502 --headless

# Direct Streamlit invocation
streamlit run application/dashboard.py

# Native OpenCV desktop webcam mode (keyboard: q=quit, p=pause, s=screenshot)
python app.py --webcam --camera 0
```

---

## 3. Linux Installation & Launch

### Automated Setup
```bash
bash setup_linux.sh
```

### Manual Setup
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 v4l-utils
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

---

## 4. Optional NVIDIA CUDA Acceleration

DetectObject automatically detects CUDA (`torch.cuda.is_available()`) when `device: auto` is set in `configuration/config.yaml`. If you have an NVIDIA GPU and currently have CPU-only PyTorch installed:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
python app.py --info
```

Verify that `"cuda_available": true` and `"selected": "cuda:0"` appear in the output.

---

## 5. Model Download, Training & Evaluation

- **Automatic Baseline Weights**: On first run, if `models/weights/best.pt` is absent, `ObjectDetector` automatically falls back to `models/weights/yolo11n.pt` (downloaded once from Ultralytics releases, ~5.4 MB).
- **Dataset Preparation**:
  ```bash
  python training/dataset_prepare.py
  ```
- **Model Fine-Tuning**:
  ```bash
  python training/train.py --epochs 8
  # Resume an interrupted run:
  python training/train.py --resume
  ```
- **Independent Test Evaluation**:
  ```bash
  python evaluation/evaluate.py
  ```
- **Optional Open-Vocabulary Model (YOLO-World)**: No setup is required at startup. Switching to **Advanced Open-Vocabulary Detection** on the Image Detection page lazy-loads `models/weights/yolov8s-worldv2.pt` (~25 MB) on demand. The core app runs without it.

---

## 6. Webcam Permission Instructions

### Windows 11
1. Press `Win + I` → **Privacy & security** → **Camera**.
2. Turn ON **Camera access**.
3. Turn ON **Let apps access your camera** and **Let desktop apps access your camera**.
4. Ensure any physical webcam privacy shutter or Fn camera toggle key is open.

### Linux
1. Ensure your user belongs to the `video` group: `sudo usermod -aG video $USER` (re-login required).
2. Verify device nodes: `ls -l /dev/video*`.

### Browser Camera Mode
When using **Browser Camera Capture** inside the Live Webcam page, click **Allow** when Chrome/Edge/Firefox displays the camera permission prompt. Note that browsers require `http://localhost` or `https://` for `getUserMedia`.

---

## 7. Troubleshooting Guide

| Symptom | Cause | Resolution |
|---|---|---|
| `Unable to open camera index 0` | Webcam in use by another app or privacy setting disabled | Close Zoom/Teams/OBS, enable Windows Camera Privacy for desktop apps, click **Refresh Cameras**. |
| Slow webcam FPS on CPU | High inference resolution (`640px`) | Set **Inference Size** to `320px` (measured ~21 FPS on Ryzen 5 7520U) or set **Frame Skip** to `2`. |
| Uploaded video fails to play in browser | Non-H.264 codec fallback | Ensure `av` (`PyAV`) is installed (`pip install av`) so videos export as H.264 `yuv420p` MP4. |
| Out-of-memory during training | Batch size too large for RAM/VRAM | Set `batch_size: auto` in `configuration/config.yaml` or pass `--batch 4 --imgsz 320`. The trainer also halves batch size automatically on OOM. |
| Missing `best.pt` | Training not run yet | Core app automatically falls back to `yolo11n.pt`; run `python training/train.py` to produce `best.pt`. |
