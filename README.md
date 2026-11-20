

# DetectObject — Real-Time Object Detection System

**DetectObject** is an end-to-end computer vision platform for real-time **multi-object detection, tracking, and counting** across images, uploaded videos, and live webcams. Built on **Ultralytics YOLO11**, **PyTorch**, **OpenCV**, **PyAV**, and **Streamlit**, it runs smoothly on CPU-only laptops while automatically utilizing NVIDIA CUDA or Apple MPS when available.

---

## Key Features

- **Image Detection**: Upload `JPG`, `JPEG`, `PNG`, or `WEBP` images (or use built-in test samples), inspect original vs. annotated outputs side by side, view total and class-wise object counts, and download annotated PNGs and JSON detection payloads.
- **Optional Open-Vocabulary Detection**: Enter custom text prompts (e.g., `person, laptop, smartphone, water bottle, helmet, bag, chair, bicycle, dog, cat`) powered by lazy-loaded **YOLO-World v2** without adding startup overhead to the core app.
- **Video Detection & Export**: Frame-by-frame processing for `MP4`, `AVI`, `MOV`, `MKV`, and `WEBM` with real-time progress bar, FPS telemetry, optional tracking, and browser-playable **H.264 MP4** exports via PyAV.
- **Live Webcam Detection (Local + Browser Modes)**:
  - Thread-safe OpenCV live webcam stream with **Start, Stop, Pause, Resume, Camera Switching**, live threshold controls, FPS HUD, per-frame & unique tracked counts, class breakdown, and **annotated screenshot capture** saved to `outputs/screenshots/`.
  - Guaranteed safe camera resource release (`cap.release()`) on Stop, page navigation, or exit.
  - Secondary **Browser Camera Capture** (`st.camera_input`) for remote/browser-permission camera snapshots, plus standalone native desktop OpenCV mode (`python app.py --webcam`).
- **Multi-Object Tracking & Counting**: Native **ByteTrack** and **BoT-SORT** integration with persistent track IDs, motion trails, directional **virtual line-crossing counters (IN/OUT)**, and **Region-of-Interest (ROI)** zone counting—explicitly distinguishing per-frame detections from unique tracked objects.
- **Reproducible ML Pipeline**: Checksum-verified dataset download, image/label integrity validation, deduplication, leak-free `train/val/test` splits, resource-aware training with OOM safety & checkpoint backups, and independent test evaluation with plots and reports.

---

## Measured Model Performance (Independent Test Split)

Evaluated on the held-out independent test split (19 images, 106 instances, zero data leakage) using `models/weights/best.pt` (fine-tuned `YOLO11n`, 2.62M parameters) on an **AMD Ryzen 5 7520U CPU**:

| Metric | Measured Value |
|---|---:|
| **Precision** | `0.6913` |
| **Recall** | `0.5563` |
| **F1-Score** | `0.6165` |
| **mAP@0.5** | `0.6039` |
| **mAP@0.5:0.95** | `0.4527` |
| **Measured CPU Speed @ 320px** | **21.29 FPS** (46.98 ms mean) |
| **Measured CPU Speed @ 480px** | **12.06 FPS** (82.90 ms mean) |
| **Measured CPU Speed @ 640px** | **5.99 FPS** (166.86 ms mean) |

Full metrics, confusion matrix, PR curve, loss curves, and class-wise breakdown are available in `reports/evaluation_report.md` and inside the **Evaluation Results** page of the dashboard.

---

## Quick Start

### Windows (Recommended)

```powershell
# Optional: automated virtual environment + dependency setup
.\setup_windows.ps1

# Launch the DetectObject web dashboard
python app.py
```

Alternative Streamlit launch command:

```powershell
streamlit run application/dashboard.py
```

Standalone native OpenCV desktop live webcam:

```powershell
python app.py --webcam
```

### Linux / macOS

```bash
bash setup_linux.sh
python app.py
# Or: streamlit run application/dashboard.py
```

---

## Pipeline Commands

```bash
# 1. Download, validate, clean, deduplicate, and build leak-free train/val/test splits
python training/dataset_prepare.py

# 2. Train / fine-tune YOLO11n with automatic checkpoint backup
python training/train.py --epochs 8

# 3. Evaluate on independent held-out test split and generate plots + report
python evaluation/evaluate.py

# 4. Run full automated test suite
pytest -v

# 5. Inspect hardware & model readiness
python app.py --info
```

---

## Project Structure

```text
detectObject/
├── app.py                        # Unified launcher (Streamlit dashboard, --webcam, --info)
├── requirements.txt              # Python dependencies
├── setup_windows.ps1             # Windows PowerShell setup script
├── setup_linux.sh                # Linux setup script
├── configuration/
│   ├── config.yaml               # Central YAML configuration
│   ├── classes.yaml              # Canonical 80-class COCO mapping
│   └── settings.py               # Config loader, auto device detection, logging
├── datasets/
│   ├── raw/                      # Verified archive + extracted raw files
│   ├── processed/                # Leak-free train/val/test splits + dataset.yaml
│   ├── manifests/                # dataset_manifest.json & split_manifest.json
│   └── dataset_loader.py         # Download, integrity checks, normalization, splitting
├── models/
│   ├── detector.py               # ObjectDetector (YOLO11) + optional OpenVocabularyDetector
│   ├── tracker.py                # ByteTrack / BoT-SORT state & trail manager
│   └── weights/                  # best.pt, yolo11n.pt, and backups/
├── training/
│   ├── dataset_prepare.py        # Dataset preparation CLI
│   ├── trainer.py                # Reproducible resource-aware YOLO trainer
│   └── train.py                  # Training CLI
├── evaluation/
│   ├── metrics.py                # Independent test evaluation & FPS benchmarking
│   ├── report.py                 # Plots & evaluation_report.md generator
│   └── evaluate.py               # Evaluation CLI
├── detection/
│   ├── image_detector.py         # Image detection & export pipeline
│   ├── video_detector.py         # Video frame-by-frame detection + PyAV H.264 export
│   ├── webcam_detector.py        # Thread-safe live webcam manager + desktop mode
│   ├── object_counter.py         # Per-frame, unique tracked, line-crossing & ROI counters
│   └── visualizer.py             # Modern bounding boxes, trails, zones & HUD renderer
├── application/
│   ├── dashboard.py              # 9-page responsive Streamlit dashboard
│   └── components/               # Dark/light theme CSS & reusable widgets
├── outputs/                      # Saved annotated images, videos & screenshots
├── reports/                      # evaluation_report.md, metrics JSON & plots/
├── docs/                         # Technical docs, deployment guide, viva Q&A & full report
├── logs/                         # Application & training logs
└── tests/                        # Automated pytest suite
```

---

## Webcam Permissions & Troubleshooting

- **Windows Privacy Settings**: Open *Settings → Privacy & security → Camera* and enable **Camera access** and **Let desktop apps access your camera**.
- **Camera Busy Error**: Close Zoom, Teams, OBS, or browser tabs currently holding the camera, then click **Refresh Cameras → Start Webcam**.
- **Remote Browser Access**: Switch to **Browser Camera Capture** inside the Live Webcam page so the browser handles `getUserMedia` camera permissions directly.
- **Low FPS on CPU**: Lower **Inference Size** to `320px` (measured **21.3 FPS** on Ryzen 5 7520U) or increase **Frame Skip** to `2`.

---

## Production Deployment

DetectObject is pre-configured for instant deployment across multiple environments:

```bash
# Docker Compose (1-Command Local or Server Deployment)
docker-compose up -d --build

# Standalone Docker
docker build -t detectobject:latest .
docker run -d -p 8501:8501 -p 8511:8511 detectobject:latest

# Streamlit Community Cloud (Free 1-Click Git Deploy)
# Push to GitHub -> Deploy at share.streamlit.io (uses packages.txt and requirements.txt)
```

See the complete **[Production Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** for detailed instructions covering Streamlit Community Cloud, Hugging Face Spaces, Docker, and Ubuntu VPS with Nginx and SSL.

---

## Documentation Deliverables

- [Production Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Setup & Deployment Guide](docs/SETUP_AND_DEPLOYMENT.md)
- [Model Evaluation Report](reports/evaluation_report.md)
- [Final Academic Project Report](docs/FINAL_PROJECT_REPORT.md)
- [Presentation Outline](docs/PRESENTATION_OUTLINE.md)
- [Viva Preparation Q&A](docs/VIVA_QA.md)
