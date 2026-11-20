# DetectObject — Technical Documentation

## 1. Architectural Overview

DetectObject follows a strict layered architecture separating configuration, data engineering, ML inference/tracking, media pipelines, and UI presentation:

```text
+-------------------------------------------------------------------------+
|                       Presentation Layer (Streamlit)                    |
|   application/dashboard.py (9 pages) + application/components/{theme,   |
|   widgets}.py  <--- Launched via app.py (`python app.py`)               |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                      Detection & Analytics Pipelines                    |
|   detection/image_detector.py  | detection/video_detector.py            |
|   detection/webcam_detector.py | detection/object_counter.py            |
|   detection/visualizer.py                                               |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                       Model & Tracking Core Layer                       |
|   models/detector.py (ObjectDetector + OpenVocabularyDetector)          |
|   models/tracker.py  (ObjectTracker: ByteTrack / BoT-SORT)              |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|            Data, Training, Evaluation & Configuration Layer             |
|   configuration/settings.py   | datasets/dataset_loader.py              |
|   training/trainer.py         | evaluation/{metrics,report}.py          |
+-------------------------------------------------------------------------+
```

---

## 2. Module Reference

### 2.1 `configuration/`
- `config.yaml`: Single source of truth for model weights, inference thresholds, tracking parameters, counting defaults, webcam resolution/skip, dataset source & split ratios, training hyperparameters, evaluation settings, and UI defaults.
- `classes.yaml`: Canonical 80-class COCO taxonomy (`0: person` through `79: toothbrush`).
- `settings.py`:
  - `load_config()` / `save_config()`: Strongly structured YAML reader/writer (`AppConfig`).
  - `detect_compute_device(preferred)`: Inspects `torch.cuda.is_available()` and `torch.backends.mps.is_available()`, selecting `cuda:0`, `mps`, or `cpu` with explicit fallback flags.
  - `get_system_resources()`: Reports OS, CPU core counts, live RAM utilization, free disk space, and active compute device via `psutil`.
  - `get_logger(name)`: Dual file (`logs/detectobject.log`) + console structured logger.

### 2.2 `datasets/dataset_loader.py`
Implements `DatasetManager` with defensive validation at every step:
1. **Pre-flight Disk Check**: Verifies sufficient free space before network downloads.
2. **Resilient Download**: Streams archive to a `.part` temporary file, retries up to 3 times with backoff, validates `Content-Length`, checks ZIP structural integrity via `zipfile.testzip()`, computes **SHA-256**, and atomically promotes `.part` → `.zip`. Failed downloads raise `RuntimeError` and clean up partial files.
3. **Zip-Slip Safe Extraction**: Validates every archive member path resolves inside the extraction root.
4. **Image Integrity & Deduplication**: Validates images with both Pillow `Image.verify()` and OpenCV `cv2.imread()`, removing corrupt/empty files and deduplicating identical files via SHA-256 content hash.
5. **Label Validation & Normalization**: Filters out-of-range class IDs, NaN/Inf values, non-positive sizes, and degenerate tiny areas (`< 1e-6`), clipping slightly protruding bounding boxes strictly inside `[0.0, 1.0]`.
6. **Leak-Free Stratified Splitting**: Builds reproducible (`seed=42`) `train / val / test` splits (~70/15/15) interleaved across primary classes and asserts zero overlap in both SHA-256 hashes and filenames (`verify_no_leakage()`).
7. **Manifests & YAML**: Exports `datasets/processed/dataset.yaml`, `datasets/manifests/dataset_manifest.json`, and `datasets/manifests/split_manifest.json`.

### 2.3 `models/detector.py` & `models/tracker.py`
- **`ObjectDetector`**:
  - Loads `models/weights/best.pt` by default with automatic fallback to `models/weights/yolo11n.pt` (unless `strict_weights=True`, which raises `FileNotFoundError`).
  - Cached once per session in Streamlit via `@st.cache_resource` so model weights are never reloaded on UI interactions.
  - Provides unified `detect()` and `track()` methods returning typed `DetectionResult` objects containing `Detection` instances (`bbox`, `class_id`, `class_name`, `confidence`, `track_id`, `center`, `area`).
- **`OpenVocabularyDetector`**:
  - Optional lazy-loaded `YOLOWorld` wrapper supporting user-specified text prompts via `model.set_classes(prompts)`.
  - Filters strictly by confidence threshold; never labels low-confidence boxes as `"unknown"`.
- **`ObjectTracker`**:
  - Wraps Ultralytics `bytetrack.yaml` and `botsort.yaml` with persistent per-ID trajectory buffers (`deque(maxlen=30)`), active-track lifetime windows (`max_age_frames=30`) for occlusion resilience, and clean `reset()`.

### 2.4 `detection/`
- **`object_counter.py` (`ObjectCounter`)**:
  - Explicitly separates `frame_total` (detections in the current frame) from `unique_tracked_total` (distinct track IDs observed across time).
  - Supports directional virtual line crossing (`IN` when crossing in the positive axis direction, `OUT` in the negative direction), guarding against oscillation jitter by recording crossed IDs per direction.
  - Supports normalized rectangular ROI counting (`roi_current_count` and `roi_unique_total`).
- **`visualizer.py` (`DetectionVisualizer`)**:
  - Draws crisp bounding boxes with subtle corner accents, adaptive luminance-aware label pills (`#ID class conf%`), optional trajectory trails, semi-transparent ROI zone overlay, counting line badges, and FPS/count HUD.
- **`image_detector.py` (`ImageDetector`)**:
  - Handles EXIF orientation transpose, format validation (`jpg`, `jpeg`, `png`, `webp`), corrupt payload rejection, side-by-side RGB output, PNG byte encoding, and optional disk export.
- **`video_detector.py` (`VideoDetector`)**:
  - Validates readability before processing, supports `frame_skip`, resolution capping (`max_dimension=1280`), progress callbacks, and exports **H.264 (`libx264`, `yuv420p`) MP4** using PyAV (`av`) so output videos play natively in Chrome/Edge/Firefox HTML5 `<video>` elements (with automatic OpenCV `mp4v` fallback).
- **`webcam_detector.py` (`WebcamDetector`)**:
  - Uses DirectShow (`cv2.CAP_DSHOW`) on Windows with MSMF fallback for fast camera startup.
  - Runs capture + inference inside a dedicated background daemon thread protected by `threading.RLock`, allowing live slider adjustments (confidence, IoU, resolution, tracker, line/ROI) without dropping or re-opening the camera.
  - Guarantees hardware release (`cap.release()`) in `stop()`, `switch_camera()`, page-navigation hooks, and `atexit`.
  - Captures annotated PNG screenshots alongside structured JSON telemetry into `outputs/screenshots/`.

### 2.5 `training/trainer.py` & `evaluation/`
- **`YOLOTrainer`**:
  - Enforces fixed global seeds (`random`, `numpy`, `torch`, `deterministic=True`).
  - Automatically computes safe batch sizes from available VRAM or RAM (`choose_safe_batch_size()`).
  - Applies transfer-learning backbone freezing (`freeze=10`), cosine LR decay (`cos_lr=True`), and gentle bias warmup (`warmup_bias_lr=0.001`) to avoid BatchNorm collapse on compact datasets.
  - Catches memory-pressure `RuntimeError` exceptions, halves batch size, clears garbage/CUDA cache, and retries safely up to 2 times.
  - Backs up existing `models/weights/best.pt` to `models/weights/backups/best_backup_<timestamp>.pt` before overwriting.
- **`EvaluationEngine` & `EvaluationReportGenerator`**:
  - Runs Ultralytics `model.val(split="test")` strictly on the held-out independent test split.
  - Benchmarks real inference latency (mean, median, P95 ms) and FPS across `320px`, `480px`, and `640px` after 3 warmup passes.
  - Generates `reports/evaluation_metrics.json`, `reports/evaluation_report.md`, and 7 high-DPI evaluation plots in `reports/plots/`.

---

## 3. UI Streaming Architecture (Live Webcam in Streamlit)

Traditional `while True:` loops inside Streamlit block UI buttons and trigger full-page flicker whenever controls change. DetectObject solves this by pairing:
1. A persistent background `WebcamDetector` thread holding the latest `WebcamFramePacket`.
2. Streamlit's `@st.fragment(run_every=0.12)` rendering only the video canvas and live telemetry metrics at ~8–12 Hz while Start, Stop, Pause, Resume, Camera Switch, Screenshot, and threshold sliders remain instantaneously responsive.
3. A sidebar navigation guard that calls `stop_webcam_if_active()` whenever the user leaves the **Live Webcam** page.
