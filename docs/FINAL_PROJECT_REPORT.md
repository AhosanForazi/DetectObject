# DetectObject: AI-Powered Real-Time Object Detection System with Live Webcam
## Final Academic & Engineering Project Report

---

## 1. Project Overview

**DetectObject** is a complete, production-oriented computer vision software system engineered to detect, classify, track, and count multiple objects across static images, uploaded video files, and real-time live webcam streams. Designed around **Ultralytics YOLO11**, **PyTorch**, **OpenCV**, **PyAV**, and **Streamlit**, the system unites an automated data hygiene and training pipeline, independent test-set evaluation, multi-object tracking (ByteTrack & BoT-SORT), directional line-crossing and ROI zone counting, optional open-vocabulary text-prompt detection (YOLO-World), and a responsive 9-page web dashboard.

---

## 2. Problem Statement

Real-time object detection is foundational to smart surveillance, retail traffic analytics, workplace PPE verification, accessibility tools, and autonomous systems. However, many academic prototypes suffer from four practical gaps:
1. **Hardware Dependency:** Heavy transformer detectors assume high-end CUDA GPUs and fail to deliver interactive frame rates on standard consumer CPU laptops.
2. **Counting Ambiguity:** Naive frame-level summation conflates repeated detections of the same object across consecutive frames with distinct physical objects.
3. **Fragile Media & Webcam Handling:** Prototype scripts lack format validation, export browser-incompatible video codecs, freeze UI controls during streaming, or leave hardware camera locks dangling on exit.
4. **Unverified ML Pipelines:** Missing checksums, corrupted labels, train/test data leakage, and unbacked checkpoint overwrites undermine reproducibility and scientific integrity.

DetectObject addresses all four challenges in a single, cleanly architected, locally deployable system.

---

## 3. Objectives

1. **Multi-Modal Detection:** Support `JPG/JPEG/PNG/WEBP` images, `MP4/AVI/MOV/MKV/WEBM` videos, and real-time local + browser webcam streams.
2. **Real-Time Live Webcam Operation:** Deliver start, stop, pause/resume, camera switching, live threshold tuning, FPS telemetry, screenshot saving, and guaranteed safe hardware release.
3. **Persistent Tracking & Accurate Counting:** Integrate ByteTrack and BoT-SORT to assign stable track IDs and provide per-frame, class-wise, unique tracked, directional line-crossing, and ROI counts.
4. **Hardware-Adaptive Performance:** Run interactively on CPU-only laptops while automatically accelerating on NVIDIA CUDA or Apple MPS when available.
5. **Reproducible End-to-End ML Workflow:** Implement verified dataset acquisition, corruption/duplicate removal, leak-free `train/val/test` splits, reproducible transfer learning, and independent test-set evaluation.
6. **Optional Open-Vocabulary Capability:** Allow text-prompted detection via YOLO-World without bloating core startup time or memory usage.

---

## 4. Literature Review and Model Selection

Modern object detectors fall into two primary paradigms: **single-stage CNN/hybrid detectors** (YOLO family) and **end-to-end transformer detectors** (DETR / RT-DETR / Grounding DINO).

| Candidate Model | Paradigm | Approx. Params | CPU Suitability (Ryzen 5, 6 GB RAM) | Multi-Object Tracking | Licensing Notes | Decision |
|---|---|---:|---|---|---|---|
| **YOLO11n** | Single-stage anchor-free CNN/C3k2+C2PSA | **2.62M** (6.5 GFLOPs) | **Excellent (21.3 FPS @ 320px)** | Native ByteTrack & BoT-SORT | Ultralytics AGPL-3.0 | **Selected Baseline** |
| **YOLO11s** | Single-stage anchor-free CNN | 9.4M | Moderate (~7–10 FPS) | Native ByteTrack & BoT-SORT | Ultralytics AGPL-3.0 | Optional upgrade |
| **RT-DETR-L** | Hybrid CNN + Transformer | ~32.0M | Poor on CPU (<2 FPS, >1.2 GB RAM) | Supported via Ultralytics | Apache-2.0 / AGPL-3.0 | GPU-only alternative |
| **YOLO-World v2-S** | Vision-Language Open-Vocab YOLO | ~12.8M | Moderate (on-demand image mode) | Image / Prompt mode | GPL-3.0 / AGPL-3.0 | **Selected Optional Open-Vocab** |
| **Grounding DINO** | Heavy Vision-Language Transformer | 172M+ | Unsuitable (>2.5 GB RAM, <1 FPS CPU) | External required | Apache-2.0 | Excluded for CPU target |

**Selection Rationale:** Inspection of the execution environment revealed an **AMD Ryzen 5 7520U (4C/8T)** with **5.74 GB usable RAM** and integrated AMD Radeon 610M graphics (no CUDA). **YOLO11n** offered the optimal accuracy-to-latency frontier, minimal memory footprint (~300 MB runtime), native ByteTrack/BoT-SORT tracking, and seamless CPU/CUDA portability. **YOLO-World v2-S** was integrated as a lazy-loaded optional module for user-specified text prompts.

---

## 5. System Architecture

DetectObject is organized into decoupled Python packages:
- **`configuration/`**: Central `config.yaml`, canonical 80-class `classes.yaml`, automatic compute device detection (`detect_compute_device()`), live resource inspection (`get_system_resources()`), and dual-sink structured logging (`logs/detectobject.log`).
- **`datasets/`**: `DatasetManager` handles checksum-verified downloads, Zip-Slip safe extraction, Pillow + OpenCV validation, SHA-256 deduplication, YOLO label sanitization/normalization, leak-free stratified splitting, and JSON/YAML manifest export.
- **`models/`**: `ObjectDetector` (cached YOLO11 loader with strict missing-weight error handling and CPU fallback), `OpenVocabularyDetector` (lazy-loaded YOLO-World), and `ObjectTracker` (ByteTrack / BoT-SORT state & trajectory management).
- **`detection/`**: `ImageDetector`, `VideoDetector` (PyAV H.264 `libx264` MP4 exporter with OpenCV fallback), thread-safe `WebcamDetector`, `ObjectCounter`, and `DetectionVisualizer`.
- **`training/` & `evaluation/`**: Reproducible `YOLOTrainer` (OOM recovery, checkpoint backups, fixed seeds) and `EvaluationEngine` + `EvaluationReportGenerator` (independent test metrics, latency/FPS benchmarks, 7 plots, Markdown/JSON reports).
- **`application/` & `app.py`**: 9-page responsive Streamlit dashboard with Dark/Light themes launched via `python app.py`.

---

## 6. Dataset Description

- **Dataset Name:** COCO128 (first 128 images of MS COCO 2017 `train2017` packaged in YOLO format by Ultralytics)
- **Source URL:** `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`
- **Archive Size & Verified SHA-256:** `6,983,030` bytes (`6.66 MB`), SHA-256 `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- **Licensing:** COCO annotations are licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**; images follow Flickr Terms of Use.
- **Taxonomy:** 80 canonical everyday object classes (person, vehicles, animals, household/office items such as laptop, cell phone, bottle, chair, book, cup, etc.).
- **Suitability:** Selected intentionally after checking local disk (42.9 GiB free), RAM (5.74 GB usable), and CPU-only compute so end-to-end download, cleaning, training, and evaluation execute reliably without exhausting system resources.

---

## 7. Data Preprocessing

Executed automatically via `python training/dataset_prepare.py`:
1. **Integrity Verification:** Verified HTTP `Content-Length`, ZIP CRC integrity (`zipfile.testzip()`), and SHA-256 digest.
2. **Image Validation & Deduplication:** Every image passed both Pillow `Image.verify()` and OpenCV `cv2.imread()`. Content SHA-256 hashing verified **0 corrupted** and **0 duplicate** images across all **128** raw samples.
3. **Annotation Cleaning & Normalization:** Parsed all **929** ground-truth bounding boxes, enforcing valid integer class IDs (`0..79`), finite coordinates, positive dimensions, minimum area (`>= 1e-6`), and strict clamping into `[0.0, 1.0]`. Result: **929 valid boxes kept, 0 invalid removed**.
4. **Leak-Free Stratified Splitting (`seed=42`):** Partitioned into **90 Train (70.3%)**, **19 Validation (14.8%)**, and **19 Independent Test (14.8%)** images. Automated `verify_no_leakage()` confirmed zero hash or filename overlap across all splits and persisted `datasets/processed/dataset.yaml`, `datasets/manifests/dataset_manifest.json`, and `datasets/manifests/split_manifest.json`.

---

## 8. Model Training Methodology

Training was executed reproducibly (`seed=42`, `deterministic=True`, `workers=0`, `cache=False`) via `python training/train.py` on the AMD Ryzen 5 7520U CPU.

### Iterative Engineering Diagnosis & Improvement
| Run | Configuration | Duration | Val Precision | Val Recall | Val mAP@0.5 | Val mAP@0.5:0.95 | Diagnosis |
|---|---|---:|---:|---:|---:|---:|---|
| **Run 1** | `batch=2`, `AdamW`, `lr=0.001`, unfrozen, default bias warmup `0.1` | 167.06s | 0.6449 | 0.1456 | 0.2727 | 0.1651 | Small batch destabilized BatchNorm; high bias warmup caused catastrophic forgetting |
| **Run 2 (Final)** | `batch=8`, `SGD`, `lr=0.0003`, `freeze=10`, `warmup_bias_lr=0.001`, `cos_lr=True`, `mosaic=0.0` | **84.43s** | **0.7021** | **0.6144** | **0.6876** | **0.5077** | Stable transfer learning preserved pretrained backbone; **+2.52x mAP@0.5** and **2x faster** |

Before saving Run 2's weights to `models/weights/best.pt`, `YOLOTrainer` automatically archived Run 1's checkpoint to `models/weights/backups/best_backup_20261118_094749.pt`.

---

## 9. Evaluation Results

All metrics below were **directly measured** on the independent held-out **test split** (19 images, 106 ground-truth object instances, zero overlap with training or validation) via `python evaluation/evaluate.py`:

### 9.1 Overall Detection Metrics (`best.pt`, `imgsz=320`, `cpu`)
| Metric | Measured Value |
|---|---:|
| **Precision (mean)** | **0.6913** |
| **Recall (mean)** | **0.5563** |
| **F1-Score** | **0.6165** |
| **mAP@0.5** | **0.6039** |
| **mAP@0.5:0.95** | **0.4527** |

### 9.2 Measured Hardware Inference Speed & FPS Benchmark
Measured on **AMD Ryzen 5 7520U CPU** (`torch 2.15.0+cpu`, 3 warmup passes + 20 timed iterations per resolution):

| Input Resolution | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Measured FPS | Type |
|---:|---:|---:|---:|---:|---|
| **320 × 320** | **46.98** | 44.73 | 66.32 | **21.29 FPS** | Measured |
| **480 × 480** | **82.90** | 79.88 | 131.49 | **12.06 FPS** | Measured |
| **640 × 640** | **166.86** | 148.88 | 287.00 | **5.99 FPS** | Measured |

### 9.3 Generated Evaluation Graphs
- Training Loss Curve: `reports/plots/training_loss.png`
- Validation Loss & Detection Metrics: `reports/plots/validation_loss.png`
- Test Split Confusion Matrix: `reports/plots/confusion_matrix.png`
- Test Split Precision-Recall Curve: `reports/plots/pr_curve.png`
- Class-Wise Average Precision: `reports/plots/class_performance.png`
- Measured FPS & Latency Benchmark: `reports/plots/fps_benchmark.png`

---

## 10. Live Webcam Implementation

Live webcam detection is implemented cleanly in `detection/webcam_detector.py` and integrated into the **Live Webcam** page of `application/dashboard.py`:
- **Hardware Enumeration & Switching:** `enumerate_cameras()` safely probes indices using DirectShow (`cv2.CAP_DSHOW`) on Windows (with default backend fallback) and releases every probed handle immediately.
- **Threaded Real-Time Capture & Inference:** `WebcamDetector` runs frame acquisition, ByteTrack/BoT-SORT tracking, counting, and visualization on a dedicated background thread protected by `threading.RLock`.
- **Responsive Streamlit Fragment Streaming:** `@st.fragment(run_every=0.12)` refreshes the live canvas, FPS meter, frame object count, unique tracked ID count, and class table without interrupting Start, Stop, Pause/Resume, Camera Switch, Screenshot Capture, or live slider changes.
- **Guaranteed Resource Release:** `stop()` joins the capture thread and calls `cap.release()` in a `try...finally` block; automatic release triggers on Stop, camera switch, sidebar page navigation, and process exit (`atexit`).
- **Dual Access Modes:** Supports continuous local OpenCV streaming, browser-permission `st.camera_input` snapshots for remote browsers, and standalone native desktop OpenCV (`python app.py --webcam`).
- **Hardware Verification:** Live initialization, real frame capture (`640x480`), pause/resume, annotated screenshot saving, and clean release were verified against the local **Integrated Camera** during automated `pytest` execution.

---

## 11. Visual Artifacts & Screenshots

Generated artifacts stored in the repository for visual inspection and presentation:
- **Evaluation Charts:** `reports/plots/{training_loss,validation_loss,confusion_matrix,pr_curve,class_performance,fps_benchmark}.png`
- **Sample Annotated Image Detection:** `outputs/images/sample_verification_detected.png`
- **Sample Tracked & Counted Video Export:** `outputs/videos/sample_verification_tracked.mp4`
- **Live Webcam Annotated Capture:** `outputs/screenshots/live_webcam_verification_*.png`

---

## 12. Limitations

1. **Compact Fine-Tuning Dataset:** Fine-tuning on 90 training / 19 validation / 19 test images covers a subset of the 80 COCO classes per split; infrequent classes with 1 test instance exhibit high variance in per-class AP.
2. **CPU-Only Benchmarking:** No NVIDIA CUDA GPU was available on the development laptop; GPU throughput was therefore not measured.
3. **Extended Occlusion & Lighting:** Severe low-light noise, heavy motion blur, or prolonged complete occlusion can still cause missed detections or new track ID assignments.
4. **Open-Vocabulary Overhead:** YOLO-World v2 requires an extra ~25 MB weight download on first use and runs slower on CPU than YOLO11n, so it is scoped as an optional on-demand mode.

---

## 13. Future Improvements

1. **ONNX Runtime / OpenVINO INT8 Quantization:** Exporting `best.pt` to ONNX/OpenVINO INT8 can boost CPU webcam throughput from ~21 FPS to 35–50+ FPS.
2. **In-App Custom Dataset Training UI:** Allow users to upload custom YOLO datasets and launch fine-tuning jobs directly from the dashboard with live loss charts.
3. **Interactive Polygon ROI & Multi-Line Zones:** Support arbitrary polygon regions and multi-line flow analytics for complex intersection or retail floor plans.
4. **Multi-Camera Grid & Alert Webhooks:** Monitor simultaneous RTSP/USB feeds with configurable class-threshold alerts (email/Slack/webhook).

---

## 14. Conclusion

DetectObject successfully delivers an end-to-end, thoroughly verified AI object detection, tracking, and counting platform. Every phase—from environment auditing and SHA-256-verified leak-free dataset engineering to transfer-learning debugging (+2.5x mAP@0.5 gain), independent test evaluation (0.6039 mAP@0.5, 21.29 FPS on CPU), real-time integrated webcam verification, H.264 video export, 18/18 passing automated tests, and a modern 9-page Streamlit dashboard—was built, executed, measured, and validated cleanly on real local hardware.

---

## 15. References

1. Jocher, G., Qiu, J., & Chaurasia, A. (2024). *Ultralytics YOLO11*. Ultralytics GitHub & Documentation. https://docs.ultralytics.com/models/yolo11/
2. Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014. https://cocodataset.org/
3. Zhang, Y., Sun, P., Jiang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
4. Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv:2206.14651.
5. Cheng, T., Song, L., Ge, Y., et al. (2024). *YOLO-World: Real-Time Open-Vocabulary Object Detection*. CVPR 2024.
6. Zhao, Y., Lv, W., Xu, S., et al. (2024). *DETRs Beat YOLOs on Real-Time Object Detection (RT-DETR)*. CVPR 2024.
7. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
8. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
