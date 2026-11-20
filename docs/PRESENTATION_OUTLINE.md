# DetectObject — Project Presentation Outline (15 Slides)

## Slide 1: Title Slide
- **Title:** DetectObject: AI-Powered Real-Time Object Detection, Tracking & Counting System
- **Subtitle:** End-to-End Computer Vision Pipeline with Live Webcam & Interactive Dashboard
- **Key Visual:** Annotated real-time detection sample (`outputs/images/`) + system architecture badge

## Slide 2: Problem Statement & Motivation
- Growing demand for real-time visual intelligence across retail analytics, safety monitoring, robotics, and smart classrooms.
- Practical challenges: heavy models fail on CPU-only laptops, ad-hoc scripts lack tracking/counting accuracy (confusing per-frame sums with unique objects), and browser/webcam pipelines often leak camera locks.

## Slide 3: Project Objectives
- Multi-modal detection: Images (JPG/PNG/WEBP), Videos (MP4/AVI/MOV), and Live Webcams.
- Persistent multi-object tracking (ByteTrack / BoT-SORT) with unique IDs, line-crossing, and ROI counting.
- CPU-first efficiency with automatic CUDA/MPS scaling and reproducible dataset/training/evaluation pipelines.

## Slide 4: Literature & Model Selection
- Evaluated **YOLO11n**, **YOLO11s**, **RT-DETR-L**, **YOLO-World v2-S**, and **Grounding DINO**.
- Selected **YOLO11n** (2.62M params, 6.5 GFLOPs, 5.4 MB) for real-time CPU responsiveness + native Ultralytics ByteTrack/BoT-SORT support.
- Integrated optional **YOLO-World v2-S** (lazy-loaded) for arbitrary text-prompted open-vocabulary detection.

## Slide 5: System Architecture
- Modular 5-layer Python architecture (`configuration/`, `datasets/`, `models/`, `detection/`, `training/`, `evaluation/`, `application/`).
- Cached single-load detector (`@st.cache_resource`), PyAV H.264 browser video encoder, and thread-safe webcam capture engine.

## Slide 6: Dataset Engineering & Data Hygiene
- Dataset: **COCO128** (CC BY 4.0 annotations, 6.98 MB, SHA-256 verified).
- Automated checks: Pillow + OpenCV corruption detection, SHA-256 deduplication, YOLO bounding-box clamping/validation (929 valid boxes).
- Leak-free stratified splits: **90 Train / 19 Val / 19 Test** with automated hash & filename disjointness verification.

## Slide 7: Training Methodology & Engineering Lesson Learned
- Reproducible training (`seed=42`, automatic safe batch selection, OOM retry, checkpoint backup before overwrite).
- **Real Engineering Case Study**:
  - *Run 1* (`batch=2`, `AdamW lr=0.001`, unfrozen): BatchNorm instability caused catastrophic forgetting (`Val mAP@0.5 = 0.2727`).
  - *Run 2* (`batch=8`, `SGD lr=0.0003`, `freeze=10`, `warmup_bias_lr=0.001`, `cos_lr=True`): Stabilized training in 84.4s → **`Val mAP@0.5 = 0.6876` (+2.5x)**.
- **Visual:** `reports/plots/training_loss.png` & `reports/plots/validation_loss.png`.

## Slide 8: Independent Test Evaluation Results
- Strictly measured on the held-out **19-image test split (106 instances)**:
  - **Precision:** `0.6913` | **Recall:** `0.5563` | **F1-Score:** `0.6165`
  - **mAP@0.5:** `0.6039` | **mAP@0.5:0.95:** `0.4527`
- **Visual:** `reports/plots/confusion_matrix.png`, `reports/plots/pr_curve.png`, `reports/plots/class_performance.png`.

## Slide 9: Real-World CPU Speed Benchmarks
- Measured on **AMD Ryzen 5 7520U (CPU-only, 20 timed iterations + 3 warmups)**:
  - `320px`: **21.29 FPS** (46.98 ms mean)
  - `480px`: **12.06 FPS** (82.90 ms mean)
  - `640px`: **5.99 FPS** (166.86 ms mean)
- **Visual:** `reports/plots/fps_benchmark.png`.

## Slide 10: Live Webcam Architecture & Safety
- Background threaded `WebcamDetector` (`cv2.CAP_DSHOW` on Windows) paired with Streamlit `@st.fragment(run_every=0.12)`.
- Instant Start/Stop/Pause/Resume, camera switcher, live threshold adjustment without camera flicker, screenshot capture (`PNG` + `JSON`), guaranteed `cap.release()`, and Browser `getUserMedia` fallback.

## Slide 11: Multi-Object Tracking & Directional Counting
- ByteTrack & BoT-SORT maintain persistent IDs across frames and brief occlusions.
- Clear analytical separation between **Frame Objects** vs. **Unique Tracked Objects**.
- Configurable directional virtual line (`IN` / `OUT` counts) and customizable ROI bounding zone.

## Slide 12: Dashboard UI & User Experience
- 9-page responsive Streamlit interface with Dark/Light mode, status pills, side-by-side image inspection, real-time video progress, evaluation gallery, and live `config.yaml` editor.

## Slide 13: Automated Quality Assurance
- 18/18 `pytest` unit & integration tests passing—covering dataset integrity, leak detection, missing weights, CPU fallback, image/video pipelines, ByteTrack IDs, counting math, and real integrated webcam capture & release.

## Slide 14: Limitations & Future Roadmap
- Limitations: Compact 128-image fine-tuning split, CPU-only benchmarking, extended full-occlusion ID switches.
- Roadmap: ONNX Runtime / OpenVINO INT8 quantization for 2x+ CPU FPS, custom dataset training UI, multi-camera grid monitoring, and polygon ROI drawing.

## Slide 15: Conclusion & Live Demo
- Summary of achieved deliverables and live demonstration of Image Detection, Video Tracking, and Live Webcam Detection (`python app.py`).
