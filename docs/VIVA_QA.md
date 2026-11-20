# DetectObject — Viva & Technical Defense Q&A

### Q1. Why did you choose YOLO11n over RT-DETR or Grounding DINO as the primary detector?
**Answer:** Target deployment includes standard laptops without dedicated NVIDIA GPUs (our test machine is an AMD Ryzen 5 7520U with 5.74 GB usable RAM and no CUDA GPU). YOLO11n has only 2.62M parameters (6.5 GFLOPs, 5.4 MB) and achieves **21.29 FPS at 320px on CPU**, whereas RT-DETR-L (~32M params) and Grounding DINO (~172M+ params) consume 1–3 GB RAM and run at <1–3 FPS on CPU. Meanwhile, DetectObject still supports optional text-prompted open-vocabulary detection via lazy-loaded **YOLO-World v2** without slowing down startup.

### Q2. What is the difference between per-frame detection count and unique tracked object count?
**Answer:** Per-frame count (`frame_total`) is the instantaneous number of bounding boxes detected in a single frame. Summing per-frame counts across 100 frames of 2 stationary people would falsely report 200 people. Unique tracked count (`unique_tracked_total`) uses ByteTrack or BoT-SORT tracking IDs stored in a Python `set[int]` (globally and per class), so the same 2 tracked people across 100 frames are accurately counted as **2 unique objects**.

### Q3. How do ByteTrack and BoT-SORT differ?
**Answer:** Both are multi-object trackers built around Kalman Filter motion prediction and Hungarian matching. **ByteTrack** associates *both* high-confidence and low-confidence detections in a two-stage matching cascade, making it fast and effective during brief partial occlusions. **BoT-SORT** extends ByteTrack with camera motion compensation (CMC) and optional Re-ID features, which helps when the camera itself moves, at a slightly higher CPU cost. DetectObject lets users switch between both trackers cleanly.

### Q4. How does your virtual line-crossing counter avoid double-counting an object oscillating near the line?
**Answer:** For each tracked object ID, `ObjectCounter` compares its previous frame centroid against its current centroid relative to the line coordinate. Crossing from negative to positive increments `IN` only if that track ID is not already in `_crossed_in_ids`; crossing in reverse increments `OUT` only if not already in `_crossed_out_ids`. This eliminates jitter-induced duplicate counts.

### Q5. What went wrong during your first training run, and how did you diagnose and fix it?
**Answer:** In Run 1 (`batch=2`, `AdamW lr=0.001`, unfrozen backbone, default `warmup_bias_lr=0.1`), inspecting `results.csv` revealed that `val/cls_loss` spiked to `4.20` and `val mAP@0.5` collapsed to `0.2727` (`recall=0.1456`). Two root causes were identified: (1) batch size 2 destabilized pretrained BatchNorm running statistics, and (2) an aggressive bias warmup + unfrozen backbone caused catastrophic forgetting on a 90-image dataset. Switching to transfer-learning best practices (`freeze=10` backbone layers, `batch=8`, `SGD lr=0.0003`, `warmup_bias_lr=0.001`, `cos_lr=True`, `mosaic=0.0`) raised validation `mAP@0.5` from **0.2727 → 0.6876** and cut training time in half (167s → 84s).

### Q6. How do you guarantee there is no data leakage between train, validation, and test splits?
**Answer:** `DatasetManager` computes the **SHA-256 hash** of every image file, removes exact duplicates prior to splitting, performs deterministic (`seed=42`) stratified splitting into `train` (90), `val` (19), and `test` (19), and then executes `verify_no_leakage()`, which asserts empty set intersections for both SHA-256 digests and filenames across all split pairs before writing `split_manifest.json`.

### Q7. Explain Precision, Recall, F1-Score, mAP@0.5, and mAP@0.5:0.95.
**Answer:**
- **Precision** ($TP / (TP + FP)$): Fraction of predicted bounding boxes that are correct.
- **Recall** ($TP / (TP + FN)$): Fraction of ground-truth objects successfully detected.
- **F1-Score** ($2PR / (P + R)$): Harmonic mean balancing precision and recall.
- **mAP@0.5**: Mean Average Precision across classes where a predicted box counts as True Positive if its Intersection-over-Union (IoU) with ground truth is $\ge 0.50$.
- **mAP@0.5:0.95**: Stricter COCO metric averaging mAP across 10 IoU thresholds from $0.50$ to $0.95$ in steps of $0.05$, heavily rewarding tight localization.
On our independent test split, `best.pt` measured **P = 0.6913, R = 0.5563, F1 = 0.6165, mAP@0.5 = 0.6039, mAP@0.5:0.95 = 0.4527**.

### Q8. Why does DetectObject use PyAV (`libx264`) for video export instead of relying solely on OpenCV `mp4v`?
**Answer:** OpenCV's default `mp4v` (MPEG-4 Part 2) codec often fails to render inside HTML5 `<video>` tags in modern browsers (Chrome/Edge/Firefox). DetectObject uses PyAV (`av`) to encode standard **H.264 (`libx264`, `yuv420p`) MP4** streams for universal browser playback, while retaining automatic fallback to OpenCV `mp4v` if PyAV is unavailable.

### Q9. How do you achieve smooth live webcam detection inside Streamlit without freezing controls or leaking camera resources?
**Answer:** `WebcamDetector` runs OpenCV capture (`cv2.CAP_DSHOW` on Windows) and YOLO inference in a background thread protected by `threading.RLock`. The Streamlit UI uses `@st.fragment(run_every=0.12)` to refresh *only* the video canvas and telemetry counters, keeping Start, Stop, Pause, Resume, Screenshot, Camera Switch, and threshold sliders responsive without restarting the camera. Camera release (`cap.release()`) is enforced in `stop()`, `switch_camera()`, sidebar page-change guards, and an `atexit` hook.

### Q10. Can a remote web server access a user's local webcam through OpenCV?
**Answer:** No—OpenCV `cv2.VideoCapture(0)` accesses cameras physically attached to the machine running Python. For remote browser users, DetectObject provides **Browser Camera Capture** (`st.camera_input`), which invokes the browser's `navigator.mediaDevices.getUserMedia` API with proper browser permission prompts, while keeping local continuous streaming available on localhost and in standalone desktop mode (`python app.py --webcam`).

### Q11. What happens if a user requests Open-Vocabulary detection or uploads a low-confidence scene—do you label low-confidence objects as "unknown"?
**Answer:** No. A low-confidence detection is simply filtered out by the confidence threshold (`conf`) and is never labeled as `"unknown"`, because low confidence can result from motion blur, lighting, or occlusion rather than a novel category.

### Q12. How does the system protect against losing trained weights or crashing from Out-of-Memory (OOM) errors?
**Answer:** Before copying a newly trained `best.pt` to `models/weights/best.pt`, `backup_existing_weights()` archives the prior checkpoint into `models/weights/backups/best_backup_<timestamp>.pt`. During training, `YOLOTrainer` automatically picks a RAM/VRAM-safe batch size and wraps `model.train()` in an OOM catch block that halves batch size, clears garbage/CUDA cache, and retries safely up to 2 times.

### Q13. How do you ensure the application remains usable without a GPU?
**Answer:** `detect_compute_device("auto")` checks CUDA and MPS and gracefully falls back to `cpu`. Models are cached once (`@st.cache_resource`), default webcam inference uses `320px` (measured **21.29 FPS** on CPU), and configurable `frame_skip` allows further speed scaling on lower-end CPUs.

### Q14. How did you verify the system end-to-end?
**Answer:** We built an 18-test automated `pytest` suite covering configuration, GPU detection, CPU fallback, dataset download failure handling, corrupted/duplicate filtering, leak-free splitting, strict missing-weight errors, image/video pipelines, invalid media rejection, ByteTrack ID persistence, counting math, unavailable camera errors, and real integrated webcam capture/pause/screenshot/release—all 18/18 passing.
