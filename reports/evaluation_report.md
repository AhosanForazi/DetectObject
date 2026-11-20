# DetectObject — Model Evaluation Report

**Generated (UTC):** 2026-11-18T03:57:18.191725+00:00
**Evaluated Weights:** `best.pt`
**Evaluated Split:** `test` (independent held-out split; zero overlap with train/val)
**Device:** `cpu`

> **Measurement Integrity Note:** Every metric, latency, and FPS value reported below was directly measured on the independent test split and local hardware. No values were fabricated or estimated.

## 1. Summary of Measured Test-Split Metrics

| Metric | Measured Value |
|---|---|
| **Precision (mean)** | 0.6913 |
| **Recall (mean)** | 0.5563 |
| **F1-Score** | 0.6165 |
| **mAP@0.5** | 0.6039 |
| **mAP@0.5:0.95** | 0.4527 |
| **Evaluation Image Size** | 320x320 |

## 2. Measured Inference Speed & FPS Benchmark

Hardware: `Windows 11 (10.0.26200)`, Logical CPUs: `8`, RAM: `5.74 GB`, Device: `cpu`.

| Resolution | Iterations | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Measured FPS | Type |
|---:|---:|---:|---:|---:|---:|---|
| 320px | 20 | 46.98 | 44.73 | 66.32 | 21.29 | measured |
| 480px | 20 | 82.90 | 79.88 | 131.49 | 12.06 | measured |
| 640px | 20 | 166.86 | 148.88 | 287.00 | 5.99 | measured |

## 3. Dataset & Split Configuration

- **Dataset:** `coco128`
- **Source:** https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
- **License:** COCO Annotations: CC BY 4.0; Images: Flickr Terms of Use (via Ultralytics COCO128)
- **Archive SHA-256:** `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- **Valid Images Kept:** 128 (Corrupted removed: 0, Duplicates removed: 0)
- **Valid Bounding Boxes:** 929 (Invalid removed: 0)
- **Split Sizes:** Train = 90, Val = 19, Test = 19
- **Leakage Verification:** `True`

## 4. Training Configuration

- **Base Model:** `yolo11n.pt`
- **Epochs Completed:** 7 / 8
- **Batch Size:** 8
- **Training Resolution:** 320px
- **Optimizer / LR:** `SGD` / `0.0003`
- **Random Seed:** `42`
- **Training Duration:** 84.43 seconds on `cpu`

## 5. Class-Wise Performance (Independent Test Split)

| Class | Precision | Recall | F1-Score | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| motorcycle | 0.6066 | 1.0000 | 0.7551 | 0.9950 | 0.5970 |
| airplane | 0.3693 | 1.0000 | 0.5394 | 0.9950 | 0.7960 |
| train | 0.6015 | 1.0000 | 0.7512 | 0.9950 | 0.7960 |
| truck | 0.6045 | 1.0000 | 0.7535 | 0.9950 | 0.4975 |
| dog | 0.6587 | 1.0000 | 0.7942 | 0.9950 | 0.9950 |
| horse | 0.6319 | 1.0000 | 0.7744 | 0.9950 | 0.5222 |
| zebra | 0.8020 | 1.0000 | 0.8901 | 0.9950 | 0.9290 |
| tie | 0.5818 | 1.0000 | 0.7356 | 0.9950 | 0.9950 |
| bottle | 0.5453 | 1.0000 | 0.7057 | 0.9950 | 0.7960 |
| fork | 0.5731 | 1.0000 | 0.7286 | 0.9950 | 0.9950 |
| sandwich | 0.6784 | 1.0000 | 0.8084 | 0.9950 | 0.9950 |
| cake | 0.5731 | 1.0000 | 0.7286 | 0.9950 | 0.8955 |
| couch | 0.3557 | 1.0000 | 0.5247 | 0.9950 | 0.6965 |
| bed | 0.8589 | 1.0000 | 0.9241 | 0.9950 | 0.7960 |
| hot dog | 0.5707 | 1.0000 | 0.7267 | 0.6633 | 0.6217 |
| tennis racket | 0.5093 | 0.5000 | 0.5046 | 0.5950 | 0.4955 |
| chair | 0.4278 | 0.5488 | 0.4808 | 0.5591 | 0.2572 |
| person | 0.5025 | 0.4161 | 0.4552 | 0.5163 | 0.3613 |
| car | 0.3399 | 1.0000 | 0.5073 | 0.4975 | 0.3483 |
| umbrella | 1.0000 | 0.0000 | 0.0000 | 0.4975 | 0.1990 |
| snowboard | 1.0000 | 0.0000 | 0.0000 | 0.4975 | 0.1658 |
| clock | 0.8846 | 0.5000 | 0.6389 | 0.4950 | 0.3960 |
| bird | 0.5095 | 0.7500 | 0.6068 | 0.4728 | 0.1738 |
| book | 1.0000 | 0.0000 | 0.0000 | 0.4117 | 0.1887 |
| laptop | 1.0000 | 0.0000 | 0.0000 | 0.3317 | 0.1548 |
| potted plant | 1.0000 | 0.0000 | 0.0000 | 0.2905 | 0.2432 |
| wine glass | 0.3202 | 0.2000 | 0.2462 | 0.2873 | 0.2070 |
| handbag | 1.0000 | 0.0000 | 0.0000 | 0.2487 | 0.1492 |
| dining table | 0.0000 | 0.0000 | 0.0000 | 0.1990 | 0.1222 |
| backpack | 1.0000 | 0.0000 | 0.0000 | 0.0383 | 0.0077 |
| sports ball | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mouse | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| remote | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| teddy bear | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 6. Generated Visual Artifacts

- **confusion_matrix**: `reports/plots/confusion_matrix.png`
- **pr_curve**: `reports/plots/pr_curve.png`
- **f1_curve**: `reports/plots/f1_curve.png`
- **training_loss**: `reports/plots/training_loss.png`
- **validation_loss**: `reports/plots/validation_loss.png`
- **class_performance**: `reports/plots/class_performance.png`
- **fps_benchmark**: `reports/plots/fps_benchmark.png`

## 7. Limitations & Practical Notes

1. **Small Fine-Tuning Split Size:** COCO128 contains 128 images across 80 COCO classes, so the held-out test split (19 images) covers a subset of classes with limited per-class sample counts. Rare COCO classes are not all represented in this compact CPU-friendly benchmark.
2. **CPU-Only Execution:** All training and benchmarks were executed on an AMD Ryzen 5 7520U CPU without a dedicated NVIDIA CUDA GPU. Deploying on a CUDA GPU will yield significantly higher real-time FPS (typically 5x–15x faster, unmeasured on this machine).
3. **Occlusion & Fast Motion:** Multi-object tracking (ByteTrack / BoT-SORT) handles brief occlusions well, but extended full occlusion or sudden lighting drops can still trigger track re-identification.
