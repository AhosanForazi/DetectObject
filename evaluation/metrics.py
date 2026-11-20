"""Independent test-set evaluation and real hardware inference speed benchmarking."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configuration.settings import (
    AppConfig,
    detect_compute_device,
    get_logger,
    get_system_resources,
    load_config,
)

logger = get_logger("detectobject.evaluation")


@dataclass
class EvaluationMetrics:
    """Strictly measured evaluation results on the independent test split."""

    model_weights: str
    dataset_yaml: str
    split_evaluated: str
    device: str
    imgsz: int
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float
    class_metrics: list[dict[str, Any]] = field(default_factory=list)
    fps_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    ultralytics_val_dir: str = ""
    hardware_info: dict[str, Any] = field(default_factory=dict)
    evaluated_at_utc: str = ""
    measurement_note: str = (
        "All metrics and FPS benchmarks below were directly measured on the independent held-out "
        "test split and local hardware. No values were fabricated or estimated."
    )


class EvaluationEngine:
    """Evaluates a trained checkpoint on an independent test split and benchmarks latency/FPS."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.device_info = detect_compute_device(self.config.get("model", "device", default="auto"))
        self.device = self.device_info["selected"]

    def benchmark_inference_speed(
        self,
        weights_path: Path,
        sample_images_dir: Path,
        resolutions: tuple[int, ...] = (320, 480, 640),
        iterations: int = 20,
    ) -> list[dict[str, Any]]:
        """Measure actual end-to-end inference latency (ms) and FPS on real test images."""
        from ultralytics import YOLO

        image_files = sorted(
            p for p in sample_images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not image_files:
            raise FileNotFoundError(f"No test images found for benchmarking in {sample_images_dir}")

        loaded_images = [cv2.imread(str(p)) for p in image_files[: min(10, len(image_files))]]
        loaded_images = [img for img in loaded_images if img is not None]
        if not loaded_images:
            raise RuntimeError("Failed to decode sample images for speed benchmarking.")

        model = YOLO(str(weights_path))
        benchmarks: list[dict[str, Any]] = []

        for res in resolutions:
            # Warmup runs excluded from timed measurements
            for _ in range(3):
                model.predict(loaded_images[0], imgsz=res, device=self.device, verbose=False)

            latencies_ms: list[float] = []
            for i in range(iterations):
                img = loaded_images[i % len(loaded_images)]
                t0 = time.perf_counter()
                model.predict(img, imgsz=res, device=self.device, verbose=False)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            mean_ms = float(np.mean(latencies_ms))
            median_ms = float(np.median(latencies_ms))
            p95_ms = float(np.percentile(latencies_ms, 95))
            fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            benchmarks.append(
                {
                    "imgsz": res,
                    "iterations": iterations,
                    "mean_latency_ms": round(mean_ms, 2),
                    "median_latency_ms": round(median_ms, 2),
                    "p95_latency_ms": round(p95_ms, 2),
                    "measured_fps": round(fps, 2),
                    "device": self.device,
                    "measurement_type": "measured",
                }
            )
            logger.info(
                "Benchmark @ %dpx on %s: mean=%.2f ms (p95=%.2f ms) -> %.2f FPS",
                res,
                self.device,
                mean_ms,
                p95_ms,
                fps,
            )

        return benchmarks

    def evaluate(
        self,
        weights_path: str | Path | None = None,
        dataset_yaml: str | Path | None = None,
        split: str = "test",
        imgsz: int | None = None,
    ) -> EvaluationMetrics:
        """Run Ultralytics validation on the independent test split and collect all metrics."""
        from ultralytics import YOLO

        w_path = (
            Path(weights_path)
            if weights_path
            else self.config.resolve_path(self.config.get("model", "default_weights", default="models/weights/best.pt"))
        )
        if not w_path.exists():
            raise FileNotFoundError(f"Model weights not found for evaluation: {w_path}")

        ds_yaml = (
            Path(dataset_yaml)
            if dataset_yaml
            else self.config.resolve_path("datasets/processed/dataset.yaml")
        )
        if not ds_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found for evaluation: {ds_yaml}")

        eval_imgsz = int(imgsz if imgsz is not None else self.config.get("training", "imgsz", default=320))
        reports_dir = self.config.resolve_path(self.config.get("evaluation", "reports_dir", default="reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running independent '%s' split evaluation for %s...", split, w_path.name)
        model = YOLO(str(w_path))
        val_results = model.val(
            data=str(ds_yaml),
            split=split,
            imgsz=eval_imgsz,
            device=self.device,
            workers=0,
            plots=True,
            project=str(reports_dir),
            name="ultralytics_test_eval",
            exist_ok=True,
            verbose=True,
        )

        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)

        # Class-wise metrics for all classes present in the test split
        class_metrics: list[dict[str, Any]] = []
        ap_class_indices = getattr(val_results.box, "ap_class_index", [])
        names = getattr(val_results, "names", {})
        for idx_pos, cls_idx in enumerate(ap_class_indices):
            cls_int = int(cls_idx)
            cls_name = names.get(cls_int, f"class_{cls_int}") if isinstance(names, dict) else str(names[cls_int])
            try:
                p_c, r_c, ap50_c, ap_c = val_results.box.class_result(idx_pos)
                f1_c = (2.0 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
                class_metrics.append(
                    {
                        "class_id": cls_int,
                        "class_name": str(cls_name),
                        "precision": round(float(p_c), 4),
                        "recall": round(float(r_c), 4),
                        "f1_score": round(float(f1_c), 4),
                        "ap50": round(float(ap50_c), 4),
                        "ap50_95": round(float(ap_c), 4),
                    }
                )
            except Exception as exc:
                logger.debug("Could not extract class_result for %s: %s", cls_name, exc)

        class_metrics.sort(key=lambda item: item["ap50"], reverse=True)

        # Real speed benchmark on the test images
        test_img_dir = self.config.resolve_path("datasets/processed/images/test")
        iters = int(self.config.get("evaluation", "benchmark_iterations", default=20))
        benchmarks = self.benchmark_inference_speed(
            weights_path=w_path,
            sample_images_dir=test_img_dir,
            resolutions=(320, 480, 640),
            iterations=iters,
        )

        val_save_dir = str(getattr(val_results, "save_dir", reports_dir / "ultralytics_test_eval"))
        metrics = EvaluationMetrics(
            model_weights=str(w_path),
            dataset_yaml=str(ds_yaml),
            split_evaluated=split,
            device=self.device,
            imgsz=eval_imgsz,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            class_metrics=class_metrics,
            fps_benchmarks=benchmarks,
            ultralytics_val_dir=val_save_dir,
            hardware_info=get_system_resources(),
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        metrics_json = reports_dir / "evaluation_metrics.json"
        metrics_json.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        logger.info(
            "Test evaluation saved to %s | P=%.4f R=%.4f F1=%.4f mAP@0.5=%.4f mAP@0.5:0.95=%.4f",
            metrics_json,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.map50,
            metrics.map50_95,
        )
        return metrics
