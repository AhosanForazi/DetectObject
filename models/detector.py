"""Core object detection and optional open-vocabulary detection wrappers."""

from __future__ import annotations

import gc
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from configuration.settings import (
    AppConfig,
    PROJECT_ROOT,
    detect_compute_device,
    get_logger,
    load_config,
)

logger = get_logger("detectobject.detector")


@dataclass
class Detection:
    """Single detected (and optionally tracked) object."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 2) for v in self.bbox],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
        }


@dataclass
class DetectionResult:
    """Aggregated detection output for a single image or video frame."""

    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    fps: float = 0.0
    image_shape: tuple[int, int] = (0, 0)  # (height, width)
    model_name: str = ""
    device: str = "cpu"
    class_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "image_shape": list(self.image_shape),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 2),
            "total_count": self.total_count,
            "class_counts": self.class_counts,
            "detections": [d.to_dict() for d in self.detections],
        }


class ObjectDetector:
    """Ultralytics YOLO object detector with automatic device selection and CPU fallback."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "auto",
        config: AppConfig | None = None,
        strict_weights: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.device_info = detect_compute_device(device or self.config.get("model", "device", default="auto"))
        self.device: str = self.device_info["selected"]
        self.weights_path: Path = self._resolve_weights(weights_path, strict=strict_weights)
        self.model: Any = None
        self.class_names: dict[int, str] = {}
        self.model_info: dict[str, Any] = {}
        self._load_model()

    def _resolve_weights(self, weights_path: str | Path | None, strict: bool = False) -> Path:
        if weights_path is not None:
            candidate = Path(weights_path)
            if not candidate.is_absolute():
                rel = (PROJECT_ROOT / candidate).resolve()
                if rel.exists():
                    return rel
                if strict:
                    raise FileNotFoundError(f"Requested model weights not found: {rel}")
                return candidate
            if strict and not candidate.exists():
                raise FileNotFoundError(f"Requested model weights not found: {candidate}")
            return candidate

        default_rel = self.config.get("model", "default_weights", default="models/weights/best.pt")
        fallback_rel = self.config.get("model", "fallback_weights", default="models/weights/yolo11n.pt")
        default_path = self.config.resolve_path(default_rel)
        fallback_path = self.config.resolve_path(fallback_rel)

        if default_path.exists():
            return default_path
        if fallback_path.exists():
            logger.info("Default weights %s not found; using fallback %s", default_path.name, fallback_path.name)
            return fallback_path
        if strict:
            raise FileNotFoundError(f"Neither default ({default_path}) nor fallback ({fallback_path}) weights exist.")

        # Allow Ultralytics to download baseline yolo11n.pt directly into models/weights/
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        return fallback_path

    def _load_model(self) -> None:
        from ultralytics import YOLO

        weights_str = str(self.weights_path)
        logger.info("Loading YOLO model from %s on device=%s", weights_str, self.device)
        t0 = time.perf_counter()
        self.model = YOLO(weights_str)
        # Ensure Ultralytics-downloaded weights live inside models/weights/ if downloaded to CWD
        if not self.weights_path.exists():
            cwd_copy = Path.cwd() / self.weights_path.name
            if cwd_copy.exists():
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                cwd_copy.replace(self.weights_path)
                self.model = YOLO(str(self.weights_path))

        load_ms = (time.perf_counter() - t0) * 1000.0
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, list):
            self.class_names = {i: str(n) for i, n in enumerate(raw_names)}
        else:
            self.class_names = {int(k): str(v) for k, v in raw_names.items()}

        param_count = 0
        try:
            param_count = sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            pass

        size_mb = round(self.weights_path.stat().st_size / (1024 * 1024), 2) if self.weights_path.exists() else 0.0
        self.model_info = {
            "weights_path": str(self.weights_path),
            "weights_name": self.weights_path.name,
            "device": self.device,
            "num_classes": len(self.class_names),
            "parameters": param_count,
            "file_size_mb": size_mb,
            "load_time_ms": round(load_ms, 1),
        }
        logger.info(
            "Model loaded: %s (%d classes, %d params, %.2f MB) in %.1f ms",
            self.weights_path.name,
            len(self.class_names),
            param_count,
            size_mb,
            load_ms,
        )

    def _parse_ultralytics_result(
        self,
        result: Any,
        image_shape: tuple[int, int],
        inference_ms: float,
    ) -> DetectionResult:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )
            for coords, conf, cls_id, tid in zip(xyxy, confs, clss, track_ids):
                cls_int = int(cls_id)
                detections.append(
                    Detection(
                        bbox=(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                        class_id=cls_int,
                        class_name=self.class_names.get(cls_int, f"class_{cls_int}"),
                        confidence=float(conf),
                        track_id=int(tid) if tid is not None else None,
                    )
                )

        counts = dict(Counter(d.class_name for d in detections).most_common())
        fps = (1000.0 / inference_ms) if inference_ms > 0 else 0.0
        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_ms,
            fps=fps,
            image_shape=image_shape,
            model_name=self.weights_path.name,
            device=self.device,
            class_counts=counts,
            total_count=len(detections),
        )

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        classes: list[int] | None = None,
        max_det: int | None = None,
    ) -> DetectionResult:
        """Run single-frame object detection on a BGR numpy image."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array.")

        conf_val = float(conf if conf is not None else self.config.get("model", "confidence_threshold", default=0.35))
        iou_val = float(iou if iou is not None else self.config.get("model", "iou_threshold", default=0.45))
        imgsz_val = int(imgsz if imgsz is not None else self.config.get("model", "imgsz", default=480))
        max_det_val = int(max_det if max_det is not None else self.config.get("model", "max_detections", default=100))

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_val,
            iou=iou_val,
            imgsz=imgsz_val,
            device=self.device,
            classes=classes,
            max_det=max_det_val,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_ultralytics_result(results[0], (image.shape[0], image.shape[1]), elapsed_ms)

    def track(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        tracker_type: str = "bytetrack",
        persist: bool = True,
        classes: list[int] | None = None,
    ) -> DetectionResult:
        """Run detection + multi-object tracking (ByteTrack or BoT-SORT) on a video/webcam frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty NumPy array.")

        conf_val = float(conf if conf is not None else self.config.get("model", "confidence_threshold", default=0.35))
        iou_val = float(iou if iou is not None else self.config.get("model", "iou_threshold", default=0.45))
        imgsz_val = int(imgsz if imgsz is not None else self.config.get("model", "imgsz", default=480))
        tracker_cfg = "botsort.yaml" if tracker_type.lower().strip() == "botsort" else "bytetrack.yaml"

        t0 = time.perf_counter()
        results = self.model.track(
            source=frame,
            conf=conf_val,
            iou=iou_val,
            imgsz=imgsz_val,
            device=self.device,
            tracker=tracker_cfg,
            persist=persist,
            classes=classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_ultralytics_result(results[0], (frame.shape[0], frame.shape[1]), elapsed_ms)

    def reset_tracker(self) -> None:
        """Reset Ultralytics internal predictor/tracker state cleanly."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            predictor.trackers = None
        if hasattr(self.model, "predictor"):
            self.model.predictor = None

    def close(self) -> None:
        """Release model references to reclaim RAM."""
        self.model = None
        gc.collect()


class OpenVocabularyDetector:
    """Optional YOLO-World open-vocabulary detector for arbitrary text-prompt classes.

    Lazy-loaded only when explicitly enabled so core DetectObject remains lightweight.
    Low-confidence predictions are simply filtered by threshold and are never
    mislabelled as 'unknown'.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "auto",
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or load_config()
        self.device = detect_compute_device(device)["selected"]
        default_ov = self.config.resolve_path(
            self.config.get("model", "open_vocab_weights", default="models/weights/yolov8s-worldv2.pt")
        )
        self.weights_path = Path(weights_path) if weights_path else default_ov
        self.model: Any = None
        self.prompts: list[str] = []

    def is_available(self) -> tuple[bool, str]:
        try:
            import ultralytics  # noqa: F401
            return True, "YOLO-World supported via Ultralytics"
        except Exception as exc:
            return False, f"Open-vocabulary backend unavailable: {exc}"

    def load(self, prompts: list[str]) -> None:
        from ultralytics import YOLOWorld

        cleaned = [p.strip() for p in prompts if p and p.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty text prompt is required for open-vocabulary detection.")

        if self.model is None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Loading optional YOLO-World model: %s on %s", self.weights_path, self.device)
            self.model = YOLOWorld(str(self.weights_path))
            if not self.weights_path.exists():
                cwd_copy = Path.cwd() / self.weights_path.name
                if cwd_copy.exists():
                    cwd_copy.replace(self.weights_path)
                    self.model = YOLOWorld(str(self.weights_path))

        if cleaned != self.prompts:
            self.model.set_classes(cleaned)
            self.prompts = cleaned

    def detect(
        self,
        image: np.ndarray,
        prompts: list[str],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 480,
    ) -> DetectionResult:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array.")
        self.load(prompts)
        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0]
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for coords, c, cls_id in zip(xyxy, confs, clss):
                idx = int(cls_id)
                label = self.prompts[idx] if 0 <= idx < len(self.prompts) else f"prompt_{idx}"
                detections.append(
                    Detection(
                        bbox=(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                        class_id=idx,
                        class_name=label,
                        confidence=float(c),
                        track_id=None,
                    )
                )

        counts = dict(Counter(d.class_name for d in detections).most_common())
        fps = (1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0
        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            fps=fps,
            image_shape=(image.shape[0], image.shape[1]),
            model_name=self.weights_path.name,
            device=self.device,
            class_counts=counts,
            total_count=len(detections),
        )
