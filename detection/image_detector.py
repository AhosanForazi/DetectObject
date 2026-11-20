"""Image object detection pipeline supporting JPG, JPEG, PNG, and WEBP."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from configuration.settings import AppConfig, get_logger, load_config
from detection.visualizer import DetectionVisualizer
from models.detector import Detection, DetectionResult, ObjectDetector, OpenVocabularyDetector

logger = get_logger("detectobject.image")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageDetectionOutput:
    """Complete output of image object detection."""

    original_rgb: np.ndarray
    annotated_rgb: np.ndarray
    annotated_bgr: np.ndarray
    result: DetectionResult
    saved_path: Path | None = None

    @property
    def detections(self) -> list[Detection]:
        return self.result.detections

    @property
    def total_count(self) -> int:
        return self.result.total_count

    @property
    def class_counts(self) -> dict[str, int]:
        return self.result.class_counts

    @property
    def inference_ms(self) -> float:
        return self.result.inference_time_ms

    @property
    def inference_time_ms(self) -> float:
        return self.result.inference_time_ms

    @property
    def fps(self) -> float:
        return self.result.fps

    def to_png_bytes(self) -> bytes:
        ok, buf = cv2.imencode(".png", self.annotated_bgr)
        if not ok:
            raise RuntimeError("Failed to encode annotated image as PNG.")
        return buf.tobytes()


class ImageDetector:
    """Handles validation, decoding, detection, annotation, and export for images."""

    def __init__(
        self,
        detector: ObjectDetector,
        config: AppConfig | None = None,
        visualizer: DetectionVisualizer | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or load_config()
        self.visualizer = visualizer or DetectionVisualizer()
        self.output_dir = self.config.resolve_path(
            self.config.get("image", "output_dir", default="outputs/images")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_image_from_path(image_path: str | Path) -> np.ndarray:
        """Validate and load an image file from disk into a BGR NumPy array."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{path.suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is empty: {path}")
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                rgb = np.array(img)
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image file '{path.name}': {exc}") from exc

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise ValueError(f"Image has invalid shape: {rgb.shape}")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def load_image_from_bytes(data: bytes, filename: str = "upload.png") -> np.ndarray:
        """Validate and decode raw uploaded bytes into a BGR NumPy array."""
        if not data:
            raise ValueError("Uploaded image payload is empty.")
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                rgb = np.array(img)
        except Exception as exc:
            raise ValueError(f"Could not decode uploaded image '{filename}': {exc}") from exc

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise ValueError(f"Decoded image has invalid dimensions: {rgb.shape}")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def detect_array(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        classes: list[int] | None = None,
        save_output: bool = False,
        output_stem: str | None = None,
        show_hud: bool = False,
    ) -> ImageDetectionOutput:
        """Run object detection on an in-memory BGR image array."""
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Input image array is empty or invalid.")

        result = self.detector.detect(
            image=image_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
        )
        annotated_bgr = self.visualizer.draw(
            frame_bgr=image_bgr,
            detections=result.detections,
            fps=result.fps,
            show_hud=show_hud,
            show_confidence=True,
            show_track_ids=False,
        )
        saved_path: Path | None = None
        if save_output:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = output_stem or f"detected_{ts}"
            saved_path = self.output_dir / f"{stem}.png"
            cv2.imwrite(str(saved_path), annotated_bgr)
            logger.info("Saved annotated image to %s (%d objects)", saved_path.name, result.total_count)

        return ImageDetectionOutput(
            original_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            annotated_rgb=cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            annotated_bgr=annotated_bgr,
            result=result,
            saved_path=saved_path,
        )

    def detect_file(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        classes: list[int] | None = None,
        save_output: bool = True,
    ) -> ImageDetectionOutput:
        bgr = self.load_image_from_path(image_path)
        stem = f"{Path(image_path).stem}_detected"
        return self.detect_array(
            image_bgr=bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            save_output=save_output,
            output_stem=stem,
        )

    def detect_open_vocab(
        self,
        image_bgr: np.ndarray,
        open_vocab_detector: OpenVocabularyDetector,
        prompts: list[str],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 480,
        save_output: bool = False,
    ) -> ImageDetectionOutput:
        """Run optional text-prompted open-vocabulary detection."""
        result = open_vocab_detector.detect(
            image=image_bgr,
            prompts=prompts,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )
        annotated_bgr = self.visualizer.draw(
            frame_bgr=image_bgr,
            detections=result.detections,
            fps=result.fps,
            show_hud=False,
            show_confidence=True,
            show_track_ids=False,
        )
        saved_path: Path | None = None
        if save_output:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            saved_path = self.output_dir / f"openvocab_{ts}.png"
            cv2.imwrite(str(saved_path), annotated_bgr)

        return ImageDetectionOutput(
            original_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            annotated_rgb=cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            annotated_bgr=annotated_bgr,
            result=result,
            saved_path=saved_path,
        )
