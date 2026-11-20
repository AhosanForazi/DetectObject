"""Video object detection, tracking, counting, and browser-compatible MP4 export."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from configuration.settings import AppConfig, get_logger, load_config
from detection.object_counter import CountingConfig, CountSnapshot, ObjectCounter
from detection.visualizer import DetectionVisualizer
from models.detector import ObjectDetector
from models.tracker import ObjectTracker

logger = get_logger("detectobject.video")

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoDetectionOutput:
    """Summary and artifact paths for a processed video."""

    input_path: Path
    output_path: Path
    width: int
    height: int
    source_fps: float
    total_frames: int
    processed_frames: int
    avg_inference_ms: float
    avg_processing_fps: float
    wall_time_sec: float
    peak_frame_objects: int
    total_frame_detections: int
    unique_tracked_total: int
    unique_tracked_by_class: dict[str, int]
    cumulative_class_detections: dict[str, int]
    final_count_snapshot: CountSnapshot
    tracking_enabled: bool
    tracker_type: str
    codec_used: str
    frame_timeline: list[dict[str, Any]] = field(default_factory=list)


class _BrowserVideoWriter:
    """Writes H.264 MP4 via PyAV for universal browser playback, falling back to OpenCV mp4v."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        self.output_path = output_path
        # H.264 yuv420p requires even dimensions
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1
        self.fps = max(1.0, float(fps if fps and fps > 0 else 24.0))
        self.codec_used = "unknown"
        self._av_container: Any = None
        self._av_stream: Any = None
        self._cv_writer: cv2.VideoWriter | None = None
        self._init_writer()

    def _init_writer(self) -> None:
        try:
            import av
            from fractions import Fraction

            self._av_container = av.open(str(self.output_path), mode="w")
            fps_int = max(1, int(round(self.fps)))
            self._av_stream = self._av_container.add_stream("libx264", rate=Fraction(fps_int, 1))
            self._av_stream.width = self.width
            self._av_stream.height = self.height
            self._av_stream.pix_fmt = "yuv420p"
            self._av_stream.options = {"preset": "veryfast", "crf": "23"}
            self.codec_used = "libx264 (PyAV)"
            return
        except Exception as exc:
            logger.warning("PyAV libx264 writer unavailable (%s); falling back to OpenCV mp4v.", exc)
            if self._av_container is not None:
                try:
                    self._av_container.close()
                except Exception:
                    pass
                self._av_container = None
                self._av_stream = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not self._cv_writer.isOpened():
            raise RuntimeError(f"Unable to open video writer for {self.output_path}")
        self.codec_used = "mp4v (OpenCV)"

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        if self._av_container is not None and self._av_stream is not None:
            import av

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in self._av_stream.encode(av_frame):
                self._av_container.mux(packet)
        elif self._cv_writer is not None:
            self._cv_writer.write(frame_bgr)

    def close(self) -> None:
        if self._av_container is not None and self._av_stream is not None:
            try:
                for packet in self._av_stream.encode():
                    self._av_container.mux(packet)
            finally:
                self._av_container.close()
                self._av_container = None
                self._av_stream = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None


class VideoDetector:
    """Processes uploaded videos frame-by-frame with tracking, counting, and MP4 export."""

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
            self.config.get("video", "output_dir", default="outputs/videos")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_video(video_path: str | Path) -> dict[str, Any]:
        """Verify video readability and return metadata."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            raise ValueError(
                f"Unsupported video format '{path.suffix}'. Allowed: {sorted(SUPPORTED_VIDEO_EXTS)}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Video file is empty: {path}")

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file '{path.name}'. File may be corrupted.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, sample = cap.read()
            if not ret or sample is None or width <= 0 or height <= 0:
                raise ValueError(f"Video file '{path.name}' contains no readable frames.")
            return {
                "width": width,
                "height": height,
                "fps": fps if fps > 0 else 24.0,
                "frame_count": max(frame_count, 1),
            }
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        frame_skip: int = 1,
        max_frames: int | None = None,
        tracking_enabled: bool = True,
        tracker_type: str = "bytetrack",
        show_trails: bool = True,
        counting_config: CountingConfig | None = None,
        classes: list[int] | None = None,
        progress_callback: Callable[[int, int, float, CountSnapshot], None] | None = None,
    ) -> VideoDetectionOutput:
        """Process a video file with detection, tracking, counting, and progress callbacks."""
        in_path = Path(video_path)
        meta = self.validate_video(in_path)
        frame_skip = max(1, int(frame_skip))

        max_dim = int(self.config.get("video", "max_dimension", default=1280))
        src_w, src_h = meta["width"], meta["height"]
        scale = min(1.0, max_dim / max(src_w, src_h))
        out_w = max(2, int(src_w * scale) // 2 * 2)
        out_h = max(2, int(src_h * scale) // 2 * 2)

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"{in_path.stem}_detected_{ts}.mp4"
        else:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        counter = ObjectCounter(counting_config or CountingConfig())
        tracker = ObjectTracker(self.detector, tracker_type=tracker_type) if tracking_enabled else None
        if tracker is not None:
            tracker.reset()

        cap = cv2.VideoCapture(str(in_path))
        writer = _BrowserVideoWriter(out_path, out_w, out_h, meta["fps"])

        total_frames_est = meta["frame_count"]
        if max_frames is not None:
            total_frames_est = min(total_frames_est, int(max_frames))

        t_start = time.perf_counter()
        frame_idx = 0
        processed_frames = 0
        inference_times: list[float] = []
        peak_objects = 0
        total_frame_detections = 0
        cumulative_classes: Counter[str] = Counter()
        timeline: list[dict[str, Any]] = []
        last_detections = []
        last_trails = None
        last_snapshot = CountSnapshot()
        last_fps = 0.0

        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h))

                should_infer = ((frame_idx - 1) % frame_skip == 0)
                if should_infer:
                    if tracker is not None:
                        res = tracker.update(
                            frame=frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=classes,
                        )
                        last_trails = tracker.get_active_trails() if show_trails else None
                    else:
                        res = self.detector.detect(
                            image=frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=classes,
                        )
                        last_trails = None

                    processed_frames += 1
                    inference_times.append(res.inference_time_ms)
                    last_detections = res.detections
                    last_fps = res.fps
                    last_snapshot = counter.update(res.detections, (out_h, out_w))

                    peak_objects = max(peak_objects, res.total_count)
                    total_frame_detections += res.total_count
                    for cls_name, cnt in res.class_counts.items():
                        cumulative_classes[cls_name] += cnt

                    timeline.append(
                        {
                            "frame": frame_idx,
                            "frame_objects": last_snapshot.frame_total,
                            "unique_tracked": last_snapshot.unique_tracked_total,
                            "fps": round(last_fps, 2),
                        }
                    )

                annotated = self.visualizer.draw(
                    frame_bgr=frame,
                    detections=last_detections,
                    trails=last_trails,
                    counting_config=counting_config,
                    count_snapshot=last_snapshot,
                    fps=last_fps,
                    show_hud=True,
                    show_confidence=True,
                    show_track_ids=tracking_enabled,
                )
                writer.write(annotated)

                if progress_callback is not None:
                    progress_callback(frame_idx, max(total_frames_est, frame_idx), last_fps, last_snapshot)
        finally:
            cap.release()
            writer.close()

        wall_sec = max(1e-4, time.perf_counter() - t_start)
        avg_ms = float(np.mean(inference_times)) if inference_times else 0.0
        avg_fps = (processed_frames / wall_sec) if processed_frames > 0 else 0.0

        logger.info(
            "Processed video %s -> %s (%d/%d frames, avg %.1f FPS, %d unique tracked)",
            in_path.name,
            out_path.name,
            processed_frames,
            frame_idx,
            avg_fps,
            last_snapshot.unique_tracked_total,
        )

        return VideoDetectionOutput(
            input_path=in_path,
            output_path=out_path,
            width=out_w,
            height=out_h,
            source_fps=meta["fps"],
            total_frames=frame_idx,
            processed_frames=processed_frames,
            avg_inference_ms=round(avg_ms, 2),
            avg_processing_fps=round(avg_fps, 2),
            wall_time_sec=round(wall_sec, 2),
            peak_frame_objects=peak_objects,
            total_frame_detections=total_frame_detections,
            unique_tracked_total=last_snapshot.unique_tracked_total,
            unique_tracked_by_class=last_snapshot.unique_tracked_by_class,
            cumulative_class_detections=dict(cumulative_classes.most_common()),
            final_count_snapshot=last_snapshot,
            tracking_enabled=tracking_enabled,
            tracker_type=tracker_type if tracking_enabled else "none",
            codec_used=writer.codec_used,
            frame_timeline=timeline,
        )
