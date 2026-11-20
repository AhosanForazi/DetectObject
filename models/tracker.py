"""Multi-object tracking state manager built on ByteTrack / BoT-SORT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from models.detector import Detection, DetectionResult, ObjectDetector


@dataclass
class TrackInfo:
    """Persistent state for a single tracked object ID across frames."""

    track_id: int
    class_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    hits: int = 1
    max_confidence: float = 0.0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    history: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))


class ObjectTracker:
    """Wraps ObjectDetector tracking with trajectory history and occlusion-tolerant track state."""

    def __init__(
        self,
        detector: ObjectDetector,
        tracker_type: str = "bytetrack",
        history_length: int = 30,
        max_age_frames: int = 30,
    ) -> None:
        self.detector = detector
        self.tracker_type = tracker_type.lower().strip()
        if self.tracker_type not in {"bytetrack", "botsort"}:
            self.tracker_type = "bytetrack"
        self.history_length = max(2, int(history_length))
        self.max_age_frames = max(1, int(max_age_frames))
        self.frame_index = 0
        self.tracks: dict[int, TrackInfo] = {}
        self._next_fallback_id = 100000

    def set_tracker_type(self, tracker_type: str) -> None:
        cleaned = tracker_type.lower().strip()
        if cleaned in {"bytetrack", "botsort"} and cleaned != self.tracker_type:
            self.tracker_type = cleaned
            self.reset()

    def reset(self) -> None:
        """Clear all active tracks and reset underlying ByteTrack/BoT-SORT state."""
        self.frame_index = 0
        self.tracks.clear()
        self._next_fallback_id = 100000
        self.detector.reset_tracker()

    def update(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        classes: list[int] | None = None,
    ) -> DetectionResult:
        """Run detector + tracker on a frame and update persistent track trajectories."""
        self.frame_index += 1
        result = self.detector.track(
            frame=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker_type=self.tracker_type,
            persist=True,
            classes=classes,
        )
        self.ingest_detections(result.detections)
        return result

    def ingest_detections(self, detections: list[Detection]) -> None:
        """Update track histories from an already-computed list of detections."""
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            center = det.center
            if tid not in self.tracks:
                info = TrackInfo(
                    track_id=tid,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    first_seen_frame=self.frame_index,
                    last_seen_frame=self.frame_index,
                    hits=1,
                    max_confidence=det.confidence,
                    last_bbox=det.bbox,
                    history=deque([center], maxlen=self.history_length),
                )
                self.tracks[tid] = info
            else:
                info = self.tracks[tid]
                info.last_seen_frame = self.frame_index
                info.hits += 1
                info.max_confidence = max(info.max_confidence, det.confidence)
                info.last_bbox = det.bbox
                info.class_id = det.class_id
                info.class_name = det.class_name
                info.history.append(center)

    def get_active_trails(self) -> dict[int, list[tuple[float, float]]]:
        """Return trajectory points for tracks seen within max_age_frames."""
        cutoff = self.frame_index - self.max_age_frames
        return {
            tid: list(info.history)
            for tid, info in self.tracks.items()
            if info.last_seen_frame >= cutoff and len(info.history) >= 2
        }

    def get_unique_track_count(self) -> int:
        return len(self.tracks)

    def get_summary(self) -> dict[str, int | str]:
        cutoff = self.frame_index - self.max_age_frames
        active = sum(1 for info in self.tracks.values() if info.last_seen_frame >= cutoff)
        return {
            "tracker_type": self.tracker_type,
            "frames_processed": self.frame_index,
            "active_tracks": active,
            "total_unique_tracks": len(self.tracks),
        }
