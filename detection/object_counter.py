"""Accurate object counting: per-frame, class-wise, unique tracked, line-crossing, and ROI."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from models.detector import Detection


@dataclass
class CountingConfig:
    """Configuration for optional line-crossing and region-of-interest (ROI) counting."""

    line_enabled: bool = False
    line_orientation: str = "horizontal"  # "horizontal" or "vertical"
    line_position: float = 0.5  # normalized 0.05..0.95
    roi_enabled: bool = False
    roi_box: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)  # (x1, y1, x2, y2) normalized


@dataclass
class CountSnapshot:
    """Structured snapshot clearly separating per-frame detections from unique tracked objects."""

    frame_total: int = 0
    frame_class_counts: dict[str, int] = field(default_factory=dict)
    unique_tracked_total: int = 0
    unique_tracked_by_class: dict[str, int] = field(default_factory=dict)
    line_in_count: int = 0
    line_out_count: int = 0
    line_total_crossings: int = 0
    line_class_counts: dict[str, int] = field(default_factory=dict)
    roi_current_count: int = 0
    roi_unique_total: int = 0
    roi_class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_total": self.frame_total,
            "frame_class_counts": dict(self.frame_class_counts),
            "unique_tracked_total": self.unique_tracked_total,
            "unique_tracked_by_class": dict(self.unique_tracked_by_class),
            "line_in_count": self.line_in_count,
            "line_out_count": self.line_out_count,
            "line_total_crossings": self.line_total_crossings,
            "line_class_counts": dict(self.line_class_counts),
            "roi_current_count": self.roi_current_count,
            "roi_unique_total": self.roi_unique_total,
            "roi_class_counts": dict(self.roi_class_counts),
        }


class ObjectCounter:
    """Computes per-frame counts, unique tracked IDs, directional line crossings, and ROI counts."""

    def __init__(self, config: CountingConfig | None = None) -> None:
        self.config = config or CountingConfig()
        self.reset()

    def reset(self) -> None:
        self._unique_ids_all: set[int] = set()
        self._unique_ids_by_class: dict[str, set[int]] = defaultdict(set)
        self._last_centroids: dict[int, tuple[float, float]] = {}
        self._crossed_in_ids: set[int] = set()
        self._crossed_out_ids: set[int] = set()
        self._line_in_count: int = 0
        self._line_out_count: int = 0
        self._line_class_counts: Counter[str] = Counter()
        self._roi_unique_ids: set[int] = set()
        self._roi_unique_by_class: dict[str, set[int]] = defaultdict(set)

    def update_config(self, config: CountingConfig) -> None:
        self.config = config

    def _is_inside_roi(self, cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
        rx1, ry1, rx2, ry2 = self.config.roi_box
        x1, x2 = min(rx1, rx2) * frame_w, max(rx1, rx2) * frame_w
        y1, y2 = min(ry1, ry2) * frame_h, max(ry1, ry2) * frame_h
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> CountSnapshot:
        """Process one frame's detections and return updated counts."""
        frame_h, frame_w = frame_shape
        frame_class_counter: Counter[str] = Counter()
        roi_current_counter: Counter[str] = Counter()
        roi_current_total = 0

        line_pos = max(0.05, min(0.95, float(self.config.line_position)))
        is_horizontal = self.config.line_orientation.lower() != "vertical"
        line_coord = (line_pos * frame_h) if is_horizontal else (line_pos * frame_w)

        for det in detections:
            frame_class_counter[det.class_name] += 1
            cx, cy = det.center

            # Region of interest (ROI) counting
            if self.config.roi_enabled and self._is_inside_roi(cx, cy, frame_w, frame_h):
                roi_current_total += 1
                roi_current_counter[det.class_name] += 1
                if det.track_id is not None:
                    self._roi_unique_ids.add(det.track_id)
                    self._roi_unique_by_class[det.class_name].add(det.track_id)

            # Unique tracking ID & directional line crossing
            if det.track_id is not None:
                tid = det.track_id
                self._unique_ids_all.add(tid)
                self._unique_ids_by_class[det.class_name].add(tid)

                if self.config.line_enabled and tid in self._last_centroids:
                    prev_x, prev_y = self._last_centroids[tid]
                    prev_val = prev_y if is_horizontal else prev_x
                    curr_val = cy if is_horizontal else cx

                    if prev_val < line_coord <= curr_val and tid not in self._crossed_in_ids:
                        self._crossed_in_ids.add(tid)
                        self._line_in_count += 1
                        self._line_class_counts[det.class_name] += 1
                    elif prev_val > line_coord >= curr_val and tid not in self._crossed_out_ids:
                        self._crossed_out_ids.add(tid)
                        self._line_out_count += 1
                        self._line_class_counts[det.class_name] += 1

                self._last_centroids[tid] = (cx, cy)

        unique_by_class = {
            cls_name: len(ids)
            for cls_name, ids in sorted(
                self._unique_ids_by_class.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        }

        return CountSnapshot(
            frame_total=len(detections),
            frame_class_counts=dict(frame_class_counter.most_common()),
            unique_tracked_total=len(self._unique_ids_all),
            unique_tracked_by_class=unique_by_class,
            line_in_count=self._line_in_count,
            line_out_count=self._line_out_count,
            line_total_crossings=self._line_in_count + self._line_out_count,
            line_class_counts=dict(self._line_class_counts.most_common()),
            roi_current_count=roi_current_total,
            roi_unique_total=len(self._roi_unique_ids),
            roi_class_counts=dict(roi_current_counter.most_common()),
        )
