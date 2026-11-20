"""Modern high-contrast visualization utilities for bounding boxes, trails, ROI, and HUD overlays."""

from __future__ import annotations

import cv2
import numpy as np

from detection.object_counter import CountingConfig, CountSnapshot
from models.detector import Detection

# High-contrast colorblind-friendly BGR palette
PALETTE_BGR: list[tuple[int, int, int]] = [
    (255, 132, 56),   # vivid blue
    (82, 204, 46),    # emerald green
    (68, 92, 245),    # coral red
    (22, 188, 250),   # amber gold
    (235, 99, 180),   # magenta
    (212, 188, 0),    # cyan teal
    (99, 130, 255),   # warm orange
    (186, 85, 211),   # orchid purple
    (112, 224, 150),  # mint
    (255, 178, 102),  # sky blue
]


class DetectionVisualizer:
    """Renders bounding boxes, labels, track IDs, trails, counting lines, ROI zones, and HUD."""

    @staticmethod
    def color_for_class(class_id: int) -> tuple[int, int, int]:
        return PALETTE_BGR[abs(int(class_id)) % len(PALETTE_BGR)]

    def draw(
        self,
        frame_bgr: np.ndarray,
        detections: list[Detection],
        trails: dict[int, list[tuple[float, float]]] | None = None,
        counting_config: CountingConfig | None = None,
        count_snapshot: CountSnapshot | None = None,
        fps: float | None = None,
        show_hud: bool = True,
        show_confidence: bool = True,
        show_track_ids: bool = True,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]
        scale_factor = max(0.5, min(1.2, w / 960.0))
        box_thickness = max(2, int(round(2.4 * scale_factor)))
        font_scale = max(0.48, 0.58 * scale_factor)
        font_thickness = max(1, int(round(1.6 * scale_factor)))

        # 1. Optional ROI zone
        if counting_config and counting_config.roi_enabled:
            rx1, ry1, rx2, ry2 = counting_config.roi_box
            x1, x2 = int(min(rx1, rx2) * w), int(max(rx1, rx2) * w)
            y1, y2 = int(min(ry1, ry2) * h), int(max(ry1, ry2) * h)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (212, 188, 0), -1)
            cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 220, 40), 2, cv2.LINE_AA)
            roi_text = "ROI ZONE"
            if count_snapshot is not None:
                roi_text = f"ROI: {count_snapshot.roi_current_count} now | {count_snapshot.roi_unique_total} unique"
            self._draw_pill(canvas, roi_text, (x1 + 6, max(22, y1 + 22)), (30, 30, 30), (255, 220, 40), font_scale * 0.9)

        # 2. Optional counting line
        if counting_config and counting_config.line_enabled:
            pos = max(0.05, min(0.95, float(counting_config.line_position)))
            if counting_config.line_orientation.lower() == "vertical":
                lx = int(pos * w)
                cv2.line(canvas, (lx, 0), (lx, h), (0, 215, 255), 2, cv2.LINE_AA)
                badge_pos = (min(w - 180, lx + 8), 32)
            else:
                ly = int(pos * h)
                cv2.line(canvas, (0, ly), (w, ly), (0, 215, 255), 2, cv2.LINE_AA)
                badge_pos = (12, max(26, ly - 10))
            if count_snapshot is not None:
                line_label = (
                    f"LINE  IN: {count_snapshot.line_in_count} | "
                    f"OUT: {count_snapshot.line_out_count} | "
                    f"TOTAL: {count_snapshot.line_total_crossings}"
                )
                self._draw_pill(canvas, line_label, badge_pos, (25, 25, 25), (0, 215, 255), font_scale * 0.9)

        # 3. Optional track trails
        if trails:
            det_color_by_tid = {d.track_id: self.color_for_class(d.class_id) for d in detections if d.track_id is not None}
            for tid, pts in trails.items():
                if len(pts) < 2:
                    continue
                color = det_color_by_tid.get(tid, (200, 200, 200))
                int_pts = np.array([(int(px), int(py)) for px, py in pts], dtype=np.int32)
                cv2.polylines(canvas, [int_pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, tuple(int_pts[-1]), 4, color, -1, cv2.LINE_AA)

        # 4. Bounding boxes + labels
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            color = self.color_for_class(det.class_id)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

            # Subtle corner accents for modern look
            corner_len = min(14, max(6, (x2 - x1) // 5), max(6, (y2 - y1) // 5))
            accent_t = box_thickness + 1
            for cx0, cy0, dx, dy in (
                (x1, y1, corner_len, corner_len),
                (x2, y1, -corner_len, corner_len),
                (x1, y2, corner_len, -corner_len),
                (x2, y2, -corner_len, -corner_len),
            ):
                cv2.line(canvas, (cx0, cy0), (cx0 + dx, cy0), color, accent_t, cv2.LINE_AA)
                cv2.line(canvas, (cx0, cy0), (cx0, cy0 + dy), color, accent_t, cv2.LINE_AA)

            parts: list[str] = []
            if show_track_ids and det.track_id is not None:
                parts.append(f"#{det.track_id}")
            parts.append(det.class_name)
            if show_confidence:
                parts.append(f"{det.confidence * 100:.0f}%")
            label = " ".join(parts)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            pad_x, pad_y = 6, 4
            ly_top = y1 - th - 2 * pad_y
            if ly_top < 0:
                ly_top = y1 + 2
            ly_bot = ly_top + th + 2 * pad_y
            lx_right = min(w - 1, x1 + tw + 2 * pad_x)

            cv2.rectangle(canvas, (x1, ly_top), (lx_right, ly_bot), color, -1)
            luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (15, 15, 15) if luminance > 150 else (250, 250, 250)
            cv2.putText(
                canvas,
                label,
                (x1 + pad_x, ly_bot - pad_y - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

        # 5. Optional top HUD bar
        if show_hud:
            hud_parts: list[str] = []
            if fps is not None:
                hud_parts.append(f"FPS: {fps:.1f}")
            hud_parts.append(f"Objects: {len(detections)}")
            if count_snapshot is not None and count_snapshot.unique_tracked_total > 0:
                hud_parts.append(f"Unique IDs: {count_snapshot.unique_tracked_total}")
            hud_text = "  |  ".join(hud_parts)
            self._draw_pill(canvas, hud_text, (10, 28), (18, 22, 30), (240, 245, 255), font_scale * 0.92)

        return canvas

    @staticmethod
    def _draw_pill(
        canvas: np.ndarray,
        text: str,
        org: tuple[int, int],
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        font_scale: float = 0.55,
    ) -> None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x, y = org
        pad_x, pad_y = 8, 5
        x1 = max(0, x - pad_x)
        y1 = max(0, y - th - pad_y)
        x2 = min(canvas.shape[1] - 1, x + tw + pad_x)
        y2 = min(canvas.shape[0] - 1, y + baseline + pad_y)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
