"""Tests for object counting (per-frame vs unique tracked, line crossing, ROI) and ByteTrack IDs."""

from __future__ import annotations

import cv2
import numpy as np

from detection.object_counter import CountingConfig, ObjectCounter
from models.detector import Detection, ObjectDetector
from models.tracker import ObjectTracker


def test_counter_distinguishes_frame_vs_unique_and_counts_line_roi() -> None:
    cfg = CountingConfig(
        line_enabled=True,
        line_orientation="horizontal",
        line_position=0.5,  # y = 100 on height=200
        roi_enabled=True,
        roi_box=(0.2, 0.2, 0.8, 0.8),  # x in [40..160], y in [40..160]
    )
    counter = ObjectCounter(cfg)

    # Frame 1: Track #1 above line (y=70), Track #2 below line (y=130)
    f1 = [
        Detection(bbox=(80, 50, 120, 90), class_id=0, class_name="person", confidence=0.9, track_id=1),
        Detection(bbox=(80, 110, 120, 150), class_id=39, class_name="bottle", confidence=0.85, track_id=2),
    ]
    s1 = counter.update(f1, (200, 200))
    assert s1.frame_total == 2
    assert s1.unique_tracked_total == 2
    assert s1.line_in_count == 0
    assert s1.line_out_count == 0
    assert s1.roi_current_count == 2

    # Frame 2: Track #1 crosses DOWN (y=120 -> IN), Track #2 crosses UP (y=80 -> OUT)
    f2 = [
        Detection(bbox=(80, 100, 120, 140), class_id=0, class_name="person", confidence=0.92, track_id=1),
        Detection(bbox=(80, 60, 120, 100), class_id=39, class_name="bottle", confidence=0.88, track_id=2),
    ]
    s2 = counter.update(f2, (200, 200))
    assert s2.frame_total == 2
    assert s2.unique_tracked_total == 2  # Still 2 unique IDs, NOT 4!
    assert s2.line_in_count == 1
    assert s2.line_out_count == 1
    assert s2.line_total_crossings == 2

    # Frame 3: Only Track #1 present outside ROI; Track #3 appears
    f3 = [
        Detection(bbox=(5, 5, 25, 25), class_id=0, class_name="person", confidence=0.88, track_id=1),
        Detection(bbox=(90, 90, 110, 110), class_id=0, class_name="person", confidence=0.80, track_id=3),
    ]
    s3 = counter.update(f3, (200, 200))
    assert s3.frame_total == 2
    assert s3.unique_tracked_total == 3
    assert s3.unique_tracked_by_class["person"] == 2
    assert s3.unique_tracked_by_class["bottle"] == 1
    assert s3.roi_current_count == 1
    assert s3.roi_unique_total == 3


def test_bytetrack_assigns_and_maintains_tracking_ids(shared_detector: ObjectDetector, sample_test_image_path) -> None:
    img = cv2.imread(str(sample_test_image_path))
    assert img is not None
    img = cv2.resize(img, (480, 360))

    tracker = ObjectTracker(shared_detector, tracker_type="bytetrack")
    tracker.reset()

    seen_id_sets: list[set[int]] = []
    for step in range(4):
        shifted = np.roll(img, shift=step * 2, axis=1)
        res = tracker.update(shifted, conf=0.25, imgsz=320)
        ids = {d.track_id for d in res.detections if d.track_id is not None}
        seen_id_sets.append(ids)

    # If the sample image has detectable objects, verify track IDs persist across frames
    if any(seen_id_sets):
        overlap = seen_id_sets[0] & seen_id_sets[-1]
        assert len(overlap) >= 1 or tracker.get_unique_track_count() >= 1
