"""Tests for image detection, video detection, and invalid media handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from detection.image_detector import ImageDetector
from detection.video_detector import VideoDetector
from models.detector import ObjectDetector


def test_image_detection_pipeline(shared_detector: ObjectDetector, sample_test_image_path: Path, tmp_path: Path) -> None:
    img_det = ImageDetector(detector=shared_detector)
    img_det.output_dir = tmp_path
    out = img_det.detect_file(sample_test_image_path, conf=0.25, imgsz=320, save_output=True)

    assert out.original_rgb.shape == out.annotated_rgb.shape
    assert out.saved_path is not None and out.saved_path.exists()
    assert len(out.to_png_bytes()) > 100
    assert out.result.inference_time_ms > 0
    assert out.result.total_count == len(out.result.detections)
    assert sum(out.result.class_counts.values()) == out.result.total_count


def test_invalid_image_handling(shared_detector: ObjectDetector, tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"corrupted-bytes-not-an-image")
    with pytest.raises(ValueError):
        ImageDetector.load_image_from_path(corrupt)

    wrong_ext = tmp_path / "doc.txt"
    wrong_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        ImageDetector.load_image_from_path(wrong_ext)

    with pytest.raises(ValueError):
        ImageDetector.load_image_from_bytes(b"", "empty.png")


def test_video_detection_and_export(shared_detector: ObjectDetector, synthetic_video_path: Path, tmp_path: Path) -> None:
    vid_det = VideoDetector(detector=shared_detector)
    out_video = tmp_path / "processed_out.mp4"
    progress_calls: list[int] = []

    out = vid_det.process_video(
        video_path=synthetic_video_path,
        output_path=out_video,
        conf=0.25,
        imgsz=320,
        frame_skip=2,
        tracking_enabled=True,
        tracker_type="bytetrack",
        progress_callback=lambda curr, total, fps, snap: progress_calls.append(curr),
    )

    assert out.output_path.exists()
    assert out.output_path.stat().st_size > 1000
    assert out.total_frames == 12
    assert out.processed_frames == 6
    assert len(progress_calls) == 12
    assert out.avg_processing_fps > 0


def test_invalid_video_handling(tmp_path: Path) -> None:
    bad_vid = tmp_path / "corrupt.mp4"
    bad_vid.write_bytes(b"not-a-video")
    with pytest.raises(ValueError):
        VideoDetector.validate_video(bad_vid)

    wrong_vid = tmp_path / "notes.pdf"
    wrong_vid.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        VideoDetector.validate_video(wrong_vid)
