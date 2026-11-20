"""Shared pytest fixtures for DetectObject test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def shared_detector():
    from models.detector import ObjectDetector

    det = ObjectDetector(device="auto")
    yield det
    det.close()


@pytest.fixture(scope="session")
def sample_test_image_path(project_root: Path) -> Path:
    test_dir = project_root / "datasets" / "processed" / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))
    if images:
        return images[0]
    # Fallback synthetic image if dataset was not prepared yet
    fallback = project_root / "outputs" / "images" / "_pytest_sample.png"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((320, 480, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (220, 260), (200, 180, 160), -1)
    cv2.imwrite(str(fallback), img)
    return fallback


@pytest.fixture()
def synthetic_video_path(tmp_path: Path, sample_test_image_path: Path) -> Path:
    """Create a short 12-frame moving-object MP4 using a real test image shifted slightly across frames."""
    base = cv2.imread(str(sample_test_image_path))
    assert base is not None
    base = cv2.resize(base, (320, 240))
    video_path = tmp_path / "sample_motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 12.0, (320, 240))
    assert writer.isOpened()
    try:
        for i in range(12):
            shifted = np.roll(base, shift=i * 3, axis=1)
            writer.write(shifted)
    finally:
        writer.release()
    return video_path
