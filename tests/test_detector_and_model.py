"""Tests for model loading, missing weights handling, and CPU inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.detector import ObjectDetector


def test_model_loading_and_metadata(shared_detector: ObjectDetector) -> None:
    assert shared_detector.model is not None
    assert len(shared_detector.class_names) == 80
    assert shared_detector.device in {"cpu", "cuda:0", "mps"}
    assert shared_detector.model_info["parameters"] > 0


def test_missing_model_weights_raises_strict(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist_weights.pt"
    with pytest.raises(FileNotFoundError):
        ObjectDetector(weights_path=missing_path, device="cpu", strict_weights=True)


def test_invalid_empty_array_rejected(shared_detector: ObjectDetector) -> None:
    with pytest.raises(ValueError):
        shared_detector.detect(np.array([]))
