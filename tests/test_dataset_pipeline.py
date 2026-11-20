"""Tests for dataset download error handling, image/label validation, deduplication, and leak-free splits."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from datasets.dataset_loader import DatasetManager


def test_label_validation_and_normalization(tmp_path: Path) -> None:
    mgr = DatasetManager()
    lbl = tmp_path / "sample.txt"
    lbl.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.4 0.4",        # valid
                "999 0.5 0.5 0.2 0.2",      # invalid class id
                "1 0.5 0.5 -0.2 0.3",       # negative width
                "2 0.99 0.5 0.2 0.2",       # slightly overflowing right edge -> clipped cleanly
                "bad line format",          # malformed
                "3 NaN 0.5 0.2 0.2",        # NaN coordinate
            ]
        ),
        encoding="utf-8",
    )
    valid, raw_count, invalid_count = mgr.validate_and_normalize_labels(lbl)
    assert raw_count == 6
    assert invalid_count == 4
    assert len(valid) == 2
    for cls_id, cx, cy, w, h in valid:
        assert 0 <= cls_id < 80
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_corrupted_duplicate_and_leak_free_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "raw_mini"
    img_dir = src / "images"
    lbl_dir = src / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create 6 distinct valid images
    for i in range(6):
        arr = np.full((64, 64, 3), fill_value=(i * 35 + 10) % 255, dtype=np.uint8)
        cv2.circle(arr, (20 + i * 4, 32), 10, (255, 255, 255), -1)
        p = img_dir / f"img_{i}.jpg"
        cv2.imwrite(str(p), arr)
        (lbl_dir / f"img_{i}.txt").write_text(f"{i % 3} 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    # Add 1 exact duplicate image
    shutil.copy2(img_dir / "img_0.jpg", img_dir / "img_0_dup.jpg")
    (lbl_dir / "img_0_dup.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    # Add 1 corrupted image
    (img_dir / "corrupt.jpg").write_bytes(b"not-a-valid-jpeg-stream")
    (lbl_dir / "corrupt.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    mgr = DatasetManager()
    mgr.processed_dir = tmp_path / "processed"
    mgr.manifests_dir = tmp_path / "manifests"
    mgr.processed_dir.mkdir(parents=True, exist_ok=True)
    mgr.manifests_dir.mkdir(parents=True, exist_ok=True)

    summary = mgr.prepare_from_directory(src, archive_sha256="unit-test-sha")
    assert summary.total_raw_images == 8
    assert summary.corrupted_images_removed == 1
    assert summary.duplicate_images_removed == 1
    assert summary.valid_images_kept == 6
    assert summary.train_count + summary.val_count + summary.test_count == 6
    assert summary.leakage_verified_clean is True
    assert Path(summary.dataset_yaml_path).exists()
    assert (mgr.manifests_dir / "dataset_manifest.json").exists()
    assert (mgr.manifests_dir / "split_manifest.json").exists()


def test_failed_download_raises_without_silent_success(tmp_path: Path) -> None:
    mgr = DatasetManager()
    bad_dest = tmp_path / "nonexistent.zip"
    with pytest.raises(RuntimeError):
        mgr.download_archive(
            url="http://127.0.0.1:9/definitely-unreachable.zip",
            dest_path=bad_dest,
            max_retries=1,
            timeout_sec=2,
        )
    assert not bad_dest.exists()
