"""Dataset download, integrity validation, cleaning, deduplication, and leak-free splitting."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import requests
import yaml
from PIL import Image

from configuration.settings import (
    AppConfig,
    get_logger,
    load_classes,
    load_config,
)

logger = get_logger("detectobject.dataset")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class DatasetSummary:
    dataset_name: str
    source_url: str
    license_info: str
    archive_sha256: str
    total_raw_images: int
    corrupted_images_removed: int
    duplicate_images_removed: int
    valid_images_kept: int
    total_boxes_raw: int
    invalid_boxes_removed: int
    valid_boxes_kept: int
    train_count: int
    val_count: int
    test_count: int
    leakage_verified_clean: bool
    dataset_yaml_path: str
    prepared_at_utc: str


class DatasetManager:
    """Manages end-to-end dataset acquisition, validation, normalization, and splitting."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.classes = load_classes()
        self.nc = len(self.classes)

        self.dataset_name = str(self.config.get("dataset", "name", default="coco128"))
        self.source_url = str(self.config.get("dataset", "source_url"))
        self.license_info = str(self.config.get("dataset", "license", default="CC BY 4.0"))
        self.raw_dir = self.config.resolve_path(self.config.get("dataset", "raw_dir", default="datasets/raw"))
        self.processed_dir = self.config.resolve_path(
            self.config.get("dataset", "processed_dir", default="datasets/processed")
        )
        self.manifests_dir = self.config.resolve_path(
            self.config.get("dataset", "manifests_dir", default="datasets/manifests")
        )
        self.train_ratio = float(self.config.get("dataset", "train_ratio", default=0.70))
        self.val_ratio = float(self.config.get("dataset", "val_ratio", default=0.15))
        self.test_ratio = float(self.config.get("dataset", "test_ratio", default=0.15))
        self.seed = int(self.config.get("dataset", "seed", default=42))

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def check_disk_space(self, min_free_mb: int = 500) -> bool:
        usage = shutil.disk_usage(self.raw_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_free_mb:
            raise RuntimeError(
                f"Insufficient disk space: {free_mb:.1f} MB free, need at least {min_free_mb} MB."
            )
        logger.info("Disk space check passed: %.1f MB available.", free_mb)
        return True

    def download_archive(
        self,
        url: str | None = None,
        dest_path: Path | None = None,
        max_retries: int = 3,
        timeout_sec: int = 60,
    ) -> tuple[Path, str]:
        """Download dataset archive with retry, size verification, and SHA-256 digest."""
        target_url = url or self.source_url
        archive_path = dest_path or (self.raw_dir / f"{self.dataset_name}.zip")

        if archive_path.exists() and archive_path.stat().st_size > 0:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    if zf.testzip() is None:
                        sha = self.compute_sha256(archive_path)
                        logger.info("Using verified existing archive: %s (SHA-256: %s)", archive_path.name, sha[:16])
                        return archive_path, sha
            logger.warning("Existing archive %s failed integrity check; re-downloading.", archive_path)
            archive_path.unlink(missing_ok=True)

        self.check_disk_space(min_free_mb=300)
        part_path = archive_path.with_suffix(".zip.part")
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Downloading %s (attempt %d/%d)...", target_url, attempt, max_retries)
                with requests.get(target_url, stream=True, timeout=timeout_sec) as response:
                    response.raise_for_status()
                    expected_len = int(response.headers.get("Content-Length", "0"))
                    downloaded = 0
                    with part_path.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                fh.write(chunk)
                                downloaded += len(chunk)

                if expected_len > 0 and downloaded != expected_len:
                    raise IOError(
                        f"Incomplete download: expected {expected_len} bytes, received {downloaded} bytes."
                    )

                if not zipfile.is_zipfile(part_path):
                    raise ValueError("Downloaded file is not a valid ZIP archive.")

                with zipfile.ZipFile(part_path, "r") as zf:
                    bad_member = zf.testzip()
                    if bad_member is not None:
                        raise ValueError(f"Corrupted ZIP member detected: {bad_member}")

                shutil.move(str(part_path), str(archive_path))
                sha = self.compute_sha256(archive_path)
                logger.info(
                    "Download verified: %s (%d bytes, SHA-256: %s)",
                    archive_path.name,
                    archive_path.stat().st_size,
                    sha[:16],
                )
                return archive_path, sha
            except Exception as exc:
                last_error = exc
                logger.error("Download attempt %d failed: %s", attempt, exc)
                part_path.unlink(missing_ok=True)
                if attempt < max_retries:
                    time.sleep(2 * attempt)

        raise RuntimeError(f"Failed to download dataset after {max_retries} attempts: {last_error}")

    def extract_archive(self, archive_path: Path, extract_root: Path | None = None) -> Path:
        """Safely extract ZIP archive while blocking Zip-Slip path traversal."""
        target_dir = extract_root or (self.raw_dir / "extracted")
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_root = target_dir.resolve()

        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                member_path = (resolved_root / member.filename).resolve()
                if not str(member_path).startswith(str(resolved_root)):
                    raise ValueError(f"Unsafe archive path detected: {member.filename}")
            zf.extractall(resolved_root)

        logger.info("Extracted %s to %s", archive_path.name, resolved_root)
        return resolved_root

    @staticmethod
    def validate_image(image_path: Path) -> tuple[bool, tuple[int, int] | None, str | None]:
        """Verify image readability via both PIL and OpenCV."""
        try:
            if not image_path.exists() or image_path.stat().st_size == 0:
                return False, None, "Empty or missing file"
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                width, height = img.size
            if width <= 1 or height <= 1:
                return False, None, f"Invalid dimensions {width}x{height}"
            bgr = cv2.imread(str(image_path))
            if bgr is None or bgr.size == 0:
                return False, None, "OpenCV decode failure"
            return True, (width, height), None
        except Exception as exc:
            return False, None, str(exc)

    def validate_and_normalize_labels(self, label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], int, int]:
        """Validate YOLO label lines, remove invalid boxes, and normalize coordinates into [0, 1].

        Returns:
            (valid_boxes, raw_count, invalid_count)
        """
        if not label_path.exists():
            return [], 0, 0

        valid_boxes: list[tuple[int, float, float, float, float]] = []
        raw_count = 0
        invalid_count = 0

        text = label_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw_count += 1
            parts = stripped.split()
            if len(parts) < 5:
                invalid_count += 1
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                invalid_count += 1
                continue

            if not (0 <= cls_id < self.nc):
                invalid_count += 1
                continue
            if any(math.isnan(v) or math.isinf(v) for v in (cx, cy, w, h)):
                invalid_count += 1
                continue
            if w <= 0.0 or h <= 0.0 or w > 1.05 or h > 1.05:
                invalid_count += 1
                continue
            if cx < -0.05 or cx > 1.05 or cy < -0.05 or cy > 1.05:
                invalid_count += 1
                continue

            # Clip corners into [0.0, 1.0] and recompute clean normalized center/size
            x1 = max(0.0, min(1.0, cx - w / 2.0))
            y1 = max(0.0, min(1.0, cy - h / 2.0))
            x2 = max(0.0, min(1.0, cx + w / 2.0))
            y2 = max(0.0, min(1.0, cy + h / 2.0))
            nw = x2 - x1
            nh = y2 - y1
            if nw < 1e-4 or nh < 1e-4 or (nw * nh) < 1e-6:
                invalid_count += 1
                continue
            ncx = x1 + nw / 2.0
            ncy = y1 + nh / 2.0
            valid_boxes.append((cls_id, round(ncx, 6), round(ncy, 6), round(nw, 6), round(nh, 6)))

        return valid_boxes, raw_count, invalid_count

    def _find_raw_images_and_labels(self, extract_root: Path) -> list[tuple[Path, Path | None]]:
        """Discover image files and match them to YOLO .txt label files."""
        image_files = sorted(
            p for p in extract_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        label_lookup: dict[str, Path] = {}
        for lbl in extract_root.rglob("*.txt"):
            if lbl.name.lower() in {"readme.txt", "license.txt", "classes.txt"}:
                continue
            label_lookup[lbl.stem] = lbl

        pairs: list[tuple[Path, Path | None]] = []
        for img in image_files:
            pairs.append((img, label_lookup.get(img.stem)))
        return pairs

    def _create_leak_free_splits(self, records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Create reproducible train/val/test splits stratified by primary class with strict disjointness."""
        rng = random.Random(self.seed)
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            primary_cls = rec["class_ids"][0] if rec["class_ids"] else -1
            buckets[primary_cls].append(rec)

        ordered_records: list[dict[str, Any]] = []
        for cls_key in sorted(buckets.keys()):
            items = buckets[cls_key]
            rng.shuffle(items)
            ordered_records.extend(items)

        # Round-robin stratified assignment approximates exact split ratios across classes
        total = len(ordered_records)
        if total < 3:
            raise ValueError(f"Need at least 3 valid images to build train/val/test splits; found {total}.")

        n_test = max(1, round(total * self.test_ratio))
        n_val = max(1, round(total * self.val_ratio))
        n_train = total - n_val - n_test
        if n_train < 1:
            n_train = max(1, total - 2)
            n_val = 1
            n_test = total - n_train - n_val

        # Interleave indices to distribute classes evenly across train, val, and test
        test_indices: set[int] = set()
        val_indices: set[int] = set()

        step_test = max(1, total // n_test)
        idx = 0
        while len(test_indices) < n_test and idx < total:
            test_indices.add(idx)
            idx += step_test
        for i in range(total):
            if len(test_indices) >= n_test:
                break
            if i not in test_indices:
                test_indices.add(i)

        remaining = [i for i in range(total) if i not in test_indices]
        step_val = max(1, len(remaining) // n_val)
        pos = 0
        while len(val_indices) < n_val and pos < len(remaining):
            val_indices.add(remaining[pos])
            pos += step_val
        for i in remaining:
            if len(val_indices) >= n_val:
                break
            if i not in val_indices:
                val_indices.add(i)

        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for i, rec in enumerate(ordered_records):
            if i in test_indices:
                splits["test"].append(rec)
            elif i in val_indices:
                splits["val"].append(rec)
            else:
                splits["train"].append(rec)

        return splits

    @staticmethod
    def verify_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> bool:
        """Assert zero overlap in image hashes or filenames between train, val, and test."""
        seen_hashes: dict[str, set[str]] = {}
        seen_names: dict[str, set[str]] = {}
        for split_name, recs in splits.items():
            seen_hashes[split_name] = {r["sha256"] for r in recs}
            seen_names[split_name] = {r["filename"] for r in recs}

        split_keys = list(splits.keys())
        for i in range(len(split_keys)):
            for j in range(i + 1, len(split_keys)):
                a, b = split_keys[i], split_keys[j]
                if seen_hashes[a] & seen_hashes[b]:
                    raise RuntimeError(f"Data leakage detected (duplicate image hash) between {a} and {b}!")
                if seen_names[a] & seen_names[b]:
                    raise RuntimeError(f"Data leakage detected (duplicate filename) between {a} and {b}!")
        return True

    def prepare_from_directory(
        self,
        source_dir: Path,
        archive_sha256: str = "local-directory",
    ) -> DatasetSummary:
        """Validate, deduplicate, normalize, split, and export a YOLO dataset from an extracted directory."""
        pairs = self._find_raw_images_and_labels(source_dir)
        total_raw = len(pairs)
        logger.info("Discovered %d raw image files in %s", total_raw, source_dir)

        corrupted_removed = 0
        duplicate_removed = 0
        total_boxes_raw = 0
        invalid_boxes_removed = 0
        valid_boxes_kept = 0
        seen_hashes: set[str] = set()
        valid_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []

        for img_path, lbl_path in pairs:
            ok, dims, err = self.validate_image(img_path)
            if not ok or dims is None:
                corrupted_removed += 1
                rejected_records.append({"file": str(img_path), "reason": f"corrupted: {err}"})
                continue

            img_hash = self.compute_sha256(img_path)
            if img_hash in seen_hashes:
                duplicate_removed += 1
                rejected_records.append({"file": str(img_path), "reason": "duplicate_hash"})
                continue
            seen_hashes.add(img_hash)

            boxes, raw_b, inv_b = (
                self.validate_and_normalize_labels(lbl_path) if lbl_path else ([], 0, 0)
            )
            total_boxes_raw += raw_b
            invalid_boxes_removed += inv_b
            valid_boxes_kept += len(boxes)

            valid_records.append(
                {
                    "source_image": img_path,
                    "filename": img_path.name,
                    "stem": img_path.stem,
                    "sha256": img_hash,
                    "width": dims[0],
                    "height": dims[1],
                    "boxes": boxes,
                    "class_ids": [b[0] for b in boxes],
                }
            )

        splits = self._create_leak_free_splits(valid_records)
        leakage_clean = self.verify_no_leakage(splits)

        # Clean previous processed images/labels directories safely
        for sub in ("images", "labels"):
            target = self.processed_dir / sub
            if target.exists():
                shutil.rmtree(target)

        split_manifest: dict[str, list[dict[str, Any]]] = {}
        class_distribution: dict[str, dict[str, int]] = {}

        for split_name, recs in splits.items():
            img_out_dir = self.processed_dir / "images" / split_name
            lbl_out_dir = self.processed_dir / "labels" / split_name
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            cls_counter: Counter[str] = Counter()
            manifest_entries: list[dict[str, Any]] = []

            for rec in recs:
                dst_img = img_out_dir / rec["filename"]
                shutil.copy2(rec["source_image"], dst_img)

                dst_lbl = lbl_out_dir / f"{rec['stem']}.txt"
                label_lines = [
                    f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for (cls_id, cx, cy, w, h) in rec["boxes"]
                ]
                dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

                for cls_id in rec["class_ids"]:
                    cls_counter[self.classes.get(cls_id, str(cls_id))] += 1

                manifest_entries.append(
                    {
                        "filename": rec["filename"],
                        "sha256": rec["sha256"],
                        "width": rec["width"],
                        "height": rec["height"],
                        "num_objects": len(rec["boxes"]),
                    }
                )

            split_manifest[split_name] = manifest_entries
            class_distribution[split_name] = dict(cls_counter.most_common())

        # Write YOLO-compatible dataset.yaml
        dataset_yaml_path = self.processed_dir / "dataset.yaml"
        dataset_yaml_payload = {
            "path": str(self.processed_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": self.nc,
            "names": self.classes,
        }
        with dataset_yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dataset_yaml_payload, fh, sort_keys=False, allow_unicode=True)

        summary = DatasetSummary(
            dataset_name=self.dataset_name,
            source_url=self.source_url,
            license_info=self.license_info,
            archive_sha256=archive_sha256,
            total_raw_images=total_raw,
            corrupted_images_removed=corrupted_removed,
            duplicate_images_removed=duplicate_removed,
            valid_images_kept=len(valid_records),
            total_boxes_raw=total_boxes_raw,
            invalid_boxes_removed=invalid_boxes_removed,
            valid_boxes_kept=valid_boxes_kept,
            train_count=len(splits["train"]),
            val_count=len(splits["val"]),
            test_count=len(splits["test"]),
            leakage_verified_clean=leakage_clean,
            dataset_yaml_path=str(dataset_yaml_path),
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        # Save manifests
        manifest_payload = {
            "summary": asdict(summary),
            "class_distribution": class_distribution,
            "rejected_files": rejected_records,
        }
        (self.manifests_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8"
        )
        (self.manifests_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )

        logger.info(
            "Dataset prepared successfully: train=%d, val=%d, test=%d (corrupted=%d, duplicates=%d, invalid_boxes=%d)",
            summary.train_count,
            summary.val_count,
            summary.test_count,
            summary.corrupted_images_removed,
            summary.duplicate_images_removed,
            summary.invalid_boxes_removed,
        )
        return summary

    def run_full_pipeline(self) -> DatasetSummary:
        """Download, verify, extract, clean, split, and generate YAML/manifests."""
        archive_path, sha = self.download_archive()
        extract_root = self.extract_archive(archive_path)
        return self.prepare_from_directory(extract_root, archive_sha256=sha)
