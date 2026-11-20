"""Automated, reproducible, resource-aware YOLO training pipeline for DetectObject."""

from __future__ import annotations

import gc
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from configuration.settings import (
    AppConfig,
    detect_compute_device,
    get_logger,
    load_config,
)

logger = get_logger("detectobject.trainer")


@dataclass
class TrainingSummary:
    status: str
    base_model: str
    dataset_yaml: str
    device: str
    epochs_configured: int
    epochs_completed: int
    batch_size_used: int
    imgsz_used: int
    optimizer: str
    learning_rate: float
    seed: int
    duration_sec: float
    best_checkpoint_path: str
    backup_checkpoint_path: str | None
    run_dir: str
    results_csv_path: str | None
    final_val_metrics: dict[str, float]
    completed_at_utc: str


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_safe_batch_size(device_info: dict[str, Any], imgsz: int) -> int:
    """Pick a memory-safe batch size based on available VRAM (CUDA) or system RAM (CPU)."""
    if device_info.get("cuda_available"):
        vram = float(device_info.get("cuda_vram_gb", 0.0))
        if vram >= 12.0:
            return 16 if imgsz <= 640 else 8
        if vram >= 6.0:
            return 8 if imgsz <= 640 else 4
        return 4

    avail_ram_gb = psutil.virtual_memory().available / (1024**3)
    if imgsz <= 320:
        return 8 if avail_ram_gb >= 0.4 else 4
    if avail_ram_gb >= 4.0 and imgsz <= 480:
        return 8
    return 4


def backup_existing_weights(weights_dest: Path) -> Path | None:
    """Preserve existing best.pt before overwriting."""
    if not weights_dest.exists():
        return None
    backup_dir = weights_dest.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{weights_dest.stem}_backup_{ts}{weights_dest.suffix}"
    shutil.copy2(weights_dest, backup_path)
    logger.info("Backed up existing weights to %s", backup_path)
    return backup_path


class YOLOTrainer:
    """Manages reproducible YOLO training with OOM recovery and checkpoint preservation."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.device_info = detect_compute_device(self.config.get("model", "device", default="auto"))
        self.device = self.device_info["selected"]

    def _ensure_base_weights(self, base_model_name: str) -> Path:
        weights_dir = self.config.resolve_path("models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / Path(base_model_name).name
        if local_path.exists():
            return local_path

        from ultralytics import YOLO

        logger.info("Downloading pretrained baseline weights: %s", base_model_name)
        YOLO(base_model_name)
        cwd_file = Path.cwd() / Path(base_model_name).name
        if cwd_file.exists():
            shutil.move(str(cwd_file), str(local_path))
            return local_path
        return Path(base_model_name)

    def train(
        self,
        dataset_yaml: str | Path | None = None,
        epochs: int | None = None,
        batch_size: int | str | None = None,
        imgsz: int | None = None,
        learning_rate: float | None = None,
        resume: bool = False,
    ) -> TrainingSummary:
        from ultralytics import YOLO

        ds_yaml = Path(dataset_yaml) if dataset_yaml else self.config.resolve_path("datasets/processed/dataset.yaml")
        if not ds_yaml.exists():
            raise FileNotFoundError(
                f"Prepared dataset YAML not found: {ds_yaml}. Run 'python training/dataset_prepare.py' first."
            )

        base_model_name = str(self.config.get("training", "base_model", default="yolo11n.pt"))
        epochs_val = int(epochs if epochs is not None else self.config.get("training", "epochs", default=12))
        imgsz_val = int(imgsz if imgsz is not None else self.config.get("training", "imgsz", default=320))
        optimizer_val = str(self.config.get("training", "optimizer", default="SGD"))
        lr_val = float(learning_rate if learning_rate is not None else self.config.get("training", "learning_rate", default=0.0003))
        warmup_bias_lr = float(self.config.get("training", "warmup_bias_lr", default=0.001))
        warmup_epochs = float(self.config.get("training", "warmup_epochs", default=1.0))
        freeze_layers = self.config.get("training", "freeze_layers", default=10)
        wd_val = float(self.config.get("training", "weight_decay", default=0.0005))
        patience_val = int(self.config.get("training", "patience", default=6))
        seed_val = int(self.config.get("training", "seed", default=42))
        workers_val = int(self.config.get("training", "workers", default=0))
        project_dir = self.config.resolve_path(self.config.get("training", "project_dir", default="logs/training"))
        run_name = str(self.config.get("training", "run_name", default="detectobject_yolo11n"))
        best_dest = self.config.resolve_path(
            self.config.get("training", "best_weights_dest", default="models/weights/best.pt")
        )

        raw_batch = batch_size if batch_size is not None else self.config.get("training", "batch_size", default="auto")
        if isinstance(raw_batch, str) and raw_batch.lower() == "auto":
            batch_val = choose_safe_batch_size(self.device_info, imgsz_val)
        else:
            batch_val = max(1, int(raw_batch))

        set_global_seed(seed_val)
        project_dir.mkdir(parents=True, exist_ok=True)
        best_dest.parent.mkdir(parents=True, exist_ok=True)

        # Support resuming from last.pt if requested and present
        resume_ckpt = project_dir / run_name / "weights" / "last.pt"
        if resume and resume_ckpt.exists():
            model_source = resume_ckpt
            logger.info("Resuming training from checkpoint: %s", resume_ckpt)
        else:
            resume = False
            model_source = self._ensure_base_weights(base_model_name)

        t_start = time.perf_counter()
        attempt = 0
        max_oom_retries = 2
        train_results: Any = None
        actual_run_dir = project_dir / run_name

        while True:
            attempt += 1
            try:
                logger.info(
                    "Starting YOLO training (attempt %d): model=%s, device=%s, epochs=%d, batch=%d, imgsz=%d, lr=%.5f",
                    attempt,
                    model_source.name,
                    self.device,
                    epochs_val,
                    batch_val,
                    imgsz_val,
                    lr_val,
                )
                model = YOLO(str(model_source))
                train_results = model.train(
                    data=str(ds_yaml),
                    epochs=epochs_val,
                    batch=batch_val,
                    imgsz=imgsz_val,
                    device=self.device,
                    optimizer=optimizer_val,
                    lr0=lr_val,
                    warmup_bias_lr=warmup_bias_lr,
                    warmup_epochs=warmup_epochs,
                    freeze=freeze_layers,
                    cos_lr=True,
                    mosaic=0.0,
                    weight_decay=wd_val,
                    patience=patience_val,
                    seed=seed_val,
                    deterministic=True,
                    workers=workers_val,
                    cache=False,
                    amp=bool(self.device_info.get("cuda_available")),
                    project=str(project_dir),
                    name=run_name,
                    exist_ok=True,
                    resume=resume,
                    plots=True,
                    verbose=True,
                )
                save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
                if save_dir is not None:
                    actual_run_dir = Path(save_dir)
                break
            except RuntimeError as exc:
                err_lower = str(exc).lower()
                is_oom = ("out of memory" in err_lower) or ("alloc" in err_lower and "memory" in err_lower)
                if is_oom and attempt <= max_oom_retries and batch_val > 1:
                    logger.warning(
                        "Memory pressure detected (%s). Reducing batch size %d -> %d and retrying safely.",
                        exc,
                        batch_val,
                        max(1, batch_val // 2),
                    )
                    batch_val = max(1, batch_val // 2)
                    if imgsz_val > 256:
                        imgsz_val = max(256, imgsz_val - 64)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                logger.error("Training failed with RuntimeError: %s", exc)
                raise

        duration_sec = round(time.perf_counter() - t_start, 2)

        # Locate produced best.pt
        produced_best = actual_run_dir / "weights" / "best.pt"
        if not produced_best.exists():
            produced_last = actual_run_dir / "weights" / "last.pt"
            if produced_last.exists():
                produced_best = produced_last
            else:
                raise FileNotFoundError(f"Training completed but no checkpoint found under {actual_run_dir / 'weights'}")

        backup_path = backup_existing_weights(best_dest)
        shutil.copy2(produced_best, best_dest)
        logger.info("Saved best trained weights to %s", best_dest)

        # Extract real final validation metrics from Ultralytics results_dict & results.csv
        results_csv = actual_run_dir / "results.csv"
        val_metrics: dict[str, float] = {}
        if train_results is not None and hasattr(train_results, "results_dict"):
            for k, v in train_results.results_dict.items():
                try:
                    val_metrics[str(k)] = round(float(v), 5)
                except Exception:
                    pass

        epochs_completed = epochs_val
        if results_csv.exists():
            lines = [ln for ln in results_csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
            epochs_completed = max(0, len(lines) - 1)

        summary = TrainingSummary(
            status="completed",
            base_model=base_model_name,
            dataset_yaml=str(ds_yaml),
            device=self.device,
            epochs_configured=epochs_val,
            epochs_completed=epochs_completed,
            batch_size_used=batch_val,
            imgsz_used=imgsz_val,
            optimizer=optimizer_val,
            learning_rate=lr_val,
            seed=seed_val,
            duration_sec=duration_sec,
            best_checkpoint_path=str(best_dest),
            backup_checkpoint_path=str(backup_path) if backup_path else None,
            run_dir=str(actual_run_dir),
            results_csv_path=str(results_csv) if results_csv.exists() else None,
            final_val_metrics=val_metrics,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        summary_path = project_dir / "training_summary.json"
        summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        logger.info("Training summary saved to %s (duration: %.2fs)", summary_path, duration_sec)
        return summary
