"""Central configuration, hardware inspection, and structured logging for DetectObject."""

from __future__ import annotations

import logging
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configuration" / "config.yaml"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "configuration" / "classes.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class AppConfig:
    """Wrapper around the YAML configuration dictionary with path helpers."""

    raw: dict[str, Any] = field(default_factory=dict)
    config_path: Path = DEFAULT_CONFIG_PATH

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, value: Any, *keys: str) -> None:
        if not keys:
            return
        node = self.raw
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def resolve_path(self, relative_or_abs: str | Path) -> Path:
        path = Path(relative_or_abs)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML, creating defaults if needed."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return AppConfig(raw=data, config_path=path)


def save_config(config: AppConfig, config_path: str | Path | None = None) -> Path:
    """Persist configuration back to YAML."""
    path = Path(config_path) if config_path else config.config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.raw, fh, sort_keys=False, allow_unicode=True)
    return path


def load_classes(classes_path: str | Path | None = None) -> dict[int, str]:
    """Load canonical class ID -> label mapping."""
    path = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def detect_compute_device(preferred: str = "auto") -> dict[str, Any]:
    """Automatically detect best available compute device (CUDA, MPS, or CPU)."""
    pref = (preferred or "auto").lower().strip()
    cuda_available = False
    cuda_device_name = None
    cuda_vram_gb = 0.0
    mps_available = False

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            cuda_vram_gb = round(props.total_memory / (1024**3), 2)
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
    except Exception:
        pass

    if pref == "cuda":
        selected = "cuda:0" if cuda_available else "cpu"
        fallback_used = not cuda_available
    elif pref == "mps":
        selected = "mps" if mps_available else "cpu"
        fallback_used = not mps_available
    elif pref == "cpu":
        selected = "cpu"
        fallback_used = False
    else:
        if cuda_available:
            selected = "cuda:0"
        elif mps_available:
            selected = "mps"
        else:
            selected = "cpu"
        fallback_used = False

    return {
        "requested": pref,
        "selected": selected,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "cuda_vram_gb": cuda_vram_gb,
        "mps_available": mps_available,
        "fallback_to_cpu": fallback_used,
    }


def get_system_resources() -> dict[str, Any]:
    """Return live OS, CPU, RAM, disk, and device statistics."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(PROJECT_ROOT)
    device_info = detect_compute_device("auto")
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_cores_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_cores_logical": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_used_percent": vm.percent,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "device": device_info,
    }


def get_logger(name: str, log_file: str = "detectobject.log") -> logging.Logger:
    """Create or return a structured console + file logger."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
