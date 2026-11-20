"""DetectObject configuration package."""

from configuration.settings import (
    PROJECT_ROOT,
    AppConfig,
    detect_compute_device,
    get_logger,
    get_system_resources,
    load_classes,
    load_config,
    save_config,
)

__all__ = [
    "PROJECT_ROOT",
    "AppConfig",
    "detect_compute_device",
    "get_logger",
    "get_system_resources",
    "load_classes",
    "load_config",
    "save_config",
]
