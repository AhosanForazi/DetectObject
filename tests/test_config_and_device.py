"""Tests for configuration loading, GPU detection, and CPU fallback."""

from __future__ import annotations

from configuration.settings import (
    detect_compute_device,
    get_system_resources,
    load_classes,
    load_config,
)


def test_config_and_classes_load() -> None:
    cfg = load_config()
    assert cfg.get("project", "name") == "DetectObject"
    classes = load_classes()
    assert len(classes) == 80
    assert classes[0] == "person"


def test_gpu_detection_structure() -> None:
    info = detect_compute_device("auto")
    assert info["selected"] in {"cpu", "cuda:0", "mps"}
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["mps_available"], bool)


def test_cpu_explicit_and_cuda_fallback() -> None:
    cpu_info = detect_compute_device("cpu")
    assert cpu_info["selected"] == "cpu"
    assert cpu_info["fallback_to_cpu"] is False

    cuda_req = detect_compute_device("cuda")
    if not cuda_req["cuda_available"]:
        assert cuda_req["selected"] == "cpu"
        assert cuda_req["fallback_to_cpu"] is True
    else:
        assert cuda_req["selected"] == "cuda:0"


def test_system_resources_snapshot() -> None:
    res = get_system_resources()
    assert res["ram_total_gb"] > 0
    assert res["disk_free_gb"] > 0
    assert res["cpu_cores_logical"] >= 1
