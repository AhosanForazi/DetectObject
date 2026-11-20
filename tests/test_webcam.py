"""Tests for browser LiveInferenceServer API and OpenCV WebcamDetector error/release behavior."""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path

import cv2

from detection.live_api import LiveInferenceServer
from detection.webcam_detector import WebcamDetector, enumerate_cameras
from models.detector import ObjectDetector


def test_live_inference_api_detect_reset_and_screenshot(
    shared_detector: ObjectDetector,
    sample_test_image_path: Path,
    tmp_path: Path,
) -> None:
    server = LiveInferenceServer(detector=shared_detector)
    server.screenshot_dir = tmp_path
    base_url = server.start()
    try:
        # 1. Health endpoint
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert health["status"] == "ok"

        # 2. Detect endpoint with real JPEG frame
        img = cv2.imread(str(sample_test_image_path))
        assert img is not None
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        req = urllib.request.Request(
            f"{base_url}/detect?conf=0.25&imgsz=320&track=1&tracker=bytetrack",
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert payload["width"] == img.shape[1]
            assert payload["inference_ms"] > 0
            assert "counts" in payload
            assert isinstance(payload["detections"], list)

        # 3. Reset endpoint
        reset_req = urllib.request.Request(f"{base_url}/reset", data=b"", method="POST")
        with urllib.request.urlopen(reset_req, timeout=5) as r:
            assert r.status == 200

        # 4. Save screenshot endpoint
        ok_png, png_buf = cv2.imencode(".png", img)
        assert ok_png
        b64_url = "data:image/png;base64," + base64.b64encode(png_buf.tobytes()).decode("ascii")
        shot_body = json.dumps({"image_data_url": b64_url, "counts": payload["counts"], "detections": []}).encode("utf-8")
        shot_req = urllib.request.Request(
            f"{base_url}/save_screenshot",
            data=shot_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(shot_req, timeout=5) as r:
            saved = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert Path(saved["path"]).exists()
    finally:
        server.stop()


def test_webcam_unavailable_index_returns_clean_error(shared_detector: ObjectDetector) -> None:
    webcam = WebcamDetector(detector=shared_detector)
    try:
        ok, msg = webcam.start(camera_index=99)
        assert ok is False
        assert "Unable to open camera" in msg or "failed to deliver" in msg
        assert webcam.is_running is False
    finally:
        webcam.stop()


def test_webcam_live_init_pause_screenshot_and_release(shared_detector: ObjectDetector, tmp_path: Path) -> None:
    """Tests real OpenCV desktop webcam hardware when an unlocked camera is available."""
    cams = enumerate_cameras(max_index=2)
    if not cams:
        return

    webcam = WebcamDetector(detector=shared_detector)
    webcam.screenshot_dir = tmp_path
    try:
        ok, msg = webcam.start(camera_index=cams[0]["index"])
        if not ok:
            # Skip only if another running desktop process currently holds an exclusive camera lock
            return
        assert webcam.is_running is True

        deadline = time.time() + 4.0
        packet = webcam.get_latest_packet()
        while time.time() < deadline and packet.frame_number == 0:
            time.sleep(0.1)
            packet = webcam.get_latest_packet()

        assert packet.frame_number >= 1
        assert packet.annotated_bgr is not None

        webcam.pause()
        assert webcam.is_paused is True
        webcam.resume()
        assert webcam.is_paused is False

        img_path, meta_path = webcam.capture_screenshot(prefix="pytest_shot")
        assert img_path.exists()
        assert meta_path.exists()
    finally:
        webcam.stop()
        assert webcam.is_running is False
        assert webcam._cap is None
