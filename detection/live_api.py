"""Lightweight localhost HTTP inference server for real-time browser webcam detection."""

from __future__ import annotations

import base64
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from configuration.settings import AppConfig, get_logger, load_config
from detection.object_counter import CountingConfig, ObjectCounter
from models.detector import ObjectDetector
from models.tracker import ObjectTracker

logger = get_logger("detectobject.live_api")


class LiveInferenceServer:
    """Localhost-only HTTP inference bridge connecting browser getUserMedia frames to YOLO11 + ByteTrack."""

    def __init__(self, detector: ObjectDetector, config: AppConfig | None = None) -> None:
        self.detector = detector
        self.config = config or load_config()
        self.screenshot_dir = self.config.resolve_path(
            self.config.get("webcam", "screenshot_dir", default="outputs/screenshots")
        )
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._tracker = ObjectTracker(self.detector, tracker_type="bytetrack")
        self._counter = ObjectCounter(CountingConfig())
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.host: str = "0.0.0.0"
        self.port: int = 0

    def set_detector(self, detector: ObjectDetector) -> None:
        with self._lock:
            if self.detector is not detector:
                self.detector = detector
                self._tracker = ObjectTracker(self.detector, tracker_type=self._tracker.tracker_type)
                self._counter.reset()

    def reset_state(self) -> None:
        with self._lock:
            self._tracker.reset()
            self._counter.reset()

    def infer_jpeg(
        self,
        jpeg_bytes: bytes,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
        tracking_enabled: bool = True,
        tracker_type: str = "bytetrack",
        line_enabled: bool = False,
        line_orientation: str = "horizontal",
        line_position: float = 0.5,
        roi_enabled: bool = False,
    ) -> dict[str, Any]:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Invalid JPEG frame payload.")

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        with self._lock:
            self._counter.update_config(
                CountingConfig(
                    line_enabled=line_enabled,
                    line_orientation=line_orientation,
                    line_position=line_position,
                    roi_enabled=roi_enabled,
                    roi_box=(0.15, 0.15, 0.85, 0.85),
                )
            )
            if tracking_enabled:
                self._tracker.set_tracker_type(tracker_type)
                res = self._tracker.update(frame=frame, conf=conf, iou=iou, imgsz=imgsz)
                trails = {
                    str(tid): [[round(px / w, 4), round(py / h, 4)] for px, py in pts[-18:]]
                    for tid, pts in self._tracker.get_active_trails().items()
                }
            else:
                res = self.detector.detect(image=frame, conf=conf, iou=iou, imgsz=imgsz)
                trails = {}

            snap = self._counter.update(res.detections, (h, w))

        total_ms = (time.perf_counter() - t0) * 1000.0
        boxes = [
            {
                "x1": round(d.bbox[0] / w, 4),
                "y1": round(d.bbox[1] / h, 4),
                "x2": round(d.bbox[2] / w, 4),
                "y2": round(d.bbox[3] / h, 4),
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "track_id": d.track_id,
            }
            for d in res.detections
        ]

        return {
            "width": w,
            "height": h,
            "inference_ms": round(total_ms, 1),
            "model": self.detector.weights_path.name,
            "device": self.detector.device,
            "detections": boxes,
            "trails": trails,
            "counts": snap.to_dict(),
        }

    def save_screenshot(self, payload: dict[str, Any]) -> dict[str, str]:
        data_url = str(payload.get("image_data_url", ""))
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = self.screenshot_dir / f"webcam_capture_{ts}.png"
        meta_path = self.screenshot_dir / f"webcam_capture_{ts}.json"

        img_path.write_bytes(img_bytes)
        meta_payload = {
            "captured_at": datetime.now().isoformat(),
            "image_path": str(img_path),
            "counts": payload.get("counts", {}),
            "detections": payload.get("detections", []),
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        logger.info("Saved browser webcam screenshot to %s", img_path.name)
        return {"filename": img_path.name, "path": str(img_path)}

    def start(self, host: str = "0.0.0.0", port: int = 8511) -> str:
        if self._server is not None:
            return self.base_url

        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Cache-Control", "no-store")

            def _send_json(self, code: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors()
                self.end_headers()

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/health":
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "model": owner.detector.weights_path.name,
                            "device": owner.detector.device,
                        },
                    )
                    return
                self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b""

                try:
                    if path == "/reset":
                        owner.reset_state()
                        self._send_json(200, {"status": "reset"})
                        return

                    if path == "/save_screenshot":
                        payload = json.loads(raw_body.decode("utf-8"))
                        saved = owner.save_screenshot(payload)
                        self._send_json(200, saved)
                        return

                    if path == "/detect":
                        qs = parse_qs(parsed.query)
                        conf = float(qs.get("conf", ["0.35"])[0])
                        iou = float(qs.get("iou", ["0.45"])[0])
                        imgsz = int(qs.get("imgsz", ["320"])[0])
                        track = qs.get("track", ["1"])[0] in {"1", "true", "True"}
                        tracker = qs.get("tracker", ["bytetrack"])[0]
                        line_on = qs.get("line", ["0"])[0] in {"1", "true", "True"}
                        line_orient = qs.get("line_orient", ["horizontal"])[0]
                        line_pos = float(qs.get("line_pos", ["0.5"])[0])
                        roi_on = qs.get("roi", ["0"])[0] in {"1", "true", "True"}

                        result = owner.infer_jpeg(
                            raw_body,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            tracking_enabled=track,
                            tracker_type=tracker,
                            line_enabled=line_on,
                            line_orientation=line_orient,
                            line_position=line_pos,
                            roi_enabled=roi_on,
                        )
                        self._send_json(200, result)
                        return

                    self._send_json(404, {"error": "not_found"})
                except Exception as exc:
                    logger.error("Live API error on %s: %s", path, exc)
                    self._send_json(400, {"error": str(exc)})

        server: ThreadingHTTPServer | None = None
        # Try requested port first, then fall back to dynamic ephemeral port (0)
        ports_to_try = [port, 0] if port > 0 else [0]
        for candidate_port in ports_to_try:
            try:
                server = ThreadingHTTPServer((host, candidate_port), _Handler)
                break
            except OSError:
                if candidate_port == 0:
                    raise
                continue

        if server is None:
            raise RuntimeError(f"Unable to bind LiveInferenceServer to {host}:{port}")

        server.daemon_threads = True
        self._server = server
        self.host = host
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name=f"DetectObject-LiveAPI-{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Live browser inference API started at %s (host=%s, port=%s)", self.base_url, host, self.port)
        return self.base_url

    @property
    def base_url(self) -> str:
        host_repr = "127.0.0.1" if self.host in {"0.0.0.0", ""} else self.host
        return f"http://{host_repr}:{self.port}" if self.port else ""

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        self._thread = None
        self.port = 0
