"""Thread-safe real-time live webcam object detection with built-in local MJPEG streaming."""

from __future__ import annotations

import atexit
import json
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configuration.settings import AppConfig, get_logger, load_config
from detection.object_counter import CountingConfig, CountSnapshot, ObjectCounter
from detection.visualizer import DetectionVisualizer
from models.detector import DetectionResult, ObjectDetector
from models.tracker import ObjectTracker

logger = get_logger("detectobject.webcam")


def _open_camera_capture(camera_index: int) -> cv2.VideoCapture:
    """Open camera with OS-appropriate backend (DirectShow first on Windows)."""
    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def enumerate_cameras(max_index: int = 3) -> list[dict[str, Any]]:
    """Safely probe available local cameras and release every handle immediately."""
    available: list[dict[str, Any]] = []
    for idx in range(max_index):
        cap = _open_camera_capture(idx)
        try:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    available.append(
                        {
                            "index": idx,
                            "label": f"Camera {idx} ({w}x{h})",
                            "width": w,
                            "height": h,
                        }
                    )
        except Exception as exc:
            logger.debug("Camera probe error at index %d: %s", idx, exc)
        finally:
            cap.release()
    return available


@dataclass
class WebcamFramePacket:
    """Latest processed frame and telemetry emitted by WebcamDetector."""

    annotated_bgr: np.ndarray | None = None
    annotated_rgb: np.ndarray | None = None
    jpeg_bytes: bytes | None = None
    result: DetectionResult = field(default_factory=DetectionResult)
    count_snapshot: CountSnapshot = field(default_factory=CountSnapshot)
    fps: float = 0.0
    frame_number: int = 0
    camera_index: int = 0
    paused: bool = False
    running: bool = False
    low_variance_warning: bool = False
    error: str | None = None
    timestamp: float = 0.0


class _MJPEGStreamServer:
    """Lightweight localhost-only MJPEG HTTP server for silky-smooth browser video playback."""

    def __init__(self, owner: "WebcamDetector") -> None:
        self._owner = owner
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def start(self) -> str:
        if self._server is not None:
            return self.url

        owner = self._owner

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/snapshot.jpg":
                    packet = owner.get_latest_packet()
                    data = packet.jpeg_bytes
                    if not data:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                if path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    last_sent_frame = -1
                    try:
                        while owner.is_running:
                            packet = owner.get_latest_packet()
                            data = packet.jpeg_bytes
                            if data and (packet.frame_number != last_sent_frame or packet.paused):
                                last_sent_frame = packet.frame_number
                                header = (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    + f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                                )
                                self.wfile.write(header)
                                self.wfile.write(data)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                                if packet.paused:
                                    time.sleep(0.15)
                                else:
                                    time.sleep(0.025)
                            else:
                                time.sleep(0.025)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        return
                    return

                self.send_response(404)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name=f"DetectObject-MJPEG-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream.mjpg" if self.port else ""

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


class WebcamDetector:
    """Thread-safe real-time webcam detection controller with same-thread capture and MJPEG streaming."""

    def __init__(
        self,
        detector: ObjectDetector,
        config: AppConfig | None = None,
        visualizer: DetectionVisualizer | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or load_config()
        self.visualizer = visualizer or DetectionVisualizer()

        self.screenshot_dir = self.config.resolve_path(
            self.config.get("webcam", "screenshot_dir", default="outputs/screenshots")
        )
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._mjpeg = _MJPEGStreamServer(self)

        self.camera_index: int = int(self.config.get("webcam", "default_camera_index", default=0))
        self.frame_width: int = int(self.config.get("webcam", "frame_width", default=640))
        self.frame_height: int = int(self.config.get("webcam", "frame_height", default=480))
        self.conf: float = float(self.config.get("model", "confidence_threshold", default=0.35))
        self.iou: float = float(self.config.get("model", "iou_threshold", default=0.45))
        self.imgsz: int = int(self.config.get("webcam", "inference_imgsz", default=320))
        self.frame_skip: int = max(1, int(self.config.get("webcam", "frame_skip", default=1)))
        self.tracking_enabled: bool = bool(self.config.get("tracking", "enabled", default=True))
        self.tracker_type: str = str(self.config.get("tracking", "tracker_type", default="bytetrack"))
        self.show_trails: bool = bool(self.config.get("tracking", "show_trails", default=True))
        self.selected_classes: list[int] | None = None
        self.counting_config = CountingConfig(
            line_enabled=bool(self.config.get("counting", "line_enabled", default=False)),
            line_orientation=str(self.config.get("counting", "line_orientation", default="horizontal")),
            line_position=float(self.config.get("counting", "line_position", default=0.5)),
            roi_enabled=bool(self.config.get("counting", "roi_enabled", default=False)),
            roi_box=tuple(self.config.get("counting", "roi_box", default=[0.15, 0.15, 0.85, 0.85])),  # type: ignore[arg-type]
        )

        self._tracker = ObjectTracker(self.detector, tracker_type=self.tracker_type)
        self._counter = ObjectCounter(self.counting_config)
        self._paused: bool = False
        self._running: bool = False
        self._start_ok: bool = False
        self._start_message: str = ""
        self._last_error: str | None = None
        self._latest_packet = WebcamFramePacket()
        self._fps_timestamps: deque[float] = deque(maxlen=25)
        self._session_nonce: int = 0

        atexit.register(self.stop)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def stream_url(self) -> str:
        base = self._mjpeg.url
        return f"{base}?s={self._session_nonce}" if base else ""

    def update_settings(
        self,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        frame_skip: int | None = None,
        tracking_enabled: bool | None = None,
        tracker_type: str | None = None,
        show_trails: bool | None = None,
        counting_config: CountingConfig | None = None,
        selected_classes: list[int] | None = None,
    ) -> None:
        """Update live inference/tracking/counting settings without restarting the webcam."""
        with self._lock:
            if conf is not None:
                self.conf = float(conf)
            if iou is not None:
                self.iou = float(iou)
            if imgsz is not None:
                self.imgsz = int(imgsz)
            if frame_skip is not None:
                self.frame_skip = max(1, int(frame_skip))
            if tracking_enabled is not None:
                self.tracking_enabled = bool(tracking_enabled)
            if tracker_type is not None and tracker_type.lower() != self.tracker_type:
                self.tracker_type = tracker_type.lower()
                self._tracker.set_tracker_type(self.tracker_type)
            if show_trails is not None:
                self.show_trails = bool(show_trails)
            if counting_config is not None:
                self.counting_config = counting_config
                self._counter.update_config(counting_config)
            self.selected_classes = selected_classes

    def reset_counts(self) -> None:
        with self._lock:
            self._tracker.reset()
            self._counter.reset()

    def start(self, camera_index: int | None = None) -> tuple[bool, str]:
        """Start live webcam capture on a dedicated thread (open/read/release all on same thread)."""
        with self._lock:
            if self._running:
                if camera_index is None or camera_index == self.camera_index:
                    return True, f"Webcam {self.camera_index} is already running."
                self.stop()

            target_idx = self.camera_index if camera_index is None else int(camera_index)
            self.camera_index = target_idx
            self._stop_event.clear()
            self._ready_event.clear()
            self._start_ok = False
            self._start_message = ""
            self._paused = False
            self._last_error = None
            self._fps_timestamps.clear()
            self._tracker.reset()
            self._counter.reset()
            self._session_nonce += 1

            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(target_idx,),
                name=f"DetectObject-Webcam-{target_idx}",
                daemon=True,
            )
            self._thread.start()

        # Wait until camera opens and delivers its first frame (or fails cleanly)
        if not self._ready_event.wait(timeout=8.0):
            self.stop()
            msg = f"Timed out while initializing camera index {target_idx}."
            with self._lock:
                self._last_error = msg
            return False, msg

        with self._lock:
            if not self._start_ok:
                return False, self._start_message

        self._mjpeg.start()
        return True, self._start_message

    def stop(self) -> None:
        """Stop capture thread, stop local MJPEG server, and release camera hardware resources safely."""
        self._stop_event.set()
        self._mjpeg.stop()

        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=4.0)

        with self._lock:
            was_active = self._running or (self._cap is not None)
            self._thread = None
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._running = False
            self._paused = False
            self._latest_packet.running = False
            self._latest_packet.paused = False
            if was_active:
                try:
                    logger.info("Webcam resources released cleanly.")
                except Exception:
                    pass

    def pause(self) -> None:
        with self._lock:
            if self._running:
                self._paused = True
                self._latest_packet.paused = True

    def resume(self) -> None:
        with self._lock:
            if self._running:
                self._paused = False
                self._latest_packet.paused = False

    def switch_camera(self, new_camera_index: int) -> tuple[bool, str]:
        """Safely release current camera and switch to another index."""
        self.stop()
        return self.start(camera_index=new_camera_index)

    def get_latest_packet(self) -> WebcamFramePacket:
        with self._lock:
            return self._latest_packet

    def capture_screenshot(self, prefix: str = "webcam_capture") -> tuple[Path, Path]:
        """Save the latest annotated webcam frame and its detection JSON metadata."""
        packet = self.get_latest_packet()
        if packet.annotated_bgr is None:
            raise RuntimeError("No webcam frame available to capture. Start the webcam first.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = self.screenshot_dir / f"{prefix}_{ts}.png"
        meta_path = self.screenshot_dir / f"{prefix}_{ts}.json"

        cv2.imwrite(str(img_path), packet.annotated_bgr)
        payload = {
            "captured_at": datetime.now().isoformat(),
            "camera_index": packet.camera_index,
            "fps": round(packet.fps, 2),
            "detection_result": packet.result.to_dict(),
            "counts": packet.count_snapshot.to_dict(),
            "image_path": str(img_path),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved webcam screenshot to %s (%d objects)", img_path.name, packet.result.total_count)
        return img_path, meta_path

    def _capture_loop(self, target_idx: int) -> None:
        cap: cv2.VideoCapture | None = None
        frame_num = 0
        last_result = DetectionResult()
        last_trails = None
        last_snapshot = CountSnapshot()
        consecutive_failures = 0

        try:
            cap = _open_camera_capture(target_idx)
            if not cap.isOpened():
                msg = (
                    f"Unable to open camera index {target_idx}. "
                    "Verify the webcam is connected, not used by another application, "
                    "and OS camera permissions are enabled."
                )
                with self._lock:
                    self._start_ok = False
                    self._start_message = msg
                    self._last_error = msg
                    self._latest_packet = WebcamFramePacket(running=False, camera_index=target_idx, error=msg)
                logger.error(msg)
                self._ready_event.set()
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warmup read on the same thread that owns the capture handle
            ret, first_frame = False, None
            for _ in range(3):
                ret, first_frame = cap.read()
                if ret and first_frame is not None:
                    break
                time.sleep(0.05)

            if not ret or first_frame is None:
                msg = (
                    f"Camera {target_idx} opened but failed to deliver a frame. "
                    "Another app or privacy shutter may be blocking the camera."
                )
                with self._lock:
                    self._start_ok = False
                    self._start_message = msg
                    self._last_error = msg
                    self._latest_packet = WebcamFramePacket(running=False, camera_index=target_idx, error=msg)
                logger.error(msg)
                self._ready_event.set()
                return

            ok_jpg, first_buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with self._lock:
                self._cap = cap
                self._running = True
                self._start_ok = True
                self._start_message = f"Webcam {target_idx} started."
                self._latest_packet = WebcamFramePacket(
                    annotated_bgr=first_frame,
                    annotated_rgb=cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                    jpeg_bytes=first_buf.tobytes() if ok_jpg else None,
                    camera_index=target_idx,
                    running=True,
                    timestamp=time.time(),
                )
            logger.info(
                "Webcam %d started successfully (%dx%d).",
                target_idx,
                first_frame.shape[1],
                first_frame.shape[0],
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                with self._lock:
                    paused = self._paused
                    conf = self.conf
                    iou = self.iou
                    imgsz = self.imgsz
                    frame_skip = self.frame_skip
                    tracking_enabled = self.tracking_enabled
                    show_trails = self.show_trails
                    counting_cfg = self.counting_config
                    selected_classes = self.selected_classes
                    cam_idx = self.camera_index

                if paused:
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 15:
                        with self._lock:
                            self._last_error = f"Lost signal from camera {cam_idx}."
                            self._latest_packet.error = self._last_error
                        break
                    time.sleep(0.02)
                    continue

                consecutive_failures = 0
                frame_num += 1
                now = time.perf_counter()
                self._fps_timestamps.append(now)
                if len(self._fps_timestamps) >= 2:
                    dt = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    live_fps = (len(self._fps_timestamps) - 1) / dt if dt > 0 else 0.0
                else:
                    live_fps = 0.0

                # Detect privacy shutter / blank uniform frame (very low spatial variance)
                low_var = bool(float(np.std(frame[::8, ::8])) < 14.0)

                should_infer = ((frame_num - 1) % frame_skip == 0)
                if should_infer:
                    if tracking_enabled:
                        last_result = self._tracker.update(
                            frame=frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=selected_classes,
                        )
                        last_trails = self._tracker.get_active_trails() if show_trails else None
                    else:
                        last_result = self.detector.detect(
                            image=frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=selected_classes,
                        )
                        last_trails = None

                    last_snapshot = self._counter.update(last_result.detections, frame.shape[:2])

                annotated_bgr = self.visualizer.draw(
                    frame_bgr=frame,
                    detections=last_result.detections,
                    trails=last_trails,
                    counting_config=counting_cfg,
                    count_snapshot=last_snapshot,
                    fps=live_fps if live_fps > 0 else last_result.fps,
                    show_hud=True,
                    show_confidence=True,
                    show_track_ids=tracking_enabled,
                )
                ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                packet = WebcamFramePacket(
                    annotated_bgr=annotated_bgr,
                    annotated_rgb=annotated_rgb,
                    jpeg_bytes=jpg_buf.tobytes() if ok_jpg else None,
                    result=last_result,
                    count_snapshot=last_snapshot,
                    fps=live_fps if live_fps > 0 else last_result.fps,
                    frame_number=frame_num,
                    camera_index=cam_idx,
                    paused=False,
                    running=True,
                    low_variance_warning=low_var,
                    error=None,
                    timestamp=time.time(),
                )
                with self._lock:
                    self._latest_packet = packet
        except Exception as exc:
            logger.exception("Unhandled error in webcam capture loop: %s", exc)
            with self._lock:
                self._start_ok = False
                self._start_message = f"Webcam stream error: {exc}"
                self._last_error = self._start_message
                self._latest_packet.error = self._last_error
            self._ready_event.set()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self._lock:
                self._cap = None
                self._running = False
                self._latest_packet.running = False


def run_desktop_webcam(camera_index: int = 0) -> int:
    """Standalone OpenCV desktop live webcam window (controls: q=quit, p=pause, s=screenshot)."""
    detector = ObjectDetector()
    webcam = WebcamDetector(detector)
    ok, msg = webcam.start(camera_index=camera_index)
    if not ok:
        print(f"ERROR: {msg}")
        return 1

    print("DetectObject Live Webcam running. Press 'q' to quit, 'p' to pause/resume, 's' to save screenshot.")
    try:
        while webcam.is_running:
            packet = webcam.get_latest_packet()
            if packet.annotated_bgr is not None:
                cv2.imshow("DetectObject — Live Webcam (q: quit | p: pause | s: screenshot)", packet.annotated_bgr)
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                if webcam.is_paused:
                    webcam.resume()
                else:
                    webcam.pause()
            if key == ord("s"):
                img_path, _ = webcam.capture_screenshot()
                print(f"Saved screenshot: {img_path}")
    finally:
        webcam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_desktop_webcam())
