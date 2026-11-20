"""DetectObject primary entry point.

Default usage (Windows & Linux):
    python app.py

Optional modes:
    python app.py --port 8501
    python app.py --webcam          # Launch standalone OpenCV desktop webcam window
    python app.py --info            # Print hardware & model readiness summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DetectObject — AI Real-Time Object Detection Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard (default: 8501)")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode without opening a browser")
    parser.add_argument("--webcam", action="store_true", help="Run standalone desktop OpenCV live webcam detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for --webcam mode (default: 0)")
    parser.add_argument("--info", action="store_true", help="Print system resources and model status")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.info:
        from configuration.settings import get_system_resources, load_config

        cfg = load_config()
        best_path = cfg.resolve_path(cfg.get("model", "default_weights", default="models/weights/best.pt"))
        fallback_path = cfg.resolve_path(cfg.get("model", "fallback_weights", default="models/weights/yolo11n.pt"))
        payload = {
            "project": cfg.get("project", "name", default="DetectObject"),
            "version": cfg.get("project", "version", default="1.0.0"),
            "best_weights_exists": best_path.exists(),
            "fallback_weights_exists": fallback_path.exists(),
            "system_resources": get_system_resources(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.webcam:
        from detection.webcam_detector import run_desktop_webcam

        return run_desktop_webcam(camera_index=args.camera)

    dashboard_path = PROJECT_ROOT / "application" / "dashboard.py"
    if not dashboard_path.exists():
        print(f"ERROR: Dashboard script not found at {dashboard_path}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    print(f"Launching DetectObject Dashboard on http://localhost:{args.port} ...")
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
