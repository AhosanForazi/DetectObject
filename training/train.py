"""CLI entrypoint to train the DetectObject YOLO model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration.settings import load_config
from training.trainer import YOLOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DetectObject YOLO model.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=str, default=None, help="Batch size or 'auto'")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image resolution")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    trainer = YOLOTrainer(config=config)
    summary = trainer.train(
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        learning_rate=args.lr,
        resume=args.resume,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
