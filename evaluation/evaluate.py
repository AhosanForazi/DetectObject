"""CLI entrypoint to evaluate the trained DetectObject model and generate report + plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration.settings import load_config
from evaluation.metrics import EvaluationEngine
from evaluation.report import EvaluationReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DetectObject model on independent test split.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=None, help="Evaluation image size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    engine = EvaluationEngine(config=config)
    metrics = engine.evaluate(
        weights_path=args.weights,
        dataset_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
    )
    generator = EvaluationReportGenerator(config=config)
    artifacts = generator.generate(metrics)
    print(
        json.dumps(
            {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "map50": metrics.map50,
                "map50_95": metrics.map50_95,
                "fps_benchmarks": metrics.fps_benchmarks,
                "report_path": artifacts["report_path"],
                "plots": artifacts["plots"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
