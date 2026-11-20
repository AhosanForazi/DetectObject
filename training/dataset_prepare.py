"""CLI entrypoint to download, validate, clean, and split the DetectObject dataset."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration.settings import load_config
from datasets.dataset_loader import DatasetManager


def main() -> int:
    config = load_config()
    manager = DatasetManager(config=config)
    summary = manager.run_full_pipeline()
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
