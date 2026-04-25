#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dysarthria_detection.cli import _load_config
from dysarthria_detection.downloads import prepare_kaggle_data


if __name__ == "__main__":
    cfg = _load_config(sys.argv[1:], base_dir=Path(__file__).resolve().parent)
    prepare_kaggle_data(cfg.data)
