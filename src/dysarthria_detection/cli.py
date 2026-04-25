from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .pipeline import run_pipeline
from .utils import resolve_device


def _resolve_config_dir(config_path: str, base_dir: Path | None = None) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate

    search_roots: list[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
    search_roots.append(Path.cwd())

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    fallback_root = search_roots[0] if search_roots else Path.cwd()
    return (fallback_root / candidate).resolve()


def _load_config(argv: Sequence[str], base_dir: Path | None = None) -> DictConfig:
    default_config_path = "../config" if base_dir is not None else "config"
    parser = argparse.ArgumentParser(description="Run the dysarthria detection experiment pipeline.")
    parser.add_argument("--config-path", default=default_config_path, help="Path to the Hydra config directory.")
    parser.add_argument("--config-name", default="config", help="Hydra config name, with or without .yaml.")
    args, overrides = parser.parse_known_args(list(argv))

    config_dir = _resolve_config_dir(args.config_path, base_dir=base_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")

    config_name = args.config_name[:-5] if args.config_name.endswith(".yaml") else args.config_name
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def main(argv: Sequence[str] | None = None, base_dir: Path | None = None) -> dict:
    cfg = _load_config(argv if argv is not None else sys.argv[1:], base_dir=base_dir)
    device = resolve_device(
        str(cfg.runtime.device),
        strict_cuda_check=bool(getattr(cfg.runtime, "strict_cuda_check", True)),
    )
    return run_pipeline(cfg, device=device)
