from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    metadata_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_device(requested: str = "auto", strict_cuda_check: bool = True) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and strict_cuda_check and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return requested


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def prepare_run_paths(outputs_cfg: Any, experiment_name: str) -> RunPaths:
    root_dir = Path(str(outputs_cfg.root_dir))
    if not root_dir.is_absolute():
        root_dir = Path.cwd() / root_dir

    run_name = str(outputs_cfg.run_name) if outputs_cfg.run_name else ""
    if not run_name:
        run_name = f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"

    if bool(outputs_cfg.append_slurm_job_id):
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if slurm_job_id and f"slurm{slurm_job_id}" not in run_name:
            run_name = f"{run_name}_slurm{slurm_job_id}"

    run_dir = ensure_dir(root_dir / run_name)
    return RunPaths(
        run_dir=run_dir,
        metadata_dir=ensure_dir(run_dir / "metadata"),
        models_dir=ensure_dir(run_dir / "models"),
        reports_dir=ensure_dir(run_dir / "reports"),
        figures_dir=ensure_dir(run_dir / "figures"),
        tables_dir=ensure_dir(run_dir / "tables"),
    )


def save_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=True)


def save_config_snapshots(cfg: Any, run_dir: str | Path) -> None:
    target = Path(run_dir)
    OmegaConf.save(cfg, target / "config.composed.yaml", resolve=False)
    OmegaConf.save(cfg, target / "config.resolved.yaml", resolve=True)


def environment_summary(device: str) -> dict[str, str]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
    }
