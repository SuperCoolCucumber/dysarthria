from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from joblib import load
from omegaconf import OmegaConf

from dysarthria_detection.audio import drop_unreadable_audio_rows
from dysarthria_detection.baseline import make_feature_matrix
from dysarthria_detection.pipeline import _save_test_predictions
from dysarthria_detection.utils import resolve_device, set_seed
from dysarthria_detection.wav2vec import build_wav2vec2_model, evaluate_wav2vec2


def _load_cfg(run_dir: Path):
    resolved = run_dir / "config.resolved.yaml"
    if resolved.exists():
        return OmegaConf.load(resolved)

    composed = run_dir / "config.composed.yaml"
    if composed.exists():
        return OmegaConf.load(composed)

    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config.yaml")


def _export_baseline(run_dir: Path, task: str) -> Path:
    cfg = _load_cfg(run_dir)
    label_col = "binary_label" if task == "binary" else "severity_label"
    test_path = run_dir / "metadata" / f"test_{task}.csv"
    model_path = run_dir / "models" / f"baseline_{task}.joblib"
    output_path = run_dir / "tables" / f"baseline_{task}_test_predictions.csv"

    test_df = pd.read_csv(test_path)
    model = load(model_path)
    x_test, aligned_test_df = make_feature_matrix(
        test_df,
        audio_cfg=cfg.audio,
        baseline_cfg=cfg.baseline,
        desc=f"export baseline {task} test",
    )
    y_pred = model.predict(x_test).tolist()
    y_true = aligned_test_df[label_col].astype(int).tolist()
    _save_test_predictions(output_path, aligned_test_df, label_col, y_true, y_pred)
    return output_path


def _infer_num_labels(test_df: pd.DataFrame, label_col: str, run_dir: Path) -> int:
    label_max = int(test_df[label_col].max())
    for split_name in ("train", "val"):
        split_path = run_dir / "metadata" / f"{split_name}_{'binary' if label_col == 'binary_label' else 'severity'}.csv"
        if split_path.exists():
            label_max = max(label_max, int(pd.read_csv(split_path)[label_col].max()))
    return label_max + 1


def _export_wav2vec(run_dir: Path, task: str, device: str) -> Path:
    cfg = _load_cfg(run_dir)
    label_col = "binary_label" if task == "binary" else "severity_label"
    average_mode = "binary" if task == "binary" else "macro"
    test_path = run_dir / "metadata" / f"test_{task}.csv"
    model_path = run_dir / "models" / f"wav2vec2_{task}_best.pt"
    output_path = run_dir / "tables" / f"wav2vec_{task}_test_predictions.csv"

    test_df = pd.read_csv(test_path)
    clean_test_df = drop_unreadable_audio_rows(
        test_df,
        target_sr=int(cfg.audio.target_sr),
        max_audio_sec=float(cfg.audio.max_audio_sec),
        desc=f"export wav2vec {task} test",
    )
    model = build_wav2vec2_model(cfg.wav2vec, num_labels=_infer_num_labels(clean_test_df, label_col, run_dir))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    eval_result = evaluate_wav2vec2(
        model=model,
        df=clean_test_df,
        label_col=label_col,
        average_mode=average_mode,
        audio_cfg=cfg.audio,
        wav2vec_cfg=cfg.wav2vec,
        device=device,
    )
    _save_test_predictions(
        output_path,
        clean_test_df,
        label_col,
        eval_result["y_true"],
        eval_result["y_pred"],
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test prediction CSVs for completed runs.")
    parser.add_argument("--baseline-binary", type=Path, action="append", default=[])
    parser.add_argument("--baseline-severity", type=Path, action="append", default=[])
    parser.add_argument("--wav2vec-binary", type=Path, action="append", default=[])
    parser.add_argument("--wav2vec-severity", type=Path, action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict-cuda-check", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device, strict_cuda_check=bool(args.strict_cuda_check))
    set_seed(42, device=device)
    exported: list[Path] = []
    for run_dir in args.baseline_binary:
        exported.append(_export_baseline(run_dir, "binary"))
    for run_dir in args.baseline_severity:
        exported.append(_export_baseline(run_dir, "severity"))
    for run_dir in args.wav2vec_binary:
        exported.append(_export_wav2vec(run_dir, "binary", device=device))
    for run_dir in args.wav2vec_severity:
        exported.append(_export_wav2vec(run_dir, "severity", device=device))

    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
