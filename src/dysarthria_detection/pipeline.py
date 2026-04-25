from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import dump
from omegaconf import DictConfig

from .audio import drop_unreadable_audio_rows
from .baseline import run_baseline_task, run_cross_eval_baseline
from .downloads import prepare_kaggle_data
from .evaluation import build_comparison_table
from .interpretability import (
    plot_wav2vec_attention_maps,
    plot_wav2vec_saliency,
    run_baseline_permutation_importance,
)
from .metadata import build_metadata
from .metrics import build_classification_report, print_metrics_table
from .reporting import build_report_text, plot_comparison_f1
from .splits import prepare_splits
from .utils import environment_summary, prepare_run_paths, save_config_snapshots, save_json, save_text, set_seed
from .wav2vec import evaluate_wav2vec2, run_cross_eval_wav2vec, train_one_task


def _save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _clean_split_for_wav2vec(df: pd.DataFrame, cfg: DictConfig, desc: str) -> pd.DataFrame:
    return drop_unreadable_audio_rows(
        df,
        target_sr=int(cfg.audio.target_sr),
        max_audio_sec=float(cfg.audio.max_audio_sec),
        desc=desc,
    )


def run_pipeline(cfg: DictConfig, device: str) -> dict[str, Any]:
    if bool(cfg.runtime.suppress_warnings):
        warnings.filterwarnings("ignore")

    set_seed(int(cfg.seed), device=device)
    run_paths = prepare_run_paths(cfg.outputs, experiment_name=str(cfg.experiment.name))
    prepare_kaggle_data(cfg.data)
    save_config_snapshots(cfg, run_paths.run_dir)
    save_json(run_paths.run_dir / "environment.json", environment_summary(device=device))

    print(f"Using device: {device}")
    metadata_df = build_metadata(cfg.data, seed=int(cfg.seed))
    print("\nDataset counts:")
    print(metadata_df["dataset"].value_counts())
    print("\nBinary label counts:")
    print(metadata_df["binary_label"].value_counts(dropna=False))
    print("\nSeverity label counts:")
    print(metadata_df["severity_label"].value_counts(dropna=False).sort_index())

    if bool(cfg.outputs.save_metadata):
        _save_dataframe(run_paths.metadata_dir / "metadata.csv", metadata_df)

    splits = prepare_splits(metadata_df, cfg.data, seed=int(cfg.seed))
    print(f"\nBinary train/test sizes: {len(splits['train_df'])} / {len(splits['test_df'])}")
    print(f"Severity train/test sizes: {len(splits['sev_train_df'])} / {len(splits['sev_test_df'])}")
    if splits["severity_note"]:
        print(f"[severity] {splits['severity_note']}")

    if bool(cfg.outputs.save_split_csvs):
        _save_dataframe(run_paths.metadata_dir / "train_binary.csv", splits["train_df"])
        _save_dataframe(run_paths.metadata_dir / "test_binary.csv", splits["test_df"])
        _save_dataframe(run_paths.metadata_dir / "train_severity.csv", splits["sev_train_df"])
        _save_dataframe(run_paths.metadata_dir / "test_severity.csv", splits["sev_test_df"])

    baseline_results: dict[str, Any] = {}
    wav2vec_results: dict[str, Any] = {}
    cross_results: dict[str, Any] = {}
    run_baseline = bool(cfg.tasks.run_baseline)
    run_wav2vec = bool(cfg.tasks.run_wav2vec)
    run_wav2vec_binary = run_wav2vec and bool(getattr(cfg.tasks, "run_wav2vec_binary", True))
    run_wav2vec_severity = run_wav2vec and bool(getattr(cfg.tasks, "run_wav2vec_severity", True))
    severity_cross_df = splits["severity_df"]

    if run_baseline:
        baseline_results["binary"] = run_baseline_task(
            train_df=splits["train_df"],
            test_df=splits["test_df"],
            label_col="binary_label",
            average_mode="binary",
            audio_cfg=cfg.audio,
            baseline_cfg=cfg.baseline,
            seed=int(cfg.seed),
            desc_prefix="MFCC binary",
            return_test_features=bool(cfg.evaluation.enable_interpretability),
        )
        if run_wav2vec_severity or bool(cfg.tasks.run_cross_dataset):
            baseline_results["severity"] = run_baseline_task(
                train_df=splits["sev_train_df"],
                test_df=splits["sev_test_df"],
                label_col="severity_label",
                average_mode="macro",
                audio_cfg=cfg.audio,
                baseline_cfg=cfg.baseline,
                seed=int(cfg.seed),
                desc_prefix="MFCC severity",
                return_test_features=False,
            )

        print_metrics_table("Baseline MFCC+SVM (Binary)", baseline_results["binary"]["metrics"])
        if "severity" in baseline_results:
            print_metrics_table("Baseline MFCC+SVM (Severity)", baseline_results["severity"]["metrics"])

        save_text(
            run_paths.reports_dir / "baseline_binary_classification_report.txt",
            baseline_results["binary"]["report_text"],
        )
        if "severity" in baseline_results:
            save_text(
                run_paths.reports_dir / "baseline_severity_classification_report.txt",
                baseline_results["severity"]["report_text"],
            )
        if bool(cfg.outputs.save_models):
            dump(baseline_results["binary"]["model"], run_paths.models_dir / "baseline_binary.joblib")
            if "severity" in baseline_results:
                dump(baseline_results["severity"]["model"], run_paths.models_dir / "baseline_severity.joblib")

        clean_train_df = baseline_results["binary"]["train_df"]
        clean_test_df = baseline_results["binary"]["test_df"]
        clean_sev_train_df = baseline_results.get("severity", {}).get("train_df", splits["sev_train_df"])
        clean_sev_test_df = baseline_results.get("severity", {}).get("test_df", splits["sev_test_df"])
    elif run_wav2vec:
        clean_train_df = _clean_split_for_wav2vec(splits["train_df"], cfg, "binary wav2vec train")
        clean_test_df = _clean_split_for_wav2vec(splits["test_df"], cfg, "binary wav2vec test")
        if run_wav2vec_severity:
            clean_sev_train_df = _clean_split_for_wav2vec(splits["sev_train_df"], cfg, "severity wav2vec train")
            clean_sev_test_df = _clean_split_for_wav2vec(splits["sev_test_df"], cfg, "severity wav2vec test")
        else:
            clean_sev_train_df = splits["sev_train_df"]
            clean_sev_test_df = splits["sev_test_df"]
    else:
        clean_train_df = splits["train_df"]
        clean_test_df = splits["test_df"]
        clean_sev_train_df = splits["sev_train_df"]
        clean_sev_test_df = splits["sev_test_df"]

    if run_wav2vec_binary:
        if clean_train_df.empty or clean_test_df.empty:
            raise RuntimeError("Binary wav2vec split is empty after filtering unreadable audio.")

        binary_model, binary_train_log = train_one_task(
            train_df=clean_train_df,
            valid_df=clean_test_df,
            label_col="binary_label",
            num_labels=2,
            epochs=int(cfg.wav2vec.epochs_binary),
            average_mode="binary",
            run_name="wav2vec2_binary",
            audio_cfg=cfg.audio,
            wav2vec_cfg=cfg.wav2vec,
            device=device,
        )
        binary_eval = evaluate_wav2vec2(
            model=binary_model,
            df=clean_test_df,
            label_col="binary_label",
            average_mode="binary",
            audio_cfg=cfg.audio,
            wav2vec_cfg=cfg.wav2vec,
            device=device,
        )

        wav2vec_results["binary"] = {
            **binary_eval,
            "model": binary_model,
            "train_log": binary_train_log,
            "report_text": build_classification_report(binary_eval["y_true"], binary_eval["y_pred"]),
        }

        print_metrics_table("wav2vec2 (Binary)", wav2vec_results["binary"]["metrics"])
        save_text(
            run_paths.reports_dir / "wav2vec_binary_classification_report.txt",
            wav2vec_results["binary"]["report_text"],
        )
        if bool(cfg.outputs.save_models):
            torch.save(binary_model.state_dict(), run_paths.models_dir / "wav2vec2_binary_best.pt")

    if run_wav2vec_severity:
        if splits["severity_df"].empty:
            raise RuntimeError(
                "No valid severity samples were found. Provide curated metadata or adjust the UA severity mapping."
            )
        if clean_sev_train_df.empty or clean_sev_test_df.empty:
            raise RuntimeError("Severity wav2vec split is empty after filtering unreadable audio.")

        severity_num_labels = int(
            max(clean_sev_train_df["severity_label"].max(), clean_sev_test_df["severity_label"].max()) + 1
        )
        severity_model, severity_train_log = train_one_task(
            train_df=clean_sev_train_df,
            valid_df=clean_sev_test_df,
            label_col="severity_label",
            num_labels=severity_num_labels,
            epochs=int(cfg.wav2vec.epochs_severity),
            average_mode="macro",
            run_name="wav2vec2_severity",
            audio_cfg=cfg.audio,
            wav2vec_cfg=cfg.wav2vec,
            device=device,
        )
        severity_eval = evaluate_wav2vec2(
            model=severity_model,
            df=clean_sev_test_df,
            label_col="severity_label",
            average_mode="macro",
            audio_cfg=cfg.audio,
            wav2vec_cfg=cfg.wav2vec,
            device=device,
        )

        wav2vec_results["severity"] = {
            **severity_eval,
            "model": severity_model,
            "train_log": severity_train_log,
            "report_text": build_classification_report(severity_eval["y_true"], severity_eval["y_pred"]),
        }

        print_metrics_table("wav2vec2 (Severity)", wav2vec_results["severity"]["metrics"])

        save_text(
            run_paths.reports_dir / "wav2vec_severity_classification_report.txt",
            wav2vec_results["severity"]["report_text"],
        )
        if bool(cfg.outputs.save_models):
            torch.save(severity_model.state_dict(), run_paths.models_dir / "wav2vec2_severity_best.pt")

    if bool(cfg.evaluation.enable_interpretability) and run_wav2vec_binary and not clean_test_df.empty:
        binary_model = wav2vec_results["binary"]["model"]
        if run_baseline:
            x_test = baseline_results["binary"].get("x_test")
            if x_test is not None:
                run_baseline_permutation_importance(
                    model=baseline_results["binary"]["model"],
                    x_test=x_test,
                    y_test=np.asarray(baseline_results["binary"]["y_true"]),
                    evaluation_cfg=cfg.evaluation,
                    seed=int(cfg.seed),
                    output_path=run_paths.figures_dir / "baseline_permutation_importance.png",
                )
        sample_path = clean_test_df.iloc[0]["audio_path"]
        plot_wav2vec_saliency(
            model=binary_model,
            audio_path=sample_path,
            audio_cfg=cfg.audio,
            wav2vec_cfg=cfg.wav2vec,
            device=device,
            output_path=run_paths.figures_dir / "wav2vec_binary_saliency.png",
            window_ms=int(cfg.evaluation.saliency_window_ms),
        )

    if bool(cfg.evaluation.enable_attention_maps) and run_wav2vec_binary and not clean_test_df.empty:
        binary_model = wav2vec_results["binary"]["model"]
        max_samples = min(int(cfg.evaluation.attention_num_samples), len(clean_test_df))
        for index in range(max_samples):
            audio_path = clean_test_df.iloc[index]["audio_path"]
            plot_wav2vec_attention_maps(
                model=binary_model,
                audio_path=audio_path,
                audio_cfg=cfg.audio,
                wav2vec_cfg=cfg.wav2vec,
                device=device,
                output_path=run_paths.figures_dir / f"attention_map_binary_sample{index}.png",
            )

    if bool(cfg.tasks.run_cross_dataset):
        for train_ds, test_ds in [("TORGO", "UA"), ("UA", "TORGO")]:
            split_name = f"{train_ds}_to_{test_ds}"
            cross_results[split_name] = {}
            if run_baseline:
                cross_results[split_name]["baseline_binary"] = run_cross_eval_baseline(
                    metadata_df,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    label_col="binary_label",
                    average_mode="binary",
                    audio_cfg=cfg.audio,
                    baseline_cfg=cfg.baseline,
                    seed=int(cfg.seed),
                )
                severity_train_exists = not severity_cross_df[severity_cross_df["dataset"] == train_ds].empty
                severity_test_exists = not severity_cross_df[severity_cross_df["dataset"] == test_ds].empty
                if severity_train_exists and severity_test_exists:
                    cross_results[split_name]["baseline_severity"] = run_cross_eval_baseline(
                        severity_cross_df,
                        train_ds=train_ds,
                        test_ds=test_ds,
                        label_col="severity_label",
                        average_mode="macro",
                        audio_cfg=cfg.audio,
                        baseline_cfg=cfg.baseline,
                        seed=int(cfg.seed),
                    )
                else:
                    cross_results[split_name]["baseline_severity"] = {
                        "metrics": {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
                    }

            if run_wav2vec_binary:
                binary_cross_epochs = max(2, int(cfg.wav2vec.epochs_binary) - 1)
                cross_results[split_name]["w2v_binary"] = run_cross_eval_wav2vec(
                    metadata_df,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    label_col="binary_label",
                    num_labels=2,
                    average_mode="binary",
                    epochs=binary_cross_epochs,
                    run_name=f"w2v_bin_{split_name}",
                    audio_cfg=cfg.audio,
                    wav2vec_cfg=cfg.wav2vec,
                    device=device,
                )
            if run_wav2vec_severity:
                severity_cross_epochs = max(2, int(cfg.wav2vec.epochs_severity) - 1)
                severity_train_exists = not severity_cross_df[severity_cross_df["dataset"] == train_ds].empty
                severity_test_exists = not severity_cross_df[severity_cross_df["dataset"] == test_ds].empty
                if severity_train_exists and severity_test_exists:
                    cross_results[split_name]["w2v_severity"] = run_cross_eval_wav2vec(
                        severity_cross_df,
                        train_ds=train_ds,
                        test_ds=test_ds,
                        label_col="severity_label",
                        num_labels=int(severity_cross_df["severity_label"].max() + 1),
                        average_mode="macro",
                        epochs=severity_cross_epochs,
                        run_name=f"w2v_sev_{split_name}",
                        audio_cfg=cfg.audio,
                        wav2vec_cfg=cfg.wav2vec,
                        device=device,
                    )
                else:
                    cross_results[split_name]["w2v_severity"] = {
                        "metrics": {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
                    }

        for split_name, split_results in cross_results.items():
            print(f"\n=== Cross Eval: {split_name} ===")
            for model_name, payload in split_results.items():
                print_metrics_table(model_name, payload["metrics"])

    comparison_df = build_comparison_table(
        baseline_results=baseline_results,
        wav2vec_results=wav2vec_results,
        cross_results=cross_results,
    )
    _save_dataframe(run_paths.tables_dir / "model_comparison.csv", comparison_df)
    if not comparison_df.empty:
        plot_comparison_f1(comparison_df, run_paths.figures_dir / "model_comparison_f1.png")

    if bool(cfg.tasks.generate_report):
        report_text = build_report_text(comparison_df, severity_note_text=splits["severity_note"])
        save_text(run_paths.reports_dir / "final_report.md", report_text)

    summary_payload = {
        "run_dir": str(run_paths.run_dir),
        "device": device,
        "severity_note": splits["severity_note"],
        "baseline": {
            task: {
                "metrics": payload.get("metrics"),
            }
            for task, payload in baseline_results.items()
        },
        "wav2vec": {
            task: {
                "metrics": payload.get("metrics"),
                "train_log": payload.get("train_log"),
            }
            for task, payload in wav2vec_results.items()
        },
        "cross_results_metrics_only": {
            split_name: {
                model_name: model_payload.get("metrics")
                for model_name, model_payload in split_payload.items()
            }
            for split_name, split_payload in cross_results.items()
        },
    }
    save_json(run_paths.run_dir / "training_logs.json", summary_payload)
    return summary_payload
