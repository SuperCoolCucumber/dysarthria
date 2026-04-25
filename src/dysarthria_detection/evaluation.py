from __future__ import annotations

from typing import Any

import pandas as pd


def cross_dataset_split(
    df: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["dataset"] == train_dataset].reset_index(drop=True)
    test_df = df[df["dataset"] == test_dataset].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError(f"Missing data for train={train_dataset} or test={test_dataset}")
    return train_df, test_df


def build_comparison_table(
    baseline_results: dict[str, Any],
    wav2vec_results: dict[str, Any],
    cross_results: dict[str, Any],
) -> pd.DataFrame:
    columns = ["setting", "task", "model", "accuracy", "precision", "recall", "f1"]
    rows: list[dict[str, Any]] = []

    def add_row(setting: str, task: str, model_name: str, metrics: dict[str, float] | None) -> None:
        if not metrics:
            return
        rows.append(
            {
                "setting": setting,
                "task": task,
                "model": model_name,
                **metrics,
            }
        )

    add_row("intra_dataset_test", "binary", "MFCC+SVM", baseline_results.get("binary", {}).get("metrics"))
    add_row("intra_dataset_test", "severity", "MFCC+SVM", baseline_results.get("severity", {}).get("metrics"))
    add_row("intra_dataset_test", "binary", "wav2vec2", wav2vec_results.get("binary", {}).get("metrics"))
    add_row("intra_dataset_test", "severity", "wav2vec2", wav2vec_results.get("severity", {}).get("metrics"))

    for split_name, split_results in cross_results.items():
        add_row(split_name, "binary", "MFCC+SVM", split_results.get("baseline_binary", {}).get("metrics"))
        add_row(split_name, "severity", "MFCC+SVM", split_results.get("baseline_severity", {}).get("metrics"))
        add_row(split_name, "binary", "wav2vec2", split_results.get("w2v_binary", {}).get("metrics"))
        add_row(split_name, "severity", "wav2vec2", split_results.get("w2v_severity", {}).get("metrics"))

    return pd.DataFrame(rows, columns=columns)
