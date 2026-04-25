from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_comparison_f1(comparison_df: pd.DataFrame, output_path: str | Path) -> None:
    plot_df = comparison_df[pd.notna(comparison_df["f1"])].copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="setting", y="f1", hue="model")
    plt.title("F1 Comparison Across Settings")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_report_text(comparison: pd.DataFrame, severity_note_text: str = "") -> str:
    def best_line(task: str, setting: str) -> str:
        subset = comparison[(comparison["task"] == task) & (comparison["setting"] == setting)]
        if subset.empty:
            return f"- No result available for {task} / {setting}."
        valid = subset[pd.notna(subset["f1"])]
        if valid.empty:
            return f"- {setting} / {task}: metrics unavailable (all candidate F1 values are NaN)."
        best_row = valid.loc[valid["f1"].idxmax()]
        accuracy = best_row["accuracy"]
        accuracy_text = f"{accuracy:.4f}" if pd.notna(accuracy) else "nan"
        return (
            f"- {setting} / {task}: best model={best_row['model']}, "
            f"F1={best_row['f1']:.4f}, Accuracy={accuracy_text}"
        )

    return f"""# Dysarthria Detection and Severity Classification Report

## Objective
Build an end-to-end pipeline for automatic dysarthria detection (binary) and severity classification (multi-class) using TORGO and UA Speech datasets, and compare a classical baseline against wav2vec2.

## Methodology
- Data preparation: load TORGO and UA Speech audio, resample to 16 kHz, trim or pad to fixed duration, and normalize.
- Baseline: MFCC + delta + delta-delta summary statistics with an SVM classifier.
- Deep model: wav2vec2 fine-tuning with partial encoder freezing, AdamW, and warmup scheduling.
- Evaluation: held-out speaker split plus cross-dataset transfer (TORGO->UA and UA->TORGO).
- Interpretability: optional SVM permutation importance and wav2vec2 saliency / attention rollout plots.

## Results
{best_line("binary", "intra_dataset_test")}
{best_line("severity", "intra_dataset_test")}
{best_line("binary", "TORGO_to_UA")}
{best_line("severity", "TORGO_to_UA")}
{best_line("binary", "UA_to_TORGO")}
{best_line("severity", "UA_to_TORGO")}

## Comparison
- wav2vec2 usually gives stronger binary detection performance when the pretrained representation transfers well.
- MFCC + SVM stays useful as a lightweight baseline and can remain competitive on smaller or noisier splits.
- Cross-dataset performance is typically worse than intra-dataset performance because speaker populations, prompts, and recording conditions differ.

## Limitations
- Path-derived labels are only a heuristic unless you provide a curated metadata CSV.
- The imported notebook design keeps the held-out split small for practicality rather than exhaustive model selection.
- Severity classes can be sparse and inconsistent across packaged dataset variants.
- {severity_note_text if severity_note_text else "Severity labels may be approximate in some UA Speech packages."}

## Conclusion
This repository now provides a reproducible, cluster-oriented project structure around the original notebook logic. For publication-grade experiments, the next step should be curated metadata and a stricter train/validation/test protocol.
"""
