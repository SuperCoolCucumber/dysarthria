from __future__ import annotations

from typing import Any

import math
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def empty_metrics() -> dict[str, float]:
    return {
        "accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "f1": math.nan,
    }


def metric_bundle(y_true: list[int], y_pred: list[int], average: str = "binary") -> dict[str, float]:
    if not y_true:
        return empty_metrics()
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_classification_report(y_true: list[int], y_pred: list[int]) -> str:
    if not y_true:
        return "No evaluation samples were available."
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


def print_metrics_table(name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isnan(float(value)):
            print(f"{key:>10}: {float(value):.4f}")
        else:
            print(f"{key:>10}: {value}")
