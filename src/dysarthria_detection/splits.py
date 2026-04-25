from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def speaker_stratified_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for dataset_name in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset_name].copy()
        speakers = dataset_df["speaker_id"].unique().tolist()
        rng.shuffle(speakers)
        num_test = max(1, int(len(speakers) * test_ratio))
        test_speakers = set(speakers[:num_test])
        test_parts.append(dataset_df[dataset_df["speaker_id"].isin(test_speakers)])
        train_parts.append(dataset_df[~dataset_df["speaker_id"].isin(test_speakers)])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def speaker_stratified_three_way_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval_df, test_df = speaker_stratified_split(df, test_ratio=test_ratio, seed=seed)
    if val_ratio <= 0:
        return trainval_df.reset_index(drop=True), trainval_df.iloc[0:0].copy(), test_df.reset_index(drop=True)

    adjusted_val_ratio = val_ratio / max(1e-9, (1.0 - test_ratio))
    train_df, val_df = speaker_stratified_split(trainval_df, test_ratio=adjusted_val_ratio, seed=seed + 10_000)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def speaker_stratified_severity_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 30 or df["speaker_id"].nunique() < 6:
        return speaker_stratified_split(df, test_ratio=test_ratio, seed=seed)

    num_splits = int(round(1.0 / max(0.05, min(0.5, test_ratio))))
    num_splits = max(2, min(num_splits, 10))
    try:
        splitter = StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=seed)
        train_index, test_index = next(
            splitter.split(
                X=np.zeros(len(df)),
                y=df["severity_label"].to_numpy(),
                groups=df["speaker_id"].to_numpy(),
            )
        )
        return (
            df.iloc[train_index].reset_index(drop=True),
            df.iloc[test_index].reset_index(drop=True),
        )
    except ValueError:
        return speaker_stratified_split(df, test_ratio=test_ratio, seed=seed)


def _coverage_counts(y: pd.Series, labels: list[int]) -> dict[int, int]:
    value_counts = y.value_counts().to_dict()
    return {label: int(value_counts.get(label, 0)) for label in labels}


def _has_full_coverage(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: list[int]) -> bool:
    train_counts = _coverage_counts(train_df["severity_label"], labels)
    test_counts = _coverage_counts(test_df["severity_label"], labels)
    return all(count > 0 for count in train_counts.values()) and all(count > 0 for count in test_counts.values())


def _has_full_three_way_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[int],
) -> bool:
    return (
        all(count > 0 for count in _coverage_counts(train_df["severity_label"], labels).values())
        and all(count > 0 for count in _coverage_counts(val_df["severity_label"], labels).values())
        and all(count > 0 for count in _coverage_counts(test_df["severity_label"], labels).values())
    )


def collapse_severity_4_to_3(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = df.copy()
    collapsed["severity_label"] = collapsed["severity_label"].map({0: 0, 1: 0, 2: 1, 3: 2}).astype(int)
    return collapsed


def find_best_severity_three_way_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    expected_labels: list[int],
    max_tries: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    best_triplet: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_score = -1
    adjusted_val_ratio = val_ratio / max(1e-9, (1.0 - test_ratio))
    for offset in range(max_tries):
        trainval_df, test_df = speaker_stratified_severity_split(df, test_ratio=test_ratio, seed=seed + offset)
        if val_ratio > 0:
            train_df, val_df = speaker_stratified_severity_split(
                trainval_df,
                test_ratio=adjusted_val_ratio,
                seed=seed + offset + 10_000,
            )
        else:
            train_df, val_df = trainval_df, trainval_df.iloc[0:0].copy()
        train_cov = _coverage_counts(train_df["severity_label"], expected_labels)
        val_cov = _coverage_counts(val_df["severity_label"], expected_labels)
        test_cov = _coverage_counts(test_df["severity_label"], expected_labels)
        score = (
            sum(int(value > 0) for value in train_cov.values())
            + sum(int(value > 0) for value in val_cov.values())
            + sum(int(value > 0) for value in test_cov.values())
        )
        if score > best_score:
            best_score = score
            best_triplet = (train_df, val_df, test_df)
        if _has_full_three_way_coverage(train_df, val_df, test_df, expected_labels):
            return train_df, val_df, test_df, True

    if best_triplet is None:
        raise RuntimeError("Could not produce a severity split.")
    return best_triplet[0], best_triplet[1], best_triplet[2], False


def prepare_splits(meta_df: pd.DataFrame, data_cfg: Any, seed: int) -> dict[str, Any]:
    val_ratio = float(data_cfg.val_ratio)
    test_ratio = float(data_cfg.test_ratio)
    if val_ratio < 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Expected 0 <= val_ratio, 0 < test_ratio, and val_ratio + test_ratio < 1.")

    train_df, val_df, test_df = speaker_stratified_three_way_split(
        meta_df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    severity_df = meta_df[(meta_df["severity_label"] >= 0) & (meta_df["binary_label"] == 1)].reset_index(drop=True)
    if severity_df.empty:
        raise RuntimeError(
            "No valid severity samples were found. Provide a curated metadata CSV or adjust the severity mapping."
        )

    severity_note = ""
    expected_labels = list(range(int(data_cfg.severity_expected_num_classes)))
    sev_train_df, sev_val_df, sev_test_df, ok_full = find_best_severity_three_way_split(
        severity_df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        expected_labels=expected_labels,
        max_tries=int(data_cfg.severity_split_max_tries),
    )

    if not ok_full and bool(data_cfg.severity_allow_collapse_4_to_3):
        print("[severity] Could not achieve full 4-class coverage. Falling back to 3 classes.")
        severity_df = collapse_severity_4_to_3(severity_df)
        expected_labels = [0, 1, 2]
        sev_train_df, sev_val_df, sev_test_df, ok_collapsed = find_best_severity_three_way_split(
            severity_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            expected_labels=expected_labels,
            max_tries=int(data_cfg.severity_split_max_tries),
        )
        if ok_collapsed:
            severity_note = "Severity labels were collapsed from 4 to 3 classes due to split coverage limits."
        else:
            severity_note = (
                "Severity split could not include all 3 classes in train, validation, and test despite retries; "
                "severity metrics should be interpreted cautiously."
            )
    elif not ok_full:
        severity_note = (
            "Severity split could not include all 4 classes in train, validation, and test despite retries; "
            "severity metrics should be interpreted cautiously."
        )

    return {
        "train_df": train_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "severity_df": severity_df.reset_index(drop=True),
        "sev_train_df": sev_train_df.reset_index(drop=True),
        "sev_val_df": sev_val_df.reset_index(drop=True),
        "sev_test_df": sev_test_df.reset_index(drop=True),
        "severity_note": severity_note,
    }
