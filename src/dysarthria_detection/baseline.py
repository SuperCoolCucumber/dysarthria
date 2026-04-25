from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from .audio import load_audio_mono
from .evaluation import cross_dataset_split
from .metrics import build_classification_report, metric_bundle


def extract_mfcc_features(path: str, audio_cfg: Any, n_mfcc: int) -> np.ndarray:
    waveform = load_audio_mono(
        path,
        target_sr=int(audio_cfg.target_sr),
        max_audio_sec=float(audio_cfg.max_audio_sec),
    )
    mfcc = librosa.feature.mfcc(y=waveform, sr=int(audio_cfg.target_sr), n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    stats = np.concatenate(
        [
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.min(features, axis=1),
            np.max(features, axis=1),
        ]
    )
    return stats.astype(np.float32)


def make_feature_matrix(
    df: pd.DataFrame,
    audio_cfg: Any,
    baseline_cfg: Any,
    desc: str = "Extracting MFCC",
) -> tuple[np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    keep_rows: list[int] = []
    failures = 0
    first_failure: str | None = None
    for index, audio_path in enumerate(tqdm(df["audio_path"].tolist(), desc=desc)):
        try:
            features.append(
                extract_mfcc_features(
                    audio_path,
                    audio_cfg=audio_cfg,
                    n_mfcc=int(baseline_cfg.n_mfcc),
                )
            )
            keep_rows.append(index)
        except Exception as exc:
            failures += 1
            if first_failure is None:
                first_failure = f"{audio_path} ({exc})"

    if not features:
        raise RuntimeError(f"No MFCC features were extracted. First failure: {first_failure}")
    if failures:
        print(f"[make_feature_matrix] Skipped {failures} files. Example: {first_failure}")

    aligned = df.iloc[keep_rows].reset_index(drop=True)
    return np.vstack(features), aligned


def build_svm_pipeline(baseline_cfg: Any) -> Pipeline:
    return build_svm_pipeline_with_params(
        baseline_cfg,
        c=float(baseline_cfg.svm.c),
        gamma=str(baseline_cfg.svm.gamma),
    )


def build_svm_pipeline_with_params(
    baseline_cfg: Any,
    c: float,
    gamma: str,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=str(baseline_cfg.svm.kernel),
                    C=float(c),
                    gamma=str(gamma),
                    class_weight=str(baseline_cfg.svm.class_weight),
                ),
            ),
        ]
    )


def _candidate_values(value: Any, fallback: Any) -> list[Any]:
    if value is None:
        return [fallback]
    if isinstance(value, (list, tuple)):
        return list(value) or [fallback]
    return [value]


def _apply_oversampling(
    x: np.ndarray,
    y: np.ndarray,
    oversample: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not oversample:
        return x, y
    sampler = RandomOverSampler(random_state=seed)
    return sampler.fit_resample(x, y)


def run_baseline_task(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    average_mode: str,
    audio_cfg: Any,
    baseline_cfg: Any,
    seed: int,
    desc_prefix: str,
    return_test_features: bool = False,
) -> dict[str, Any]:
    x_train, aligned_train = make_feature_matrix(train_df, audio_cfg, baseline_cfg, desc=f"{desc_prefix} train")
    if val_df.empty:
        x_val = np.empty((0, x_train.shape[1]), dtype=np.float32)
        aligned_val = val_df.reset_index(drop=True)
    else:
        x_val, aligned_val = make_feature_matrix(val_df, audio_cfg, baseline_cfg, desc=f"{desc_prefix} val")
    x_test, aligned_test = make_feature_matrix(test_df, audio_cfg, baseline_cfg, desc=f"{desc_prefix} test")
    y_train = aligned_train[label_col].to_numpy()
    y_val = aligned_val[label_col].to_numpy()
    y_test = aligned_test[label_col].to_numpy()

    if np.unique(y_train).size < 2:
        raise RuntimeError(f"{label_col} training split has fewer than two classes after filtering unreadable audio.")
    if len(y_val) > 0 and np.unique(y_val).size < 2:
        print(f"[run_baseline_task] {label_col} validation split has fewer than two classes after filtering.")

    selection_metric = str(getattr(baseline_cfg, "selection_metric", "f1"))
    c_candidates = [float(value) for value in _candidate_values(getattr(baseline_cfg.svm, "c_candidates", None), baseline_cfg.svm.c)]
    gamma_candidates = [
        str(value) for value in _candidate_values(getattr(baseline_cfg.svm, "gamma_candidates", None), baseline_cfg.svm.gamma)
    ]

    if len(y_val) > 0:
        best_score = -np.inf
        best_params: dict[str, Any] | None = None
        best_val_metrics: dict[str, float] | None = None
        for c_value in c_candidates:
            for gamma_value in gamma_candidates:
                x_train_fit, y_train_fit = _apply_oversampling(
                    x_train,
                    y_train,
                    oversample=bool(baseline_cfg.oversample),
                    seed=seed,
                )
                candidate_model = build_svm_pipeline_with_params(baseline_cfg, c=c_value, gamma=gamma_value)
                candidate_model.fit(x_train_fit, y_train_fit)
                y_val_pred = candidate_model.predict(x_val)
                val_metrics = metric_bundle(y_val.tolist(), y_val_pred.tolist(), average=average_mode)
                score = float(val_metrics.get(selection_metric, np.nan))
                if np.isnan(score):
                    score = -np.inf
                if score > best_score:
                    best_score = score
                    best_params = {"c": c_value, "gamma": gamma_value}
                    best_val_metrics = val_metrics

        if best_params is None or best_val_metrics is None:
            raise RuntimeError("Baseline model selection failed to produce a valid validation score.")
        x_trainval = np.vstack([x_train, x_val])
        y_trainval = np.concatenate([y_train, y_val])
    else:
        best_params = {"c": float(baseline_cfg.svm.c), "gamma": str(baseline_cfg.svm.gamma)}
        best_val_metrics = {}
        x_trainval = x_train
        y_trainval = y_train
    x_trainval_fit, y_trainval_fit = _apply_oversampling(
        x_trainval,
        y_trainval,
        oversample=bool(baseline_cfg.oversample),
        seed=seed,
    )
    model = build_svm_pipeline_with_params(
        baseline_cfg,
        c=float(best_params["c"]),
        gamma=str(best_params["gamma"]),
    )
    model.fit(x_trainval_fit, y_trainval_fit)
    y_pred = model.predict(x_test)
    result = {
        "model": model,
        "metrics": metric_bundle(y_test.tolist(), y_pred.tolist(), average=average_mode),
        "val_metrics": best_val_metrics,
        "best_params": best_params,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "report_text": build_classification_report(y_test.tolist(), y_pred.tolist()),
        "train_df": aligned_train,
        "val_df": aligned_val,
        "test_df": aligned_test,
    }
    if return_test_features:
        result["x_test"] = x_test
    return result


def run_cross_eval_baseline(
    df: pd.DataFrame,
    train_ds: str,
    test_ds: str,
    label_col: str,
    average_mode: str,
    audio_cfg: Any,
    baseline_cfg: Any,
    seed: int,
) -> dict[str, Any]:
    train_df, test_df = cross_dataset_split(df, train_ds, test_ds)
    result = run_baseline_task(
        train_df=train_df,
        val_df=test_df.iloc[0:0].copy(),
        test_df=test_df,
        label_col=label_col,
        average_mode=average_mode,
        audio_cfg=audio_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
        desc_prefix=f"MFCC {train_ds}->{test_ds} {label_col}",
        return_test_features=False,
    )
    return {
        "metrics": result["metrics"],
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
    }
