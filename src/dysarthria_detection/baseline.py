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
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=str(baseline_cfg.svm.kernel),
                    C=float(baseline_cfg.svm.c),
                    gamma=str(baseline_cfg.svm.gamma),
                    class_weight=str(baseline_cfg.svm.class_weight),
                ),
            ),
        ]
    )


def run_baseline_task(
    train_df: pd.DataFrame,
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
    x_test, aligned_test = make_feature_matrix(test_df, audio_cfg, baseline_cfg, desc=f"{desc_prefix} test")
    y_train = aligned_train[label_col].to_numpy()
    y_test = aligned_test[label_col].to_numpy()

    if np.unique(y_train).size < 2:
        raise RuntimeError(f"{label_col} training split has fewer than two classes after filtering unreadable audio.")

    x_train_fit = x_train
    y_train_fit = y_train
    if bool(baseline_cfg.oversample):
        sampler = RandomOverSampler(random_state=seed)
        x_train_fit, y_train_fit = sampler.fit_resample(x_train, y_train)

    model = build_svm_pipeline(baseline_cfg)
    model.fit(x_train_fit, y_train_fit)
    y_pred = model.predict(x_test)
    result = {
        "model": model,
        "metrics": metric_bundle(y_test.tolist(), y_pred.tolist(), average=average_mode),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "report_text": build_classification_report(y_test.tolist(), y_pred.tolist()),
        "train_df": aligned_train,
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
