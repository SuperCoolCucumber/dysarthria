from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from transformers import AutoFeatureExtractor

from .audio import load_audio_mono
from .wav2vec import backbone_name_from_checkpoint, pretrained_load_kwargs


def run_baseline_permutation_importance(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    evaluation_cfg: Any,
    seed: int,
    output_path: str | Path,
) -> None:
    x_eval = x_test
    y_eval = y_test
    max_samples = int(evaluation_cfg.perm_max_samples or 0)
    if max_samples > 0 and len(x_eval) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(x_eval), size=max_samples, replace=False)
        x_eval = x_eval[indices]
        y_eval = y_eval[indices]

    perm = permutation_importance(
        estimator=model,
        X=x_eval,
        y=y_eval,
        n_repeats=int(evaluation_cfg.perm_n_repeats),
        random_state=seed,
        scoring="f1",
        n_jobs=int(evaluation_cfg.perm_n_jobs),
    )
    plt.figure(figsize=(10, 4))
    plt.plot(perm.importances_mean)
    plt.title("MFCC+SVM Permutation Importance (Binary)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (Mean drop in F1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def wav2vec_attention_rollout(
    model: nn.Module,
    input_values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")

    with torch.no_grad():
        output = model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = output.attentions
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions were returned by the audio classification backbone.")

    layer_attentions = [attention for attention in attentions if attention is not None]
    if not layer_attentions:
        raise RuntimeError(
            "Attention weights are all None. Set wav2vec.attn_implementation=eager and rerun training."
        )

    sequence_length = layer_attentions[0].shape[-1]
    identity = torch.eye(sequence_length, device=input_values.device, dtype=layer_attentions[0].dtype)
    rollout = identity.clone()
    for attention in layer_attentions:
        attn = attention[0].mean(dim=0)
        attn = attn + identity
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        rollout = torch.matmul(attn, rollout)

    importance = rollout.mean(dim=0)
    importance = importance / (importance.sum() + 1e-9)
    return importance, tuple(layer_attentions)


def upsample_encoder_importance_to_audio(
    importance: np.ndarray,
    num_audio_samples: int,
    target_sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoder_steps = importance.shape[0]
    source_x = np.linspace(0, num_audio_samples - 1, num=encoder_steps)
    target_x = np.arange(num_audio_samples, dtype=np.float64)
    upsampled = np.interp(target_x, source_x, importance.astype(np.float64))
    upsampled = np.maximum(upsampled, 0.0)
    upsampled = upsampled / (upsampled.sum() + 1e-9)
    times = target_x / float(target_sr)
    return times, upsampled


def _feature_extractor_from_cfg(wav2vec_cfg: Any) -> AutoFeatureExtractor:
    return AutoFeatureExtractor.from_pretrained(
        str(wav2vec_cfg.checkpoint),
        **pretrained_load_kwargs(wav2vec_cfg),
    )


def plot_wav2vec_attention_maps(
    model: nn.Module,
    audio_path: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
    output_path: str | Path,
) -> None:
    model.eval()
    feature_extractor = _feature_extractor_from_cfg(wav2vec_cfg)
    waveform = load_audio_mono(
        audio_path,
        target_sr=int(audio_cfg.target_sr),
        max_audio_sec=float(audio_cfg.max_audio_sec),
    )
    model_name = backbone_name_from_checkpoint(wav2vec_cfg)
    batch = feature_extractor(
        [waveform],
        sampling_rate=int(audio_cfg.target_sr),
        return_tensors="pt",
        padding=True,
    )
    input_values = batch["input_values"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    importance, attention_tuple = wav2vec_attention_rollout(model, input_values, attention_mask)
    importance_np = importance.detach().float().cpu().numpy()
    times, audio_importance = upsample_encoder_importance_to_audio(
        importance_np,
        len(waveform),
        target_sr=int(audio_cfg.target_sr),
    )

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    waveform_times = np.arange(len(waveform)) / float(audio_cfg.target_sr)
    axes[0].plot(waveform_times, waveform, color="gray", alpha=0.7, lw=0.5)
    axes[0].set_title(f"Waveform - {Path(audio_path).name}")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")

    axes[1].fill_between(times, 0, audio_importance, alpha=0.6, color="C0")
    axes[1].plot(times, audio_importance, color="C0", lw=1)
    axes[1].set_title(f"Attention rollout importance ({model_name})")
    axes[1].set_ylabel("Normalized mass")
    axes[1].set_xlabel("Time (s)")

    last_layer = attention_tuple[-1][0].mean(dim=0).float().cpu().numpy()
    max_show = 256
    if last_layer.shape[0] > max_show:
        step = max(1, last_layer.shape[0] // max_show)
        last_layer = last_layer[::step, ::step]
    image = axes[2].imshow(last_layer, aspect="auto", origin="lower", cmap="magma")
    axes[2].set_title("Last-layer self-attention")
    axes[2].set_xlabel("Key frame")
    axes[2].set_ylabel("Query frame")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def wav2vec_saliency(
    model: nn.Module,
    audio_path: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
    target_label: Optional[int] = None,
    window_ms: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feature_extractor = _feature_extractor_from_cfg(wav2vec_cfg)
    waveform = load_audio_mono(
        audio_path,
        target_sr=int(audio_cfg.target_sr),
        max_audio_sec=float(audio_cfg.max_audio_sec),
    )
    batch = feature_extractor(
        [waveform],
        sampling_rate=int(audio_cfg.target_sr),
        return_tensors="pt",
        padding=True,
    )
    input_values = batch["input_values"].to(device)
    input_values.requires_grad_(True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output = model(input_values=input_values, attention_mask=attention_mask)
    logits = output.logits
    if target_label is None:
        target_label = int(torch.argmax(logits, dim=-1).item())

    score = logits[0, target_label]
    model.zero_grad()
    score.backward()
    saliency = input_values.grad.detach().abs().squeeze().cpu().numpy()

    step = max(int(audio_cfg.target_sr * window_ms / 1000), 1)
    pooled = np.array([saliency[index : index + step].mean() for index in range(0, len(saliency), step)])
    times = np.arange(len(pooled)) * (window_ms / 1000.0)
    return times, pooled


def plot_wav2vec_saliency(
    model: nn.Module,
    audio_path: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
    output_path: str | Path,
    window_ms: int = 50,
) -> None:
    model_name = backbone_name_from_checkpoint(wav2vec_cfg)
    times, saliency = wav2vec_saliency(
        model=model,
        audio_path=audio_path,
        audio_cfg=audio_cfg,
        wav2vec_cfg=wav2vec_cfg,
        device=device,
        window_ms=window_ms,
    )
    plt.figure(figsize=(12, 3))
    plt.plot(times, saliency)
    plt.title(f"{model_name} saliency - {Path(audio_path).name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Saliency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
