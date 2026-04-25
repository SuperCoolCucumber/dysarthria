from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_linear_schedule_with_warmup

from .audio import drop_unreadable_audio_rows, load_audio_mono
from .evaluation import cross_dataset_split
from .metrics import metric_bundle


def pretrained_load_kwargs(wav2vec_cfg: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": bool(wav2vec_cfg.local_files_only),
    }
    if wav2vec_cfg.cache_dir:
        kwargs["cache_dir"] = str(wav2vec_cfg.cache_dir)
    return kwargs


def autocast_dtype(wav2vec_cfg: Any) -> torch.dtype:
    amp_dtype = str(getattr(wav2vec_cfg, "amp_dtype", "fp16")).lower()
    if amp_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


class DysarthriaAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, audio_cfg: Any):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        waveform = load_audio_mono(
            row["audio_path"],
            target_sr=int(self.audio_cfg.target_sr),
            max_audio_sec=float(self.audio_cfg.max_audio_sec),
        )
        return {
            "input_values": waveform,
            "labels": int(row[self.label_col]),
        }


@dataclass
class Collator:
    feature_extractor: AutoFeatureExtractor
    target_sr: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [feature["input_values"] for feature in features]
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch = self.feature_extractor(
            inputs,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        batch["labels"] = labels
        return batch


def make_loader(
    df: pd.DataFrame,
    label_col: str,
    feature_extractor: AutoFeatureExtractor,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
    shuffle: bool = True,
    batch_size: int | None = None,
) -> DataLoader:
    dataset = DysarthriaAudioDataset(df, label_col=label_col, audio_cfg=audio_cfg)
    num_workers = int(wav2vec_cfg.num_workers)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size or wav2vec_cfg.batch_size),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(wav2vec_cfg, "pin_memory", True)) and device == "cuda",
        "collate_fn": Collator(feature_extractor=feature_extractor, target_sr=int(audio_cfg.target_sr)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(wav2vec_cfg, "prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(getattr(wav2vec_cfg, "persistent_workers", False))
    return DataLoader(**kwargs)


def build_feature_extractor(wav2vec_cfg: Any) -> AutoFeatureExtractor:
    return AutoFeatureExtractor.from_pretrained(
        str(wav2vec_cfg.checkpoint),
        **pretrained_load_kwargs(wav2vec_cfg),
    )


def backbone_name_from_checkpoint(wav2vec_cfg: Any) -> str:
    checkpoint = str(wav2vec_cfg.checkpoint).rstrip("/")
    return checkpoint.split("/")[-1]


def _freeze_encoder_layers(model: nn.Module, wav2vec_cfg: Any) -> None:
    num_layers = int(wav2vec_cfg.freeze_encoder_layers)
    if num_layers <= 0:
        return

    base_model_prefix = getattr(model, "base_model_prefix", "")
    base_model = getattr(model, base_model_prefix, None)
    if base_model is None:
        return

    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return

    for layer_index, layer in enumerate(layers):
        if layer_index >= num_layers:
            break
        for parameter in layer.parameters():
            parameter.requires_grad = False


def build_wav2vec2_model(wav2vec_cfg: Any, num_labels: int) -> nn.Module:
    load_kwargs: dict[str, Any] = {
        **pretrained_load_kwargs(wav2vec_cfg),
        "num_labels": num_labels,
        "problem_type": "single_label_classification",
        "attention_dropout": float(wav2vec_cfg.attention_dropout),
        "hidden_dropout": float(wav2vec_cfg.hidden_dropout),
        "feat_proj_dropout": float(wav2vec_cfg.feat_proj_dropout),
        "ignore_mismatched_sizes": True,
    }
    attn_impl = str(wav2vec_cfg.attn_implementation)
    if attn_impl.lower() not in {"", "none", "auto"}:
        load_kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForAudioClassification.from_pretrained(
            str(wav2vec_cfg.checkpoint),
            **load_kwargs,
        )
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForAudioClassification.from_pretrained(
            str(wav2vec_cfg.checkpoint),
            **load_kwargs,
        )

    if bool(wav2vec_cfg.freeze_feature_encoder):
        freeze_feature_encoder = getattr(model, "freeze_feature_encoder", None)
        if callable(freeze_feature_encoder):
            freeze_feature_encoder()
    _freeze_encoder_layers(model, wav2vec_cfg)
    return model


def resolve_learning_rate(wav2vec_cfg: Any) -> float:
    if bool(wav2vec_cfg.auto_scale_lr_with_batch):
        return float(wav2vec_cfg.lr_base) * (
            float(wav2vec_cfg.batch_size) / float(wav2vec_cfg.lr_ref_batch_size)
        )
    return float(wav2vec_cfg.lr)


def class_weights_from_labels(labels: np.ndarray, device: str) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    total = counts.sum()
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * count)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_task(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str,
    num_labels: int,
    epochs: int,
    average_mode: str,
    run_name: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
 ) -> tuple[nn.Module, dict[str, Any]]:
    feature_extractor = build_feature_extractor(wav2vec_cfg)
    model = build_wav2vec2_model(wav2vec_cfg, num_labels=num_labels).to(device)
    backbone_name = backbone_name_from_checkpoint(wav2vec_cfg)

    train_loader = make_loader(train_df, label_col, feature_extractor, audio_cfg, wav2vec_cfg, device, shuffle=True)
    valid_loader = make_loader(
        valid_df,
        label_col,
        feature_extractor,
        audio_cfg,
        wav2vec_cfg,
        device,
        shuffle=False,
        batch_size=int(getattr(wav2vec_cfg, "eval_batch_size", wav2vec_cfg.batch_size)),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=resolve_learning_rate(wav2vec_cfg),
        weight_decay=float(wav2vec_cfg.weight_decay),
    )
    accum_steps = max(1, int(wav2vec_cfg.gradient_accumulation_steps))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * float(wav2vec_cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(train_df[label_col].to_numpy(), device=device))
    use_amp = bool(wav2vec_cfg.use_amp) and device == "cuda"
    amp_dtype = autocast_dtype(wav2vec_cfg)
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler) if use_grad_scaler else None

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"[{run_name}] train {epoch}/{epochs}")):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    output = model(input_values=input_values, attention_mask=attention_mask)
                    loss = criterion(output.logits.float(), labels) / accum_steps
                if use_grad_scaler:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                output = model(input_values=input_values, attention_mask=attention_mask)
                loss = criterion(output.logits, labels) / accum_steps
                loss.backward()

            train_loss += float(loss.detach().cpu()) * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if use_grad_scaler:
                    assert scaler is not None
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[{run_name}] valid {epoch}/{epochs}"):
                input_values = batch["input_values"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)

                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        output = model(input_values=input_values, attention_mask=attention_mask)
                        loss = criterion(output.logits.float(), labels)
                else:
                    output = model(input_values=input_values, attention_mask=attention_mask)
                    loss = criterion(output.logits, labels)

                val_loss += float(loss.detach().cpu())
                preds = torch.argmax(output.logits.float(), dim=-1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        metrics = metric_bundle(y_true, y_pred, average=average_mode)
        epoch_log = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(valid_loader)),
            **metrics,
        }
        history.append(epoch_log)
        print(
            f"[{run_name}:{backbone_name}] epoch={epoch} "
            f"train_loss={epoch_log['train_loss']:.4f} "
            f"val_loss={epoch_log['val_loss']:.4f} "
            f"acc={epoch_log['accuracy']:.4f} "
            f"f1={epoch_log['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"history": history, "best_f1": best_f1, "backbone": backbone_name}


def evaluate_wav2vec2(
    model: nn.Module,
    df: pd.DataFrame,
    label_col: str,
    average_mode: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
) -> dict[str, Any]:
    feature_extractor = build_feature_extractor(wav2vec_cfg)
    loader = make_loader(
        df,
        label_col,
        feature_extractor,
        audio_cfg,
        wav2vec_cfg,
        device,
        shuffle=False,
        batch_size=int(getattr(wav2vec_cfg, "eval_batch_size", wav2vec_cfg.batch_size)),
    )
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    use_amp = bool(wav2vec_cfg.use_amp) and device == "cuda"
    amp_dtype = autocast_dtype(wav2vec_cfg)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {label_col}"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    output = model(input_values=input_values, attention_mask=attention_mask)
            else:
                output = model(input_values=input_values, attention_mask=attention_mask)

            preds = torch.argmax(output.logits.float(), dim=-1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return {
        "metrics": metric_bundle(y_true, y_pred, average=average_mode),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_cross_eval_wav2vec(
    df: pd.DataFrame,
    train_ds: str,
    test_ds: str,
    label_col: str,
    num_labels: int,
    average_mode: str,
    epochs: int,
    run_name: str,
    audio_cfg: Any,
    wav2vec_cfg: Any,
    device: str,
) -> dict[str, Any]:
    train_df, test_df = cross_dataset_split(df, train_ds, test_ds)
    clean_train_df = drop_unreadable_audio_rows(
        train_df,
        target_sr=int(audio_cfg.target_sr),
        max_audio_sec=float(audio_cfg.max_audio_sec),
        desc=f"cross w2v train {train_ds}",
    )
    clean_test_df = drop_unreadable_audio_rows(
        test_df,
        target_sr=int(audio_cfg.target_sr),
        max_audio_sec=float(audio_cfg.max_audio_sec),
        desc=f"cross w2v test {test_ds}",
    )
    if clean_train_df.empty or clean_test_df.empty:
        return {"metrics": {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}}

    model, _ = train_one_task(
        train_df=clean_train_df,
        valid_df=clean_test_df,
        label_col=label_col,
        num_labels=num_labels,
        epochs=epochs,
        average_mode=average_mode,
        run_name=run_name,
        audio_cfg=audio_cfg,
        wav2vec_cfg=wav2vec_cfg,
        device=device,
    )
    return evaluate_wav2vec2(
        model=model,
        df=clean_test_df,
        label_col=label_col,
        average_mode=average_mode,
        audio_cfg=audio_cfg,
        wav2vec_cfg=wav2vec_cfg,
        device=device,
    )
