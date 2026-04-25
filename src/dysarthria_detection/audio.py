from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal)) + 1e-9
    return signal / peak


def _load_with_wave_stdlib(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def _load_with_torchaudio(path: str) -> tuple[np.ndarray, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    audio = waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return audio, int(sample_rate)


def load_audio_mono(
    path: str | Path,
    target_sr: int,
    max_audio_sec: float,
) -> np.ndarray:
    audio_path = str(path)
    file_size = Path(audio_path).stat().st_size
    if file_size < 128:
        raise ValueError(f"Audio file too small ({file_size} bytes): {audio_path}")

    audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    errors: list[str] = []

    try:
        audio, sample_rate = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")

    if audio is None:
        try:
            audio, sample_rate = _load_with_wave_stdlib(audio_path)
        except Exception as exc:
            errors.append(f"wave: {exc}")

    if audio is None:
        try:
            audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            audio = audio.astype(np.float32)
            sample_rate = int(sample_rate)
        except Exception as exc:
            errors.append(f"librosa: {exc}")

    if audio is None:
        try:
            audio, sample_rate = _load_with_torchaudio(audio_path)
        except Exception as exc:
            errors.append(f"torchaudio: {exc}")

    if audio is None or sample_rate is None:
        raise RuntimeError(f"Could not decode audio {audio_path}. Tried: {'; '.join(errors)}")

    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)

    audio = normalize_audio(audio.astype(np.float32))
    max_audio_len = int(target_sr * max_audio_sec)
    if len(audio) >= max_audio_len:
        return audio[:max_audio_len]

    padding = max_audio_len - len(audio)
    return np.pad(audio, (0, padding), mode="constant")


def drop_unreadable_audio_rows(
    df: pd.DataFrame,
    target_sr: int,
    max_audio_sec: float,
    desc: str = "Verify audio",
) -> pd.DataFrame:
    keep_indices: list[int] = []
    unreadable = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            load_audio_mono(row["audio_path"], target_sr=target_sr, max_audio_sec=max_audio_sec)
            keep_indices.append(index)
        except Exception:
            unreadable += 1

    if unreadable:
        print(f"[drop_unreadable_audio_rows] Removed {unreadable} unreadable rows ({desc})")
    return df.loc[keep_indices].reset_index(drop=True)
