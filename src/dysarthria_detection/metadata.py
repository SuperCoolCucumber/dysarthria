from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


SEVERITY_KEYWORDS = {
    "very_low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def infer_speaker_id_from_path(path: str) -> str:
    file_path = Path(path)
    parts = [part for part in file_path.parts if part]
    for token in reversed(parts):
        if re.fullmatch(r"(?:C?[FM]\d{2}|[FM]C\d{2})", token):
            return token

    joined = "_".join(parts)
    match = re.search(r"(?:^|_)([FM]C?\d{2}|C[FM]\d{2})(?:S\d{2})?(?:_|$)", joined)
    if match:
        return match.group(1)
    return "unknown"


def infer_severity_from_path(
    path: str,
    ua_speaker_severity_map: dict[str, int],
    speaker_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> int:
    lowered = path.lower()
    if "very_low" in lowered:
        return SEVERITY_KEYWORDS["very_low"]
    if " low" in lowered or "_low" in lowered:
        return SEVERITY_KEYWORDS["low"]
    if "medium" in lowered or "moderate" in lowered or "_mid" in lowered:
        return SEVERITY_KEYWORDS["medium"]
    if "high" in lowered:
        return SEVERITY_KEYWORDS["high"]

    if dataset_name == "UA":
        raw_speaker = speaker_id or infer_speaker_id_from_path(path)
        if re.fullmatch(r"(?:C[FM]\d{2}|[FM]C\d{2})", raw_speaker):
            return -1
        speaker = raw_speaker.replace("C", "")
        return int(ua_speaker_severity_map.get(speaker, -1))

    return -1


def infer_binary_label_from_path(path: str) -> int:
    file_path = Path(path)
    parts_lower = [part.lower() for part in file_path.parts]
    name_lower = file_path.name.lower()

    if any(token in parts_lower for token in ["f_con", "m_con", "noisereduced-uaspeech-control"]):
        return 0
    if any(token in parts_lower for token in ["f_dys", "m_dys", "noisereduced-uaspeech"]):
        return 1

    speaker = infer_speaker_id_from_path(path).upper()
    if re.fullmatch(r"(?:CM|CF)\d{2}", speaker):
        return 0
    if re.fullmatch(r"(?:M|F)\d{2}", speaker):
        return 1
    if re.fullmatch(r"(?:MC|FC)\d{2}", speaker):
        return 0

    full_lower = "/".join(parts_lower) + "/" + name_lower
    if any(token in full_lower for token in ["control", "healthy", "non_dys", "nondys"]):
        return 0
    if any(token in full_lower for token in ["dys", "dysarth", "patient"]):
        return 1
    return 1


def collect_audio_files(root: str | Path) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return [
        str(path)
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]


def _dataset_roots(data_cfg: Any, dataset_name: str) -> list[str]:
    if dataset_name == "TORGO":
        return [str(data_cfg.torgo_root)]

    roots = [str(data_cfg.ua_root)]
    ua_control_root = str(data_cfg.ua_control_root).strip() if getattr(data_cfg, "ua_control_root", None) else ""
    if ua_control_root:
        roots.append(ua_control_root)
    return roots


def diagnose_dataset_paths(data_cfg: Any) -> None:
    print("\nDataset path diagnostics")
    print("------------------------")
    for dataset_name, root_key in [("TORGO", "torgo_root"), ("UA", "ua_root")]:
        root = Path(str(getattr(data_cfg, root_key)))
        exists = root.exists()
        wav_count = len(collect_audio_files(root)) if exists else 0
        print(f"{dataset_name:>6} | root={root} | exists={exists} | wav_files={wav_count}")
        if exists and wav_count == 0:
            children = list(root.iterdir())[:8]
            preview = [child.name for child in children]
            print(f"      preview children: {preview if preview else '[]'}")
    if getattr(data_cfg, "ua_control_root", None):
        control_root = Path(str(data_cfg.ua_control_root))
        control_count = len(collect_audio_files(control_root)) if control_root.exists() else 0
        print(
            f"UA_CONTROL | root={control_root} | exists={control_root.exists()} | wav_files={control_count}"
        )


def _walk_dirs_bfs(root: Path, max_depth: int):
    queue: list[tuple[Path, int]] = [(root.resolve(), 0)]
    while queue:
        current, depth = queue.pop(0)
        yield current, depth
        if depth >= max_depth:
            continue
        try:
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    queue.append((child, depth + 1))
        except (PermissionError, OSError):
            continue


def _count_wavs_quick(root: Path, cap: int = 2000) -> int:
    count = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".wav":
                count += 1
                if count >= cap:
                    return count
    except OSError:
        return count
    return count


def _discover_best_folder(search_roots: list[str], keywords: tuple[str, ...], max_depth: int) -> Optional[Path]:
    best_match: Optional[Path] = None
    best_count = 0
    for search_root in search_roots:
        base = Path(search_root)
        if not base.exists():
            continue
        for directory, _ in _walk_dirs_bfs(base, max_depth=max_depth):
            normalized = directory.name.lower().replace("-", "_").replace(" ", "_")
            if not any(keyword in normalized for keyword in keywords):
                continue
            count = _count_wavs_quick(directory, cap=1500)
            if count > best_count:
                best_count = count
                best_match = directory
    return best_match if best_count > 0 else None


def autoconfigure_dataset_paths(data_cfg: Any) -> None:
    if not bool(data_cfg.auto_discover_dataset_paths):
        return

    search_roots = [str(root) for root in (data_cfg.dataset_search_roots or [])]
    max_depth = int(data_cfg.dataset_search_max_depth)

    def needs_fix(field_name: str) -> bool:
        current = Path(str(getattr(data_cfg, field_name)))
        return (not current.exists()) or (_count_wavs_quick(current, cap=3) == 0)

    if needs_fix("torgo_root"):
        found = _discover_best_folder(search_roots, ("torgo",), max_depth)
        if found is not None:
            data_cfg.torgo_root = str(found)
            print(f"[auto-discover] TORGO -> {found}")

    if needs_fix("ua_root"):
        found = _discover_best_folder(search_roots, ("ua_speech", "uaspeech", "u_a_speech"), max_depth)
        if found is not None:
            data_cfg.ua_root = str(found)
            print(f"[auto-discover] UA Speech -> {found}")


def build_metadata(data_cfg: Any, seed: int) -> pd.DataFrame:
    metadata_csv = str(data_cfg.metadata_csv) if data_cfg.metadata_csv else ""
    if metadata_csv and Path(metadata_csv).exists():
        df = pd.read_csv(metadata_csv)
        required_cols = {"audio_path", "dataset", "binary_label", "severity_label", "speaker_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {missing}")
        return df.copy()

    autoconfigure_dataset_paths(data_cfg)

    rows: list[dict[str, Any]] = []
    for dataset_name in ["TORGO", "UA"]:
        dataset_roots = _dataset_roots(data_cfg, dataset_name)
        files = sorted({audio_path for root in dataset_roots for audio_path in collect_audio_files(root)})
        max_samples = data_cfg.max_samples_per_dataset
        if max_samples is not None and len(files) > int(max_samples):
            rng = np.random.RandomState(seed)
            files = rng.choice(files, size=int(max_samples), replace=False).tolist()
            print(f"[build_metadata] {dataset_name}: sampled down to {len(files)} wav files")

        print(f"[build_metadata] {dataset_name}: found {len(files)} wav files at {dataset_roots}")
        for audio_path in files:
            speaker_id = infer_speaker_id_from_path(audio_path)
            rows.append(
                {
                    "audio_path": audio_path,
                    "dataset": dataset_name,
                    "binary_label": int(infer_binary_label_from_path(audio_path)),
                    "severity_label": int(
                        infer_severity_from_path(
                            audio_path,
                            ua_speaker_severity_map=dict(data_cfg.ua_speaker_severity_map),
                            speaker_id=speaker_id,
                            dataset_name=dataset_name,
                        )
                    ),
                    "speaker_id": speaker_id,
                }
            )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        diagnose_dataset_paths(data_cfg)
        raise RuntimeError(
            "No audio files were discovered. Set data.torgo_root and data.ua_root to extracted "
            "dataset directories or provide data.metadata_csv."
        )
    return metadata
