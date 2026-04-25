from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


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


def _find_best_wav_root(
    base: Path,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
) -> Path | None:
    if not base.exists():
        return None

    candidates = [base]
    candidates.extend(path for path in sorted(base.rglob("*")) if path.is_dir())

    best_match: Path | None = None
    best_count = 0
    for directory in candidates:
        normalized = _normalize_name(directory.name)
        if include_keywords and not any(keyword in normalized for keyword in include_keywords):
            continue
        if exclude_keywords and any(keyword in normalized for keyword in exclude_keywords):
            continue
        wav_count = _count_wavs_quick(directory, cap=100000)
        if wav_count > best_count:
            best_match = directory
            best_count = wav_count
    return best_match if best_count > 0 else None


def _has_wavs(root: str | Path | None) -> bool:
    if not root:
        return False
    path = Path(str(root))
    return path.exists() and _count_wavs_quick(path, cap=1) > 0


def _ensure_kaggle_credentials(download_cfg: Any) -> None:
    credentials_json = str(download_cfg.credentials_json).strip() if download_cfg.credentials_json else ""
    if not credentials_json:
        if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            return
        default_json = Path.home() / ".kaggle" / "kaggle.json"
        if default_json.exists():
            return
        raise RuntimeError(
            "Kaggle download is enabled but no credentials were found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or provide data.download.credentials_json."
        )
        return

    source = Path(credentials_json).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Kaggle credentials file does not exist: {source}")

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "username" not in payload or "key" not in payload:
        raise ValueError("Kaggle credentials JSON must contain 'username' and 'key'.")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "kaggle.json"
    shutil.copyfile(source, target)
    target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _resolve_kaggle_command() -> list[str]:
    python_bin = Path(sys.executable).resolve()
    sibling = python_bin.parent / "kaggle"
    if sibling.exists():
        return [str(sibling)]

    found = shutil.which("kaggle")
    if found:
        return [found]

    return [sys.executable, "-m", "kaggle.cli"]


def _run_kaggle_download(slug: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] Downloading {slug} -> {destination}")
    kaggle_command = _resolve_kaggle_command()
    subprocess.check_call(
        [
            *kaggle_command,
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(destination),
            "--unzip",
        ],
        env={**os.environ},
    )


def _discover_kaggle_roots(download_root: Path) -> dict[str, str | None]:
    torgo_root = _find_best_wav_root(download_root, ("torgo",))
    ua_root = _find_best_wav_root(
        download_root,
        ("uaspeech", "ua_speech", "noisereduced"),
    )
    ua_control_root = _find_best_wav_root(
        download_root,
        ("control",),
    )
    return {
        "torgo_root": str(torgo_root) if torgo_root is not None else None,
        "ua_root": str(ua_root) if ua_root is not None else None,
        "ua_control_root": str(ua_control_root) if ua_control_root is not None else None,
    }


def prepare_kaggle_data(data_cfg: Any) -> dict[str, str | None]:
    download_cfg = data_cfg.download
    if not bool(download_cfg.enabled):
        return {}

    download_root = Path(str(download_cfg.root_dir))
    if not download_root.is_absolute():
        download_root = (Path.cwd() / download_root).resolve()

    _ensure_kaggle_credentials(download_cfg)

    discovered = _discover_kaggle_roots(download_root)
    needs_torgo = bool(download_cfg.force) or not discovered.get("torgo_root")
    needs_ua = bool(download_cfg.force) or not discovered.get("ua_root")

    if needs_torgo:
        _run_kaggle_download(str(download_cfg.torgo_slug), download_root)
    else:
        print(f"[kaggle] TORGO already present at {discovered['torgo_root']}")

    if needs_ua:
        _run_kaggle_download(str(download_cfg.ua_slug), download_root)
    else:
        print(f"[kaggle] UA Speech already present at {discovered['ua_root']}")

    discovered = _discover_kaggle_roots(download_root)
    if not discovered.get("torgo_root"):
        raise RuntimeError(f"Could not discover TORGO .wav files under {download_root}")
    if not discovered.get("ua_root"):
        raise RuntimeError(f"Could not discover UA Speech .wav files under {download_root}")

    data_cfg.torgo_root = discovered["torgo_root"]
    data_cfg.ua_root = discovered["ua_root"]
    if discovered.get("ua_control_root"):
        data_cfg.ua_control_root = discovered["ua_control_root"]

    print(f"[kaggle] TORGO root -> {data_cfg.torgo_root}")
    print(f"[kaggle] UA root    -> {data_cfg.ua_root}")
    if getattr(data_cfg, "ua_control_root", None):
        print(f"[kaggle] UA control -> {data_cfg.ua_control_root}")
    return discovered


def main() -> None:
    from .cli import _load_config

    cfg = _load_config(sys.argv[1:])
    prepare_kaggle_data(cfg.data)
