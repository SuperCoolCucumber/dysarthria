# Automatic Dysarthria Detection and Severity Classification

This repository provides a runnable project for local development and Slurm execution. The pipeline is organized into Python modules under `src/dysarthria_detection/`, Hydra-style YAML configs under `config/`, and Slurm entrypoints under `slurm/`.

## Layout

```text
.
├── config/
├── scripts/
├── slurm/
└── src/dysarthria_detection/
```

Core modules:

- `audio.py`: robust WAV loading, resampling, padding, unreadable-file filtering
- `downloads.py`: Kaggle dataset download and root discovery
- `metadata.py`: TORGO / UA Speech metadata construction and label inference
- `splits.py`: speaker-level train/test splitting and severity coverage handling
- `baseline.py`: MFCC + SVM baseline
- `wav2vec.py`: wav2vec2 training and evaluation
- `interpretability.py`: permutation importance, saliency, and attention rollout plots
- `pipeline.py`: end-to-end orchestration and artifact writing

## Environment setup with uv

The project was initialized with `uv` and is intended to run inside a project-local `.venv`.

```bash
export UV_CACHE_DIR=/tmp/uv-cache
uv sync
```

If you want to reproduce the environment from scratch after cloning:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
uv sync
```

Notes:

- The code targets Python `3.12`.
- `torch` / `torchaudio` wheel selection can be cluster-specific. If your Slurm cluster requires a CUDA-specific wheel set, install the matching PyTorch build inside `.venv` before training.
- wav2vec2 weights come from Hugging Face by default. On a restricted cluster, pre-populate the cache and then run with `wav2vec.local_files_only=true`.
- The repository supports direct Kaggle download and dataset root discovery.

## Configuration

The default entry config is [`config/config.yaml`](/home/daria.galimzianova/dysarthria/config/config.yaml:1). Override values from the CLI in Hydra style:

```bash
.venv/bin/python scripts/run_pipeline.py \
  --config-path=../config \
  --config-name=config.yaml \
  data.torgo_root=/path/to/TORGO \
  data.ua_root=/path/to/UA_Speech \
  outputs.run_name=debug_run
```

Useful overrides:

- `data.metadata_csv=/path/to/metadata.csv`
- `data.max_samples_per_dataset=512`
- `data.download.enabled=true`
- `data.download.root_dir=/path/to/download_root`
- `wav2vec.epochs_binary=1`
- `wav2vec.epochs_severity=1`
- `wav2vec.local_files_only=true`
- `runtime.device=cpu`

## Kaggle download workflow

The repository can download the following Kaggle datasets directly:

- TORGO: `pranaykoppula/torgo-audio`
- UA Speech: `aryashah2k/noise-reduced-uaspeech-dysarthria-dataset`

Authentication:

- Preferred: set `KAGGLE_USERNAME` and `KAGGLE_KEY`
- Alternative: point `data.download.credentials_json` at a `kaggle.json` file
- Existing `~/.kaggle/kaggle.json` also works

Download only:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...

.venv/bin/python scripts/download_kaggle_datasets.py \
  --config-path=../config \
  --config-name=config.yaml \
  data.download.enabled=true \
  data.download.root_dir=data/kaggle
```

Run the full pipeline with automatic download/discovery:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...

.venv/bin/python scripts/run_pipeline.py \
  --config-path=../config \
  --config-name=config.yaml \
  data.download.enabled=true \
  data.download.root_dir=data/kaggle \
  outputs.run_name=kaggle_run
```

## Expected data inputs

You can run the project in one of two ways:

1. Provide `data.metadata_csv` with columns:
   `audio_path`, `dataset`, `binary_label`, `severity_label`, `speaker_id`
2. Point `data.torgo_root` and `data.ua_root` at extracted dataset directories and let the project infer labels from paths

The path-based inference is heuristic. If you care about research-quality severity labels, use curated metadata.

## Outputs

Each run creates an artifact directory under `outputs/` containing:

- `metadata/` CSV snapshots
- `models/` saved SVM and wav2vec2 checkpoints
- `reports/` classification reports and final markdown report
- `figures/` optional plots
- `tables/` model comparison CSV
- `training_logs.json`

## Slurm

Two ready-to-edit Slurm scripts are provided:

- [`slurm/full_pipeline.slurm`](/home/daria.galimzianova/dysarthria/slurm/full_pipeline.slurm:1)
- [`slurm/smoke_test.slurm`](/home/daria.galimzianova/dysarthria/slurm/smoke_test.slurm:1)

Before submitting, set `TORGO_ROOT` and `UA_ROOT` in the job environment or edit the defaults in the script.

If you want the job to download from Kaggle directly, set:

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
sbatch --export=KAGGLE_DOWNLOAD=true,DATA_ROOT=/path/to/shared/kaggle_cache slurm/full_pipeline.slurm
```
