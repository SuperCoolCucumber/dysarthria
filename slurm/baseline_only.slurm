#!/bin/sh

#SBATCH --job-name=baseline-only
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=64G
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -eu

PROJECT_ROOT="/home/daria.galimzianova/dysarthria"
TORGO_ROOT="${TORGO_ROOT:-${PROJECT_ROOT}/data/kaggle/TORGO}"
UA_ROOT="${UA_ROOT:-${PROJECT_ROOT}/data/kaggle/UA_Speech}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/kaggle}"
KAGGLE_DOWNLOAD="${KAGGLE_DOWNLOAD:-false}"

mkdir -p "${PROJECT_ROOT}/slurm_err" "${PROJECT_ROOT}/slurm_out"
cd "${PROJECT_ROOT}"

export UV_CACHE_DIR="${PROJECT_ROOT}/.uv-cache"
export HF_HOME="${PROJECT_ROOT}/.hf-cache"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mpl-cache"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Keep credentials out of the script if possible; prefer exporting them before sbatch.
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  export KAGGLE_USERNAME KAGGLE_KEY
fi

"${PROJECT_ROOT}/.venv/bin/python" scripts/run_pipeline.py \
  --config-path=../config \
  --config-name=baseline_only.yaml \
  data.torgo_root="${TORGO_ROOT}" \
  data.ua_root="${UA_ROOT}" \
  data.download.enabled="${KAGGLE_DOWNLOAD}" \
  data.download.root_dir="${DATA_ROOT}" \
  outputs.run_name="baseline_only_${SLURM_JOB_ID}"
