#!/bin/sh

#SBATCH --job-name=wavlm-large-bin
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=96G
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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
SITE_PACKAGES="$(echo "${PROJECT_ROOT}"/.venv/lib/python*/site-packages)"
NVIDIA_LD_LIBRARY_PATH=""
for lib_dir in \
  "${SITE_PACKAGES}/nvidia/cublas/lib" \
  "${SITE_PACKAGES}/nvidia/cuda_cupti/lib" \
  "${SITE_PACKAGES}/nvidia/cuda_nvrtc/lib" \
  "${SITE_PACKAGES}/nvidia/cuda_runtime/lib" \
  "${SITE_PACKAGES}/nvidia/cudnn/lib" \
  "${SITE_PACKAGES}/nvidia/cufft/lib" \
  "${SITE_PACKAGES}/nvidia/cufile/lib" \
  "${SITE_PACKAGES}/nvidia/curand/lib" \
  "${SITE_PACKAGES}/nvidia/cusolver/lib" \
  "${SITE_PACKAGES}/nvidia/cusparse/lib" \
  "${SITE_PACKAGES}/nvidia/nccl/lib" \
  "${SITE_PACKAGES}/nvidia/nvjitlink/lib" \
  "${SITE_PACKAGES}/nvidia/nvtx/lib"
do
  if [ -d "${lib_dir}" ]; then
    if [ -n "${NVIDIA_LD_LIBRARY_PATH}" ]; then
      NVIDIA_LD_LIBRARY_PATH="${NVIDIA_LD_LIBRARY_PATH}:${lib_dir}"
    else
      NVIDIA_LD_LIBRARY_PATH="${lib_dir}"
    fi
  fi
done
if [ -n "${NVIDIA_LD_LIBRARY_PATH}" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  export KAGGLE_USERNAME KAGGLE_KEY
fi

"${PROJECT_ROOT}/.venv/bin/python" scripts/run_pipeline.py \
  --config-path=../config \
  --config-name=wavlm_large_binary_a100.yaml \
  data.torgo_root="${TORGO_ROOT}" \
  data.ua_root="${UA_ROOT}" \
  data.download.enabled="${KAGGLE_DOWNLOAD}" \
  data.download.root_dir="${DATA_ROOT}" \
  outputs.run_name="wavlm_large_binary_a100_${SLURM_JOB_ID}"
