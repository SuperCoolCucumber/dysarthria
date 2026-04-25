#!/bin/sh

#SBATCH --job-name=eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=40G
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

/home/daria.galimzianova/miniconda3/envs/lm-polygraph-env/bin/python scripts/run_tts_eval.py \
--config-path=../config \
--config-name=config.yaml