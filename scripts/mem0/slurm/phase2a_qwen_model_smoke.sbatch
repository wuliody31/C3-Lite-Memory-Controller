#!/bin/bash
#SBATCH --job-name=p2a-qwen-smoke
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/%u/c3_lite/evaluation/phase2a_qwen_model_smoke_v01/slurm-%j.out
#SBATCH --error=/data/%u/c3_lite/evaluation/phase2a_qwen_model_smoke_v01/slurm-%j.err

set -euo pipefail

C3_REPO=/home/$USER/projects/c3_lite_system
C3_ENV=/home/$USER/projects/c3_gpu
SMOKE_ROOT=/data/$USER/c3_lite/evaluation/phase2a_qwen_model_smoke_v01

source "$C3_ENV/env.sh"
source "$C3_ENV/.venv/bin/activate"

export PYTHONPATH="$C3_REPO:$C3_REPO/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$SMOKE_ROOT"

echo "Date: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-unknown}"

nvidia-smi

python \
  "$C3_REPO/scripts/mem0/phase2a/qwen_model_smoke.py"
