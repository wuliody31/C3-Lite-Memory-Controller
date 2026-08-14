#!/bin/bash
#SBATCH --job-name=c3v3-final-smoke
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/alyjw80/projects/c3_gpu/logs/c3v3-final-smoke-%j.out
#SBATCH --error=/home/alyjw80/projects/c3_gpu/logs/c3v3-final-smoke-%j.err

set -euo pipefail

REPO=/home/alyjw80/projects/c3_v3_final
ENV=/home/alyjw80/projects/c3_gpu/.venv

# IMPORTANT:
# These old pairs are used ONLY for GPU harness smoke.
# They are NOT the final RC8.3-vs-C3-v3 paper comparison.
PAIRS=/home/alyjw80/projects/c3_lite_system/outputs/c3_v3_counterfactual_full60/paired_prompts.jsonl

OUTPUT=/data/alyjw80/c3_lite/evaluation/c3_v3_final_qwen_smoke_${SLURM_JOB_ID}

source /home/alyjw80/projects/c3_gpu/env.sh
source "$ENV/bin/activate"

export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

export HF_HOME=/data/alyjw80/c3_lite/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"

mkdir -p "$TMPDIR"
mkdir -p "$OUTPUT"

cd "$REPO"

echo "============================================================"
echo "C3-v3 FINAL QWEN3-8B GPU SMOKE"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Harness HEAD:  $(git rev-parse HEAD)"
echo "Method commit: $(git rev-parse c3-v3-active-final-candidate)"
echo "Output:        $OUTPUT"
echo "Started:       $(date)"
echo "============================================================"

echo
echo "===== GPU ====="

nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

echo
echo "===== PYTORCH ====="

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable inside allocated GPU job."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "===== CONFIG ====="

grep -nE \
    "temperature|max_new_tokens|transformers_" \
    configs/c3_v3_final_qwen.yaml

echo
echo "===== RUNNER SMOKE: 3 QUESTIONS ====="

python \
    scripts/run_c3_v3_counterfactual_generation.py \
    --pairs "$PAIRS" \
    --config configs/c3_v3_final_qwen.yaml \
    --output-dir "$OUTPUT" \
    --max-questions 3

echo
echo "===== OUTPUT FILES ====="

find "$OUTPUT" \
    -maxdepth 2 \
    -type f \
    -printf '%p\n' \
    | sort

echo
echo "===== OUTPUT SIZES ====="

du -h "$OUTPUT"/* 2>/dev/null || true

echo
echo "============================================================"
echo "SMOKE FINISHED SUCCESSFULLY"
echo "Finished: $(date)"
echo "Output: $OUTPUT"
echo "============================================================"
