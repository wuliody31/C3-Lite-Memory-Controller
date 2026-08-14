#!/bin/bash
#SBATCH --job-name=locomo-c3-final
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/alyjw80/projects/c3_gpu/logs/locomo-c3-final-%A_%a.out
#SBATCH --error=/home/alyjw80/projects/c3_gpu/logs/locomo-c3-final-%A_%a.err

set -euo pipefail

REPO=/home/alyjw80/projects/c3_v3_final
ENV=/home/alyjw80/projects/c3_gpu/.venv
ROOT=/data/alyjw80/c3_lite/evaluation/c3_v3_locomo_final_20260814
LOCOMO=/data/alyjw80/c3_lite/external/locomo/adapter_v02
RUNNER=/home/alyjw80/projects/c3_lite_system/scripts/mem0/run_hybrid_controller_pilot200.py
QWEN_HELPER=/home/alyjw80/projects/c3_lite_system/scripts/mem0/run_qwen_hybrid_route_upperbound.py

MODE="${MODE:-array}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"

source /home/alyjw80/projects/c3_gpu/env.sh
source "$ENV/bin/activate"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export HF_HOME=/data/alyjw80/c3_lite/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-locomo-final}}"
mkdir -p "$TMPDIR" /home/alyjw80/projects/c3_gpu/logs
cd "$REPO"

if [[ "$MODE" == "smoke" ]]; then
  USER_ID="${SMOKE_USER_ID:?SMOKE_USER_ID required}"
  OUTPUT="$ROOT/qwen_smoke_job${SLURM_JOB_ID}"
else
  INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"
  USER_ID="$(sed -n "$((INDEX + 1))p" "$ROOT/users.txt")"
  [[ -n "$USER_ID" ]] || { echo "No user ID for index $INDEX"; exit 2; }
  OUTPUT="$ROOT/qwen_formal_job${SLURM_ARRAY_JOB_ID}/task_${INDEX}"
fi
mkdir -p "$OUTPUT"

echo "mode=$MODE job=${SLURM_JOB_ID:-unknown} user_id=$USER_ID output=$OUTPUT"
echo "method_commit=$(git rev-parse c3-v3-active-final-candidate)"
echo "harness_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

ARGS=(
  generate
  --inputs "$ROOT/comparison_inputs.jsonl"
  --memories "$LOCOMO/memories_locomo.json"
  --runner-module "$RUNNER"
  --qwen-helper "$QWEN_HELPER"
  --config configs/c3_v3_final_qwen.yaml
  --output-dir "$OUTPUT"
  --user-id "$USER_ID"
  --max-questions "$MAX_QUESTIONS"
)
[[ "$MODE" == "smoke" ]] || ARGS+=(--skip-backbone-smoke)
python scripts/run_locomo_final_comparison.py "${ARGS[@]}"
find "$OUTPUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT/SHA256SUMS.txt"
echo "LOCOMO FINAL JOB: PASS"
