#!/bin/bash
#SBATCH --job-name=c3-h1786-replay
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alyjw80/c3_lite/evaluation/c3_mem0_soft_probe1_heldout1786_v01/slurm-%j.out
#SBATCH --error=/data/alyjw80/c3_lite/evaluation/c3_mem0_soft_probe1_heldout1786_v01/slurm-%j.err

set -euo pipefail

C3_REPO=/home/$USER/projects/c3_lite_system
C3_ENV=/home/$USER/projects/c3_gpu

SCRIPT="$C3_REPO/scripts/mem0/run_soft_probe1_heldout_controller_replay.py"
RUNNER="$C3_REPO/scripts/mem0/run_hybrid_controller_pilot200.py"
PROBE_MODULE="$C3_REPO/scripts/mem0/run_soft_route_probe_budget_pilot200.py"
QWEN_RUNNER="$C3_REPO/scripts/mem0/run_qwen_hybrid_route_upperbound.py"

CONFIG="$C3_REPO/configs/c3_lite_v2_2_final.yaml"
PROMPT="$C3_REPO/prompts/answer_prompt.txt"

LOCOMO_ROOT=/data/$USER/c3_lite/external/locomo/adapter_v02
HELDOUT_ROOT="$LOCOMO_ROOT/heldout_from_pilot_v01"

QUESTIONS="$HELDOUT_ROOT/eval_questions_locomo_heldout1786.csv"
MEMORIES="$LOCOMO_ROOT/memories_locomo.json"
C3_PREDICTIONS="$HELDOUT_ROOT/c3_predictions_heldout1786.jsonl"

MEM0_TOP20=/data/$USER/c3_lite/external/mem0/heldout1786_v01/source_equiv_v02/retrieval_top20_source_equiv.jsonl

OUTPUT=/data/$USER/c3_lite/evaluation/c3_mem0_soft_probe1_heldout1786_v01

source "$C3_ENV/env.sh"
source "$C3_ENV/.venv/bin/activate"

cd "$C3_REPO"
export PYTHONPATH="$C3_REPO:$C3_REPO/src:${PYTHONPATH:-}"

echo "===== B13 FORMAL CONTROLLER REPLAY ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Started: $(date -Is)"
echo "Host: $(hostname)"
echo "Branch: $(git branch --show-current)"
echo "Script SHA: $(sha256sum "$SCRIPT")"

echo
echo "Existing progress:"
wc -l "$OUTPUT/predictions.jsonl" 2>/dev/null || true

python "$SCRIPT" \
  --runner-module "$RUNNER" \
  --probe-module "$PROBE_MODULE" \
  --qwen-runner-module "$QWEN_RUNNER" \
  --config "$CONFIG" \
  --prompt-template "$PROMPT" \
  --questions "$QUESTIONS" \
  --memories "$MEMORIES" \
  --c3-predictions "$C3_PREDICTIONS" \
  --mem0-top20 "$MEM0_TOP20" \
  --output-dir "$OUTPUT" \
  --expected-questions 1786 \
  --bootstrap-repetitions 50000 \
  --bootstrap-seed 20260803 \
  --evaluation-time "2026-08-03T00:00:00+00:00"

RC=$?

echo
echo "B13_CONTROLLER_RC=$RC"
echo "Finished: $(date -Is)"

exit "$RC"
