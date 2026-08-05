#!/bin/bash
#SBATCH --job-name=mem0-h1786-query
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/data/alyjw80/c3_lite/external/mem0/heldout1786_v01/slurm-%j.out
#SBATCH --error=/data/alyjw80/c3_lite/external/mem0/heldout1786_v01/slurm-%j.err

set -euo pipefail

C3_REPO=/home/$USER/projects/c3_lite_system
MEM0_ROOT=/home/$USER/projects/mem0_baseline

SCRIPT="$C3_REPO/scripts/mem0/run_locomo_mem0_heldout_query.py"
RUNNER="$C3_REPO/scripts/mem0/run_locomo_mem0_pilot200.py"

LOCOMO_ROOT=/data/$USER/c3_lite/external/locomo/adapter_v02
HELDOUT_ROOT="$LOCOMO_ROOT/heldout_from_pilot_v01"

QUESTIONS="$HELDOUT_ROOT/eval_questions_locomo_heldout1786.csv"
MEMORIES="$LOCOMO_ROOT/memories_locomo.json"

OUTPUT=/data/$USER/c3_lite/external/mem0/heldout1786_v01

source "$MEM0_ROOT/.venv/bin/activate"
source "$MEM0_ROOT/configs/runtime_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

cd "$C3_REPO"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Branch: $(git branch --show-current)"
echo "Python: $(which python)"
python --version

echo
echo "Script SHA:"
sha256sum "$SCRIPT"

echo
echo "Existing progress:"
wc -l "$OUTPUT/retrieval_top20.jsonl" 2>/dev/null || true

python "$SCRIPT" \
  --runner-module "$RUNNER" \
  --questions "$QUESTIONS" \
  --memories "$MEMORIES" \
  --output-dir "$OUTPUT" \
  --top-k 20 \
  --expected-questions 1786 \
  --dataset-role "post-development-heldout-from-pilot"

RC=$?

echo
echo "MEM0_HELDOUT_RUN_RC=$RC"
echo "Finished: $(date --iso-8601=seconds)"

exit "$RC"
