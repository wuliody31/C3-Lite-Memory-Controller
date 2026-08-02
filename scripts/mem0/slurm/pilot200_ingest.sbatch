#!/bin/bash
#SBATCH --job-name=mem0-p200-ingest
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/alyjw80/projects/mem0_baseline/logs/pilot200_ingest_%j.out
#SBATCH --error=/home/alyjw80/projects/mem0_baseline/logs/pilot200_ingest_%j.err

set -euo pipefail

MEM0_ROOT=/home/alyjw80/projects/mem0_baseline
C3_REPO=/home/alyjw80/projects/c3_lite_system
LOCOMO_ADAPTER=/data/alyjw80/c3_lite/external/locomo/adapter_v02

PILOT_SCRIPT="$C3_REPO/scripts/mem0/run_locomo_mem0_pilot200.py"
PILOT_QUESTIONS="$LOCOMO_ADAPTER/eval_questions_locomo_pilot200.csv"
PILOT_MEMORIES="$LOCOMO_ADAPTER/memories_locomo.json"
MEM0_PILOT_ROOT=/data/alyjw80/c3_lite/external/mem0/pilot200_v01

source "$MEM0_ROOT/.venv/bin/activate"
source "$MEM0_ROOT/configs/runtime_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

cd "$C3_REPO"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: $(which python)"
python --version

python "$PILOT_SCRIPT" \
  --phase ingest \
  --memories "$PILOT_MEMORIES" \
  --questions "$PILOT_QUESTIONS" \
  --output-dir "$MEM0_PILOT_ROOT"

echo "Finished: $(date --iso-8601=seconds)"
