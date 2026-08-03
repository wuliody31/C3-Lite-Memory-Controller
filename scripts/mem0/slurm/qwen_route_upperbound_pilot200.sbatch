#!/bin/bash
#SBATCH --job-name=c3-mem0-b8
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/data/alyjw80/c3_lite/evaluation/c3_mem0_qwen_route_upperbound_pilot200_v01/slurm-%j.out
#SBATCH --error=/data/alyjw80/c3_lite/evaluation/c3_mem0_qwen_route_upperbound_pilot200_v01/slurm-%j.err

set -uo pipefail

C3_REPO="/home/alyjw80/projects/c3_lite_system"
C3_ENV="/home/alyjw80/projects/c3_gpu"

source "$C3_ENV/env.sh"
source "$C3_ENV/.venv/bin/activate"

cd "$C3_REPO"

export PYTHONPATH="$C3_REPO:$C3_REPO/src:${PYTHONPATH:-}"

echo "===== B8 ENVIRONMENT ====="
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date -Is)"
python --version
nvidia-smi

echo
echo "Existing rows before resume:"
wc -l "/data/alyjw80/c3_lite/evaluation/c3_mem0_qwen_route_upperbound_pilot200_v01/predictions.jsonl"

python "/home/alyjw80/projects/c3_lite_system/scripts/mem0/run_qwen_hybrid_route_upperbound.py"   --runner-module "/home/alyjw80/projects/c3_lite_system/scripts/mem0/run_hybrid_controller_pilot200.py"   --config "/home/alyjw80/projects/c3_lite_system/configs/c3_lite_v2_2_final.yaml"   --prompt-template "/home/alyjw80/projects/c3_lite_system/prompts/answer_prompt.txt"   --questions "/data/alyjw80/c3_lite/external/locomo/adapter_v02/eval_questions_locomo_pilot200.csv"   --memories "/data/alyjw80/c3_lite/external/locomo/adapter_v02/memories_locomo.json"   --c3-predictions "/data/alyjw80/c3_lite/evaluation/frozen/locomo_pilot200_qwen3_job36163/predictions.jsonl"   --mem0-top20 "/data/alyjw80/c3_lite/external/mem0/pilot200_v01/source_equiv_v02/retrieval_top20_source_equiv.jsonl"   --output-dir "/data/alyjw80/c3_lite/evaluation/c3_mem0_qwen_route_upperbound_pilot200_v01"   --bootstrap-repetitions 50000   --bootstrap-seed 20260803

RC=$?

echo
echo "B8_RUN_RC=$RC"
echo "End: $(date -Is)"

exit $RC
