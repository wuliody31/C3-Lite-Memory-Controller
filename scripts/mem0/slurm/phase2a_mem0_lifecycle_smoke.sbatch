#!/bin/bash
#SBATCH --job-name=p2a-mem0-life
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH -G 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/data/%u/c3_lite/evaluation/phase2a_mem0_lifecycle_smoke_v01/slurm-%j.out
#SBATCH --error=/data/%u/c3_lite/evaluation/phase2a_mem0_lifecycle_smoke_v01/slurm-%j.err

set -euo pipefail

C3_REPO=/home/$USER/projects/c3_lite_system
C3_ENV=/home/$USER/projects/c3_gpu
MEM0_ROOT=/home/$USER/projects/mem0_baseline
MEM0_VENDOR="$MEM0_ROOT/vendor/mem0-v2.0.15"

ROOT=/data/$USER/c3_lite/evaluation/phase2a_mem0_lifecycle_smoke_v01
RUN_ROOT="$ROOT/job_${SLURM_JOB_ID}"

HOST=127.0.0.1
PORT=18080
BASE_URL="http://${HOST}:${PORT}/v1"

mkdir -p \
  "$RUN_ROOT/server" \
  "$RUN_ROOT/client" \
  "$RUN_ROOT/backend"

echo "========================================"
echo "PHASE II-A MEM0 LIFECYCLE SMOKE"
echo "========================================"
echo "Date: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}"
echo "Run root: $RUN_ROOT"

###############################################################################
# Start local Qwen3 server
###############################################################################

source "$C3_ENV/env.sh"
source "$C3_ENV/.venv/bin/activate"

export PYTHONPATH="$C3_REPO:$C3_REPO/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u \
  "$C3_REPO/scripts/mem0/phase2a/qwen_openai_server.py" \
  --model "Qwen/Qwen3-8B" \
  --host "$HOST" \
  --port "$PORT" \
  --request-log "$RUN_ROOT/server/requests.jsonl" \
  > "$RUN_ROOT/server/server.log" \
  2>&1 &

SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null
  then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

python \
  "$C3_REPO/scripts/mem0/phase2a/wait_for_local_llm.py" \
  --url "http://${HOST}:${PORT}/health" \
  --timeout 180

echo
echo "===== SERVER READY ====="
nvidia-smi

###############################################################################
# Run Mem0 client in its frozen environment
###############################################################################

deactivate 2>/dev/null || true

source "$MEM0_ROOT/.venv/bin/activate"

if [ -f "$MEM0_ROOT/configs/runtime_env.sh" ]
then
  source "$MEM0_ROOT/configs/runtime_env.sh"
fi

export PYTHONPATH="$MEM0_VENDOR:$C3_REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$MEM0_VENDOR"

python \
  "$C3_REPO/scripts/mem0/phase2a/mem0_lifecycle_smoke.py" \
  --base-url "$BASE_URL" \
  --output-dir "$RUN_ROOT/client" \
  --backend-dir "$RUN_ROOT/backend"

echo
echo "===== CLIENT SUMMARY ====="
cat "$RUN_ROOT/client/summary.json"

echo
echo "===== SERVER REQUEST COUNT ====="
wc -l "$RUN_ROOT/server/requests.jsonl"

echo
echo "===== SERVER LOG TAIL ====="
tail -100 "$RUN_ROOT/server/server.log"

cleanup
trap - EXIT

echo
echo "PHASE II-A MEM0 LIFECYCLE JOB: PASSED"
