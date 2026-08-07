#!/bin/bash
#SBATCH --job-name=b5-mem0-p25
#SBATCH --account=standard
#SBATCH --qos=core
#SBATCH --partition=amp20
#SBATCH -G 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/data/%u/c3_lite/evaluation/phase2b_b5_mem0_lifecycle_pilot25_v01/slurm-%j.out
#SBATCH --error=/data/%u/c3_lite/evaluation/phase2b_b5_mem0_lifecycle_pilot25_v01/slurm-%j.err

set -euo pipefail

C3_REPO=/home/$USER/projects/c3_lite_system
C3_ENV=/home/$USER/projects/c3_gpu
MEM0_ROOT=/home/$USER/projects/mem0_baseline
MEM0_VENDOR="$MEM0_ROOT/vendor/mem0-v2.0.15"

ROOT=/data/$USER/c3_lite/evaluation/phase2b_b5_mem0_lifecycle_pilot25_v01
RUN_ROOT="$ROOT/job_${SLURM_JOB_ID}"

HOST=127.0.0.1
PORT=$((30000 + SLURM_JOB_ID % 20000))
BASE_URL="http://${HOST}:${PORT}/v1"

CASES="$C3_REPO/data/lifecycle_benchmark_v01/cases.jsonl"

mkdir -p \
  "$RUN_ROOT/server" \
  "$RUN_ROOT/client" \
  "$RUN_ROOT/backend"

echo "========================================"
echo "PHASE II-B5 MEM0 25-CASE PILOT"
echo "========================================"
echo "Date: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Run root: $RUN_ROOT"

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

sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null
then
  echo "ERROR: Qwen server exited before readiness check."
  echo "===== SERVER LOG ====="
  cat "$RUN_ROOT/server/server.log" || true
  exit 1
fi

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

if ! kill -0 "$SERVER_PID" 2>/dev/null
then
  echo "ERROR: Health endpoint responded, but this job's Qwen server is not alive."
  echo "Possible port collision or foreign server."
  echo "===== SERVER LOG ====="
  cat "$RUN_ROOT/server/server.log" || true
  exit 1
fi

echo
echo "===== SERVER READY ====="
echo "Server PID: $SERVER_PID"
echo "Server port: $PORT"
nvidia-smi

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
  "$C3_REPO/scripts/mem0/phase2b/run_mem0_lifecycle_b5_pilot25.py" \
  --base-url "$BASE_URL" \
  --cases "$CASES" \
  --output-dir "$RUN_ROOT/client" \
  --backend-dir "$RUN_ROOT/backend"

echo
echo "===== CLIENT SUMMARY ====="
cat "$RUN_ROOT/client/summary.json"

echo
echo "===== RESULT COUNT ====="
wc -l "$RUN_ROOT/client/results.jsonl"

echo
echo "===== SERVER REQUEST COUNT ====="
wc -l "$RUN_ROOT/server/requests.jsonl"

echo
echo "===== SERVER LOG TAIL ====="
tail -100 "$RUN_ROOT/server/server.log"

cleanup
trap - EXIT

echo
echo "PHASE II-B5 MEM0 25-CASE PILOT JOB: PASSED"
