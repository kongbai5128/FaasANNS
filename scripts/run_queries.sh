#!/usr/bin/env bash
set -euo pipefail

DATASET="gist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"

TIMEOUT="${TIMEOUT:-650}"
PLAN_FILE="workload-generator/plan.bin"
P99_LOG_DIR="${P99_LOG_DIR:-logs/P99/7_24}"
P99_WINDOW_SECONDS="${P99_WINDOW_SECONDS:-5}"

# ISLOCAL="_local"
ISLOCAL=""
# In plan mode this is only the maximum client worker capacity. Actual arrivals
# and time-varying concurrency come from PLAN_FILE timestamps.
CLIENT_WORKERS=1000
# sift100w 0.99 221     0.95 99
# exec "${PYTHON}" tests/hnsw/run_queries.py \
#   --server-url "http://127.0.0.1:8080" \
#   --dataset "${DATASET}" \
#   --query-num 0 \
#   --concurrent-requests "${CLIENT_WORKERS}" \
#   --k 10 \
#   --candidate-k 221 \
#   --ef-search 221 \
#   --timeout "${TIMEOUT}" \
#   --plan-file "${PLAN_FILE}" \
#   --log-file "logs/run${ISLOCAL}_queries_${DATASET}.csv" \
#   "$@"

# gist 0.99 551         0.95 130
mkdir -p "${P99_LOG_DIR}"
exec "${PYTHON}" tests/hnsw/run_queries.py \
  --server-url "http://127.0.0.1:8080" \
  --dataset "${DATASET}" \
  --query-num 0 \
  --concurrent-requests "${CLIENT_WORKERS}" \
  --k 10 \
  --candidate-k 130 \
  --ef-search 130 \
  --timeout "${TIMEOUT}" \
  --plan-file "${PLAN_FILE}" \
  --log-file "logs/run${ISLOCAL}_queries_${DATASET}.csv" \
  --p99-window-seconds "${P99_WINDOW_SECONDS}" \
  --p99-log-file "${P99_LOG_DIR}/faasann_${DATASET}_p99_${P99_WINDOW_SECONDS}s.csv" \
  --latency-trace-file "${P99_LOG_DIR}/faasann_${DATASET}_query_trace.csv" \
  "$@"
