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
# sift100w 0.99 221     0.95 99
# exec "${PYTHON}" tests/hnsw/run_queries.py \
#   --server-url "http://127.0.0.1:8080" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 1 \
#   --k 10 \
#   --candidate-k 221 \
#   --ef-search 221 \
#   --timeout "${TIMEOUT}" \
#   --log-file "logs/run_local_queries_${DATASET}.csv" \
#   "$@"

# gist 0.99 551         0.95 130
exec "${PYTHON}" tests/hnsw/run_queries.py \
  --server-url "http://127.0.0.1:8080" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 30 \
  --k 10 \
  --candidate-k 130 \
  --ef-search 130 \
  --timeout "${TIMEOUT}" \
  --log-file "logs/run_local_queries_${DATASET}.csv" \
  "$@"
