#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"
DATASET="${FAASANN_DATASET:-sift100w}"
TIMEOUT="${TIMEOUT:-650}"
# sift100w 0.99
# exec "${PYTHON}" tests/hnsw/run_queries.py \
#   --server-url "http://127.0.0.1:8080" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 100 \
#   --k 10 \
#   --candidate-k 221 \
#   --ef-search 221 \
#   --timeout "${TIMEOUT}" \
#   "$@"

# gist 0.99
exec "${PYTHON}" tests/hnsw/run_queries.py \
  --server-url "http://127.0.0.1:8080" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 30 \
  --k 10 \
  --candidate-k 550 \
  --ef-search 550 \
  --timeout "${TIMEOUT}" \
  "$@"
