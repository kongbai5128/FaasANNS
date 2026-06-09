#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" tests/hnsw/run_queries.py \
  --server-url "http://127.0.0.1:8080" \
  --query-file "data/sift100w/sift_query.fvecs" \
  --groundtruth-file "data/sift100w/sift_groundtruth.ivecs" \
  --log-file "logs/run_queries.csv" \
  --query-num 1000 \
  --concurrent-requests 100 \
  --k 10 \
  --candidate-k 120 \
  --ef-search 80 \
  "$@"
