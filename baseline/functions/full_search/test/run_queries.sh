#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"
DATASET="${FAASANN_DATASET:-sift100w}"
# sift100w 0.99
# exec "${PYTHON}" baseline/functions/full_search/test/run_queries.py \
#   --endpoint "http://base-lil-search-mbidkbhcit.cn-hongkong.fcapp.run" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 100 \
#   --k 10 \
#   --candidate-k 10 \
#   --ef-search 99 \
#   "$@"


# gist 0.99
exec "${PYTHON}" baseline/functions/full_search/test/run_queries.py \
  --endpoint "http://base-lil-search-mbidkbhcit.cn-hongkong.fcapp.run" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 100 \
  --k 10 \
  --candidate-k 10 \
  --ef-search 250 \
  "$@"