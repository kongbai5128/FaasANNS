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
exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
  --endpoint "https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 100 \
  --k 10 \
  --candidate-k 221 \
  --ef-search 221 \
  "$@"

# gist 0.99
# exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
#   --endpoint "https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 100 \
#   --k 10 \
#   --candidate-k 550 \
#   --ef-search 550 \
#   "$@"
