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
exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
  --endpoint "https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run" \
  --query-file "data/sift100w/sift_query.fvecs" \
  --groundtruth-file "data/sift100w/sift_groundtruth.ivecs" \
  --log-file "baseline/functions/Two_stage_search/test/result/run_queries.csv" \
  --query-num 1000 \
  --concurrent-requests 100 \
  --k 10 \
  --candidate-k 221 \
  --ef-search 221 \
  "$@"
