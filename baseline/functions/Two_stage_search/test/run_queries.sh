#!/usr/bin/env bash
set -euo pipefail

DATASET="gist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
# ENDPOINT="http://127.0.0.1:9000"
ENDPOINT="https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"

# sift100w 0.99 221     0.95 99
# exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
#   --endpoint "${ENDPOINT}" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 1 \
#   --k 10 \
#   --candidate-k 99 \
#   --ef-search 99 \
#   --log-file "baseline/functions/Two_stage_search/test/result/run_local_queries_${DATASET}.csv" \
#   "$@"

# gist 0.99 550     0.95 130
exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 1 \
  --k 10 \
  --candidate-k 130 \
  --ef-search 130 \
  --log-file "baseline/functions/Two_stage_search/test/result/run_local_queries_${DATASET}.csv" \
  "$@"
