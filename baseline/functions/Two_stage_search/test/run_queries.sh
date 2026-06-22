#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
ENDPOINT="http://127.0.0.1:9000"
# ENDPOINT="https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"
DATASET="${FAASANN_DATASET:-sift100w}"
# sift100w 0.99
exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 1 \
  --k 10 \
  --candidate-k 221 \
  --ef-search 221 \
  --log-file "baseline/functions/Two_stage_search/test/run_local_queries_${DATASET}.log" \
  "$@"

# gist 0.99
# exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
#   --endpoint "${ENDPOINT}" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 1 \
#   --k 10 \
#   --candidate-k 550 \
#   --ef-search 550 \
#   --log-file "baseline/functions/Two_stage_search/test/result/run_local_queries_${DATASET}.csv" \
#   "$@"
