#!/usr/bin/env bash
set -euo pipefail

DATASET="gist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
# ENDPOINT="http://127.0.0.1:9000"
# ISLOCAL="_local"
ISLOCAL=""
# 内网
ENDPOINT="https://base-lie-search-aobkfnfjxd.cn-hongkong-vpc.fcapp.run"
# 公网
# ENDPOINT="https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run"

PLAN_FILE="workload-generator/plan.bin"
TIMEOUT="${TIMEOUT:-650}"
CLIENT_WORKERS="${CLIENT_WORKERS:-1000}"
P99_LOG_DIR="${P99_LOG_DIR:-baseline/functions/Two_stage_search/test/result/P99/7_24}"
P99_WINDOW_SECONDS="${P99_WINDOW_SECONDS:-5}"
TWO_STAGE_VARIANT="${TWO_STAGE_VARIANT:-memory}"
METHOD_LABEL="${METHOD_LABEL:-two_stage_${TWO_STAGE_VARIANT}}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"
mkdir -p "${P99_LOG_DIR}"

# sift100w 0.99 221     0.95 99
# exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
#   --endpoint "${ENDPOINT}" \
#   --dataset "${DATASET}" \
#   --query-num 0 \
#   --concurrent-requests "${CLIENT_WORKERS}" \
#   --k 10 \
#   --candidate-k 99 \
#   --ef-search 99 \
#   --timeout "${TIMEOUT}" \
#   --plan-file "${PLAN_FILE}" \
#   --log-file "baseline/functions/Two_stage_search/test/result/run${ISLOCAL}_queries_${DATASET}.csv" \
#   "$@"

# gist 0.99 550     0.95 130
exec "${PYTHON}" "baseline/functions/Two_stage_search/test/run_queries.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${DATASET}" \
  --query-num 0 \
  --concurrent-requests "${CLIENT_WORKERS}" \
  --k 10 \
  --candidate-k 130 \
  --ef-search 130 \
  --timeout "${TIMEOUT}" \
  --plan-file "${PLAN_FILE}" \
  --log-file "baseline/functions/Two_stage_search/test/result/run${ISLOCAL}_queries_${DATASET}.csv" \
  --method-label "${METHOD_LABEL}" \
  --p99-window-seconds "${P99_WINDOW_SECONDS}" \
  --p99-log-file "${P99_LOG_DIR}/${METHOD_LABEL}_${DATASET}_p99_${P99_WINDOW_SECONDS}s.csv" \
  --latency-trace-file "${P99_LOG_DIR}/${METHOD_LABEL}_${DATASET}_query_trace.csv" \
  "$@"
