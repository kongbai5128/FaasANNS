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
ENDPOINT="http://sharding-fazpzlktgc.cn-hongkong-vpc.fcapp.run"
# ENDPOINT="http://sharding-fazpzlktgc.cn-hongkong.fcapp.run"
PLAN_FILE="workload-generator/plan.bin"
TIMEOUT="${TIMEOUT:-650}"
CLIENT_WORKERS="${CLIENT_WORKERS:-1000}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"

# sift100w 0.99
# exec "${PYTHON}" baseline/functions/sharded_hnsw_search/test/run_queries.py \
#   --endpoint "${ENDPOINT}" \
#   --dataset "${DATASET}" \
#   --query-num 0 \
#   --concurrent-requests "${CLIENT_WORKERS}" \
#   --k 10 \
#   --candidate-k 10 \
#   --ef-search 99 \
#   --timeout "${TIMEOUT}" \
#   --plan-file "${PLAN_FILE}" \
#   --log-file "baseline/functions/sharded_hnsw_search/test/result/run${ISLOCAL}_queries_${DATASET}.csv" \
#   "$@"


# gist 0.99
exec "${PYTHON}" baseline/functions/sharded_hnsw_search/test/run_queries.py \
  --endpoint "${ENDPOINT}" \
  --dataset "${DATASET}" \
  --query-num 0 \
  --concurrent-requests "${CLIENT_WORKERS}" \
  --k 10 \
  --candidate-k 10 \
  --ef-search 60 \
  --timeout "${TIMEOUT}" \
  --plan-file "${PLAN_FILE}" \
  --log-file "baseline/functions/sharded_hnsw_search/test/result/run${ISLOCAL}_queries_${DATASET}.csv" \
  "$@"
