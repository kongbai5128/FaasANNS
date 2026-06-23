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


ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --dataset)
      if ((i + 1 < ${#ARGS[@]})); then
        DATASET="${ARGS[$((i + 1))]}"
      fi
      ;;
    --dataset=*)
      DATASET="${ARGS[$i]#--dataset=}"
      ;;
  esac
done

SERVER_URL="${FAASANN_SERVER_URL:-${SERVER_URL:-http://127.0.0.1:8080}}"
FUNCTION_URL="${FAASANN_FUNCTION_URL:-${FUNCTION_URL:-http://127.0.0.1:9000}}"
RERANK_SERVER_URL="${FAASANN_RERANK_SERVER_URL:-${RERANK_SERVER_URL:-${SERVER_URL}}}"
TIMEOUT="${TIMEOUT:-650}"
LOG_FILE="${LOG_FILE:-logs/run_local_queries_2FC_${DATASET}.csv}"

exec "${PYTHON}" tests/hnsw/run_queries.py \
  --entrypoint function \
  --server-url "${SERVER_URL}" \
  --function-url "${FUNCTION_URL}" \
  --rerank-server-url "${RERANK_SERVER_URL}" \
  --dataset "${DATASET}" \
  --query-num 1000 \
  --concurrent-requests 1 \
  --k 10 \
  --candidate-k 221 \
  --ef-search 221 \
  --timeout "${TIMEOUT}" \
  --log-file "${LOG_FILE}" \
  "$@"

# exec "${PYTHON}" tests/hnsw/run_queries.py \
#   --entrypoint function \
#   --server-url "${SERVER_URL}" \
#   --function-url "${FUNCTION_URL}" \
#   --rerank-server-url "${RERANK_SERVER_URL}" \
#   --dataset "${DATASET}" \
#   --query-num 1000 \
#   --concurrent-requests 1 \
#   --k 10 \
#   --candidate-k 130 \
#   --ef-search 130 \
#   --timeout "${TIMEOUT}" \
#   --log-file "${LOG_FILE}" \
#   "$@"
