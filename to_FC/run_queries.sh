#!/usr/bin/env bash
set -euo pipefail

DATASET="gist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

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

cd "${PROJECT_ROOT}"
export no_proxy="*"
export NO_PROXY="*"

TIMEOUT="${TIMEOUT:-650}"
FC_ACCEPT_TIMEOUT="${FC_ACCEPT_TIMEOUT:-${TIMEOUT}}"
CLIENT_WORKERS="${CLIENT_WORKERS:-1000}"
PLAN_FILE="workload-generator/plan.bin"

# local
# FUNCTION_URL="http://127.0.0.1:9000"

# aliyun FC
# FUNCTION_URL="https://to-fc-ncbfswtjlz.cn-hongkong.fcapp.run"
# 内网
FUNCTION_URL="https://to-fc-ncbfswtjlz.cn-hongkong-vpc.fcapp.run"

# VM rerank server. 如果云函数访问 VM，这里改成 VM 公网地址。
# RERANK_SERVER_URL="http://127.0.0.1:8081"
RERANK_SERVER_URL="http://47.239.188.123:8081"

# VM rerank 完成后会把最终结果 POST 回这里。跨机器测试时改成 client 可被 VM 访问的地址。
CALLBACK_ADVERTISE_HOST="127.0.0.1"
CALLBACK_PORT="18080"

# sift100w 0.95 99
# exec "${PYTHON}" to_FC/test/run_queries.py \
#   --function-url "${FUNCTION_URL}" \
#   --rerank-server-url "${RERANK_SERVER_URL}" \
#   --callback-advertise-host "${CALLBACK_ADVERTISE_HOST}" \
#   --callback-port "${CALLBACK_PORT}" \
#   --dataset "${DATASET}" \
#   --query-num 0 \
#   --concurrent-requests "${CLIENT_WORKERS}" \
#   --k 10 \
#   --candidate-k 99 \
#   --ef-search 99 \
#   --fc-accept-timeout "${FC_ACCEPT_TIMEOUT}" \
#   --result-timeout "${TIMEOUT}" \
#   --plan-file "${PLAN_FILE}" \
#   --log-file "logs/run_queries_to_FC_${DATASET}.csv" \
#   "$@"

# gist 0.95 130
exec "${PYTHON}" to_FC/test/run_queries.py \
  --function-url "${FUNCTION_URL}" \
  --rerank-server-url "${RERANK_SERVER_URL}" \
  --callback-advertise-host "${CALLBACK_ADVERTISE_HOST}" \
  --callback-port "${CALLBACK_PORT}" \
  --dataset "${DATASET}" \
  --query-num 0 \
  --concurrent-requests "${CLIENT_WORKERS}" \
  --k 10 \
  --candidate-k 130 \
  --ef-search 130 \
  --fc-accept-timeout "${FC_ACCEPT_TIMEOUT}" \
  --result-timeout "${TIMEOUT}" \
  --plan-file "${PLAN_FILE}" \
  --log-file "logs/run_queries_to_FC_${DATASET}.csv" \
  "$@"
