#!/usr/bin/env bash
# Build an Alibaba Cloud Function Compute upload package for the to_FC
# candidate callback function.
#
# Alibaba FC custom runtimes do not install requirements.txt automatically when
# a ZIP/folder is uploaded from the console. This script vendors dependencies
# into python/ by default, matching scripts/build_fc_package.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FUNC_DIR="${SCRIPT_DIR}/function"
BUILD_DIR="${PROJECT_ROOT}/dist/to_FC_candidate_callback"
ZIP_PATH="${PROJECT_ROOT}/dist/to_FC_candidate_callback.zip"
USE_DOCKER="${USE_DOCKER:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.10-slim-buster}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if [[ "${USE_DOCKER}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Run with USE_DOCKER=0 only if local Python matches FC Python 3.10 on Linux x86_64." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker daemon is not running. Start docker, or run with USE_DOCKER=0 only if local Python matches FC Python 3.10 on Linux x86_64." >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/dist"
  docker run --rm \
    -v "${PROJECT_ROOT}/dist:/dist" \
    "${DOCKER_IMAGE}" \
    bash -lc 'rm -rf /dist/to_FC_candidate_callback /dist/to_FC_candidate_callback.zip'
else
  rm -rf "${BUILD_DIR}" "${ZIP_PATH}"
fi

mkdir -p "${BUILD_DIR}"

cp "${FUNC_DIR}/app.py" "${BUILD_DIR}/"
cp "${FUNC_DIR}/handler.py" "${BUILD_DIR}/"
cp "${FUNC_DIR}/index_loader.py" "${BUILD_DIR}/"
cp "${FUNC_DIR}/requirements.txt" "${BUILD_DIR}/"
cp "${SCRIPT_DIR}/README.md" "${BUILD_DIR}/"

if [[ "${USE_DOCKER}" == "1" ]]; then
  docker run --rm \
    -e HOST_UID="${HOST_UID}" \
    -e HOST_GID="${HOST_GID}" \
    -v "${BUILD_DIR}:/build" \
    -w /build \
    "${DOCKER_IMAGE}" \
    bash -lc 'printf "%s\n" \
      "deb http://archive.debian.org/debian buster main" \
      "deb http://archive.debian.org/debian-security buster/updates main" \
      "deb http://archive.debian.org/debian buster-updates main" > /etc/apt/sources.list \
      && printf "Acquire::Check-Valid-Until false;\n" > /etc/apt/apt.conf.d/99no-check-valid-until \
      && apt-get update \
      && python -m pip install --upgrade --no-cache-dir --target /build/python -r /build/requirements.txt \
      && chown -R "${HOST_UID}:${HOST_GID}" /build'
else
  "${PYTHON_BIN}" -m pip install \
    --upgrade \
    --target "${BUILD_DIR}/python" \
    -r "${FUNC_DIR}/requirements.txt"
fi

(cd "${BUILD_DIR}" && zip -qr "${ZIP_PATH}" .)
echo "Built: ${ZIP_PATH}"
echo "Upload this ZIP to Function Compute, or upload folder: ${BUILD_DIR}"
