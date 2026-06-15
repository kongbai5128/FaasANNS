#!/usr/bin/env bash
# Build an Alibaba Cloud Function Compute upload package for two_stage_search.
#
# Build dependencies inside a Python 3.10 buster container, matching the
# ann_candidate_search Function Compute package style.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FUNC_DIR="$ROOT_DIR/baseline/functions/Two_stage_search/src"
BUILD_DIR="$ROOT_DIR/dist/two_stage_search"
ZIP_PATH="$ROOT_DIR/dist/two_stage_search.zip"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.10-slim-buster}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Build this package in a Python 3.10 buster container." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not running. Start Docker Desktop/docker service, then rerun this script." >&2
  exit 1
fi

rm -rf "$BUILD_DIR" "$ZIP_PATH"
mkdir -p "$BUILD_DIR"

cp "$FUNC_DIR"/app.py "$BUILD_DIR"/
cp "$FUNC_DIR"/handler.py "$BUILD_DIR"/
cp "$FUNC_DIR"/index_loader.py "$BUILD_DIR"/
cp "$FUNC_DIR"/requirements.txt "$BUILD_DIR"/

docker run --rm \
  -e HOST_UID="$HOST_UID" \
  -e HOST_GID="$HOST_GID" \
  -v "$BUILD_DIR:/build" \
  -w /build \
  "$DOCKER_IMAGE" \
  bash -lc 'printf "%s\n" \
    "deb http://archive.debian.org/debian buster main" \
    "deb http://archive.debian.org/debian-security buster/updates main" \
    "deb http://archive.debian.org/debian buster-updates main" > /etc/apt/sources.list \
    && printf "Acquire::Check-Valid-Until false;\n" > /etc/apt/apt.conf.d/99no-check-valid-until \
    && apt-get update \
    && python -m pip install --upgrade --no-cache-dir --target /build/python -r /build/requirements.txt \
    && chown -R "${HOST_UID}:${HOST_GID}" /build'

(
  cd "$BUILD_DIR"
  zip -qr "$ZIP_PATH" .
)

echo "Built: $ZIP_PATH"
echo "Upload this ZIP to Function Compute, or upload folder: $BUILD_DIR"
