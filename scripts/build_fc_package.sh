#!/usr/bin/env bash
# Build an Alibaba Cloud Function Compute upload package for ann_candidate_search.
#
# Alibaba FC custom runtimes do not install requirements.txt automatically when a
# folder is uploaded from the console. This script vendors numpy into a python/
# directory inside the function package and creates a ZIP that can be uploaded directly.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUNC_DIR="$ROOT_DIR/functions/ann_candidate_search"
BUILD_DIR="$ROOT_DIR/dist/ann_candidate_search"
ZIP_PATH="$ROOT_DIR/dist/ann_candidate_search.zip"
USE_DOCKER="${USE_DOCKER:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.10-slim-buster}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if [[ "$USE_DOCKER" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Run with USE_DOCKER=0 only if your local Python matches FC Python 3.10." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker daemon is not running. Start Docker Desktop/docker service, then rerun this script." >&2
    echo "Run with USE_DOCKER=0 only if your local Python is Python 3.10 on Linux x86_64." >&2
    exit 1
  fi
  mkdir -p "$ROOT_DIR/dist"
  docker run --rm \
    -v "$ROOT_DIR/dist:/dist" \
    "$DOCKER_IMAGE" \
    bash -lc 'rm -rf /dist/ann_candidate_search /dist/ann_candidate_search.zip'
else
  rm -rf "$BUILD_DIR" "$ZIP_PATH"
fi

mkdir -p "$BUILD_DIR"

cp "$FUNC_DIR"/app.py "$BUILD_DIR"/
cp "$FUNC_DIR"/handler.py "$BUILD_DIR"/
cp "$FUNC_DIR"/index_loader.py "$BUILD_DIR"/
cp "$FUNC_DIR"/requirements.txt "$BUILD_DIR"/
cp "$FUNC_DIR"/README.md "$BUILD_DIR"/

if [[ "$USE_DOCKER" == "1" ]]; then
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
else
  "$PYTHON_BIN" -m pip install \
    --upgrade \
    --target "$BUILD_DIR/python" \
    -r "$FUNC_DIR/requirements.txt"
fi

(
  cd "$BUILD_DIR"
  zip -qr "$ZIP_PATH" .
)

echo "Built: $ZIP_PATH"
echo "Upload this ZIP to Function Compute, or upload folder: $BUILD_DIR"
