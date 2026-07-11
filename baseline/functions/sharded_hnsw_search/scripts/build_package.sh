#!/usr/bin/env bash
# Build one Function Compute upload package for all sharded_hnsw_search shards.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FUNC_DIR="$ROOT_DIR/baseline/functions/sharded_hnsw_search/src"
DIST_DIR="$ROOT_DIR/dist"
BUILD_DIR="$DIST_DIR/baseline_sharded_hnsw_search"
ZIP_PATH="$DIST_DIR/baseline_sharded_hnsw_search.zip"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.10-slim-buster}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. faiss-cpu must be packaged in a Python 3.10 buster container." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not running. Start Docker/docker service, then rerun this script." >&2
  exit 1
fi

rm -rf "$BUILD_DIR"
rm -f "$ZIP_PATH"
rm -rf "$DIST_DIR"/baseline_sharded_hnsw_search_base
rm -rf "$DIST_DIR"/baseline_sharded_hnsw_search_shard*
rm -f "$DIST_DIR"/baseline_sharded_hnsw_search_shard*.zip
mkdir -p "$BUILD_DIR"

cp "$FUNC_DIR"/app.py "$BUILD_DIR"/
cp "$FUNC_DIR"/handler.py "$BUILD_DIR"/
cp "$FUNC_DIR"/index_loader.py "$BUILD_DIR"/
cp "$FUNC_DIR"/requirements.txt "$BUILD_DIR"/
cat > "$BUILD_DIR/runtime.env.example" <<'EOF'
FAASANN_DATA_ROOT=/mnt/faasann
FAASANN_DATASET=gist
FAASANN_SHARDING_DIR=/mnt/faasann/gist/sharding_8
FAASANN_PEER_ENDPOINTS=http://sharding-fazozlktgc.cn-hongkong.fcapp.run,http://sharding-faznzlktgc.cn-hongkong.fcapp.run,http://sharding-fazmzlktgc.cn-hongkong.fcapp.run,http://sharding-faztzlktgc.cn-hongkong.fcapp.run,http://sharding-fazszlktgc.cn-hongkong.fcapp.run,http://sharding-fazrzlktgc.cn-hongkong.fcapp.run,http://sharding-fazqzlktgc.cn-hongkong.fcapp.run
EOF

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

rm -rf "$BUILD_DIR"

echo "Built: $ZIP_PATH"
echo "Upload this same zip to all shard functions. Use startup command: python3 app.py <dataset> <shard_id>"
