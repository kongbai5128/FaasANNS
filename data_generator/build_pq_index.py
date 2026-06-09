"""生成云函数第一阶段 PQ 候选索引。

输出文件与 functions/ann_candidate_search/index_loader.py 约定一致：

  pq_meta.json
  pq_codebooks.npy
  pq_codes.npy
  pq_ids.npy

生成完整 PQ 索引：

python data_generator/build_pq_index.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/index/full/pq \
  --subspaces 16 \
  --codebook-size 256 \
  --train-size 100000 \
  --iterations 25 \
  --seed 0 \
  --batch-size 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path, max_vectors: int | None) -> np.ndarray:
    file_path = Path(path)
    raw = np.fromfile(file_path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {file_path}")

    dim = int(raw[0])
    record_width = dim + 1
    if raw.size % record_width != 0:
        raise ValueError(f"{file_path} does not look like fvecs with dimension {dim}")

    records = raw.reshape(-1, record_width)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"dimension mismatch in {file_path}: expected {dim}")

    vectors = records[:, 1:].view(np.float32)
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    return np.ascontiguousarray(vectors)


def build_pq(
    vectors: np.ndarray,
    *,
    subspace_count: int,
    codebook_size: int,
    train_size: int,
    iterations: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[1] % subspace_count != 0:
        raise ValueError(f"dimension={vectors.shape[1]} is not divisible by subspace_count={subspace_count}")
    if codebook_size <= 0 or codebook_size > 256:
        raise ValueError("codebook_size must be in [1, 256]")
    if train_size <= 0 or iterations <= 0 or batch_size <= 0:
        raise ValueError("train_size, iterations, and batch_size must be positive")

    rng = np.random.default_rng(seed)
    train_count = min(train_size, vectors.shape[0])
    train_ids = rng.choice(vectors.shape[0], size=train_count, replace=False)
    train = vectors[train_ids]
    subvector_dim = vectors.shape[1] // subspace_count

    codebooks = np.empty((subspace_count, codebook_size, subvector_dim), dtype=np.float32)
    codes = np.empty((vectors.shape[0], subspace_count), dtype=np.uint8)

    for part in range(subspace_count):
        start = part * subvector_dim
        end = start + subvector_dim
        print(f"Training PQ subspace {part + 1}/{subspace_count}: dims=[{start}, {end})", flush=True)
        centroids = kmeans(train[:, start:end], codebook_size, iterations, rng)
        codebooks[part] = centroids
        codes[:, part] = assign_codes_batched(vectors[:, start:end], centroids, batch_size)

    return codebooks, codes


def kmeans(data: np.ndarray, cluster_count: int, iterations: int, rng: np.random.Generator) -> np.ndarray:
    if data.shape[0] < cluster_count:
        raise ValueError(f"not enough training vectors: train={data.shape[0]}, codebook_size={cluster_count}")

    centroid_ids = rng.choice(data.shape[0], size=cluster_count, replace=False)
    centroids = np.ascontiguousarray(data[centroid_ids].copy())
    labels = np.zeros(data.shape[0], dtype=np.int64)

    for _ in range(iterations):
        labels = assign_codes(data, centroids).astype(np.int64)
        for cluster_id in range(cluster_count):
            members = data[labels == cluster_id]
            if members.size == 0:
                centroids[cluster_id] = data[int(rng.integers(0, data.shape[0]))]
            else:
                centroids[cluster_id] = members.mean(axis=0)
    return np.ascontiguousarray(centroids.astype("float32"))


def assign_codes(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = l2_distance_matrix(data, centroids)
    return np.argmin(distances, axis=1).astype(np.uint8)


def assign_codes_batched(data: np.ndarray, centroids: np.ndarray, batch_size: int) -> np.ndarray:
    codes = np.empty(data.shape[0], dtype=np.uint8)
    for start in range(0, data.shape[0], batch_size):
        end = min(start + batch_size, data.shape[0])
        codes[start:end] = assign_codes(data[start:end], centroids)
    return codes


def l2_distance_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    data_norm = np.sum(data * data, axis=1, keepdims=True)
    centroid_norm = np.sum(centroids * centroids, axis=1).reshape(1, -1)
    distances = data_norm + centroid_norm - 2.0 * data @ centroids.T
    return np.maximum(distances, 0.0)


def save_pq_index(args: argparse.Namespace) -> None:
    vectors = read_fvecs(args.src, max_vectors=args.max_vectors)
    codebooks, codes = build_pq(
        vectors,
        subspace_count=args.subspaces,
        codebook_size=args.codebook_size,
        train_size=args.train_size,
        iterations=args.iterations,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.arange(vectors.shape[0], dtype=np.int64)
    np.save(output_dir / "pq_codebooks.npy", codebooks)
    np.save(output_dir / "pq_codes.npy", codes)
    np.save(output_dir / "pq_ids.npy", ids)

    meta = {
        "source": str(Path(args.src).resolve()),
        "dimension": int(vectors.shape[1]),
        "vector_count": int(vectors.shape[0]),
        "subspace_count": int(args.subspaces),
        "codebook_size": int(args.codebook_size),
        "train_size": int(min(args.train_size, vectors.shape[0])),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
    }
    (output_dir / "pq_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved PQ index to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PQ candidate index for FaasANN cloud functions")
    parser.add_argument("--src", required=True, help="input .fvecs file")
    parser.add_argument("--dst", required=True, help="output PQ index directory")
    parser.add_argument("--max-vectors", type=int, default=None, help="optional number of base vectors to encode")
    parser.add_argument("--subspaces", type=int, required=True, help="number of PQ subspaces")
    parser.add_argument("--codebook-size", type=int, required=True, help="centroids per subspace, max 256")
    parser.add_argument("--train-size", type=int, required=True, help="number of vectors used to train PQ")
    parser.add_argument("--iterations", type=int, required=True, help="k-means iterations per subspace")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument("--batch-size", type=int, required=True, help="vectors encoded per batch")
    return parser.parse_args()


if __name__ == "__main__":
    save_pq_index(parse_args())
