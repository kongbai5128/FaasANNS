"""Build a Faiss HNSW-PQ index for function-side candidate search.

Output file:
  faiss_hnswpq.index

python data_generator/hnsw/build_pq_index.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/index/full/pq \
  --subspaces 16 \
  --codebook-size 256 \
  --train-size 1000000 \
  --iterations 25 \
  --seed 0 \
  --hnsw-space l2 \
  --hnsw-m 32 \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 160 \
  --hnsw-batch-size 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


INDEX_NAME = "faiss_hnswpq.index"


def read_fvecs(path: str | Path, max_vectors: int | None) -> np.ndarray:
    raw = np.fromfile(Path(path), dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {path}")
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"dimension mismatch in fvecs file: {path}")
    vectors = rows[:, 1:].view(np.float32)
    return np.ascontiguousarray(vectors[:max_vectors] if max_vectors is not None else vectors)


def build_hnswpq_index(
    vectors: np.ndarray,
    *,
    subspaces: int,
    codebook_size: int,
    train_size: int,
    iterations: int,
    seed: int,
    hnsw_batch_size: int,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    metric: str,
):
    if vectors.ndim != 2 or vectors.shape[0] <= 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if vectors.shape[1] % subspaces != 0:
        raise ValueError(f"dimension={vectors.shape[1]} is not divisible by subspaces={subspaces}")
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two")
    if metric != "l2":
        raise ValueError("Faiss HNSW-PQ currently supports hnsw-space=l2 in this generator")
    if hnsw_batch_size <= 0:
        raise ValueError("hnsw_batch_size must be positive")

    nbits = codebook_size.bit_length() - 1
    spec = f"HNSW{hnsw_m},PQ{subspaces}x{nbits}"
    index = faiss.index_factory(vectors.shape[1], spec, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    rng = np.random.default_rng(seed)
    train_count = min(train_size, vectors.shape[0])
    train = np.ascontiguousarray(vectors[rng.choice(vectors.shape[0], size=train_count, replace=False)])
    if hasattr(index, "pq"):
        index.pq.cp.niter = iterations
        index.pq.cp.seed = seed

    print(f"Training {spec}: vectors={train_count}", flush=True)
    index.train(train)
    for start in range(0, vectors.shape[0], hnsw_batch_size):
        end = min(start + hnsw_batch_size, vectors.shape[0])
        index.add(np.ascontiguousarray(vectors[start:end]))
        print(f"Added vectors {end}/{vectors.shape[0]}", flush=True)
    return index


def save_index(args: argparse.Namespace) -> None:
    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    for path in [
        out / INDEX_NAME,
        out / "pq_hnsw.bin",
        out / "pq_hnsw.meta.json",
        out / "pq_meta.json",
        out / "pq_codebooks.npy",
        out / "pq_codes.npy",
        out / "pq_ids.npy",
    ]:
        if path.is_file():
            path.unlink()

    vectors = read_fvecs(args.src, args.max_vectors)
    index = build_hnswpq_index(
        vectors,
        subspaces=args.subspaces,
        codebook_size=args.codebook_size,
        train_size=args.train_size,
        iterations=args.iterations,
        seed=args.seed,
        hnsw_batch_size=args.hnsw_batch_size,
        hnsw_m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
        metric=args.hnsw_space,
    )
    faiss.write_index(index, str(out / INDEX_NAME))
    print(f"Saved {out / INDEX_NAME}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Faiss HNSW-PQ index for FaasANN cloud functions")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--max-vectors", type=int)
    parser.add_argument("--subspaces", type=int, required=True)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hnsw-space", required=True, choices=["l2"])
    parser.add_argument("--hnsw-m", type=int, required=True)
    parser.add_argument("--hnsw-ef-construction", type=int, required=True)
    parser.add_argument("--hnsw-ef-search", type=int, required=True)
    parser.add_argument("--hnsw-batch-size", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    save_index(parse_args())
