"""Build k-means sharded Faiss HNSW-Flat indexes.

Output layout:

  data/gist/sharding_8/
    manifest.json
    centroids.npy
    labels.npy
    partition_0/
      base.fvecs
      ids.npy
      hnsw.index
      meta.json
    ...

Each partition index stores local ids. ``ids.npy`` maps local row ids back to
global vector ids. At runtime every server loads exactly one partition by
``FAASANN_SHARD_ID``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import faiss
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_generator.hnsw.build_pq_index import read_fvecs


def build_hnswflat_index(
    vectors: np.ndarray,
    *,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    batch_size: int,
) -> faiss.IndexHNSWFlat:
    """Build one partition-local Faiss HNSW-Flat index."""

    index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        index.add(np.ascontiguousarray(vectors[start:end]))
        print(f"  added vectors {end}/{vectors.shape[0]}", flush=True)
    return index


def train_kmeans(
    vectors: np.ndarray,
    *,
    shard_count: int,
    train_size: int,
    iterations: int,
    seed: int,
) -> faiss.Kmeans:
    """Train Faiss k-means centroids on a deterministic sample."""

    if shard_count <= 0:
        raise ValueError("shards must be positive")
    if vectors.shape[0] < shard_count:
        raise ValueError("number of vectors must be at least the number of shards")

    rng = np.random.default_rng(seed)
    sample_count = min(train_size, vectors.shape[0])
    sample_ids = rng.choice(vectors.shape[0], size=sample_count, replace=False)
    sample = np.ascontiguousarray(vectors[sample_ids])
    kmeans = faiss.Kmeans(
        d=vectors.shape[1],
        k=shard_count,
        niter=iterations,
        nredo=1,
        seed=seed,
        verbose=True,
        gpu=False,
    )
    print(f"Training k-means: shards={shard_count}, train_vectors={sample_count}", flush=True)
    kmeans.train(sample)
    return kmeans


def assign_partitions(vectors: np.ndarray, centroids: np.ndarray, batch_size: int) -> np.ndarray:
    """Assign every vector to its nearest centroid."""

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(centroids.astype(np.float32)))
    labels = np.empty(vectors.shape[0], dtype=np.int32)
    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        _, batch_labels = index.search(np.ascontiguousarray(vectors[start:end]), 1)
        labels[start:end] = batch_labels[:, 0].astype(np.int32, copy=False)
        print(f"Assigned vectors {end}/{vectors.shape[0]}", flush=True)
    return labels


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    """Write vectors in standard .fvecs format."""

    dimension = vectors.shape[1]
    with path.open("wb") as fp:
        header = np.full((vectors.shape[0], 1), dimension, dtype=np.int32)
        rows = np.concatenate([header.view(np.float32), vectors.astype(np.float32, copy=False)], axis=1)
        rows.tofile(fp)


def save_indexes(args: argparse.Namespace) -> None:
    """Build k-means partitions and one HNSW index per partition."""

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    vectors = read_fvecs(args.src, args.max_vectors)
    if args.hnsw_space != "l2":
        raise ValueError("this generator currently supports hnsw-space=l2")

    kmeans = train_kmeans(
        vectors,
        shard_count=args.shards,
        train_size=args.train_size,
        iterations=args.kmeans_iterations,
        seed=args.seed,
    )
    centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))
    labels = assign_partitions(vectors, centroids, args.assign_batch_size)
    np.save(out / "centroids.npy", centroids)
    np.save(out / "labels.npy", labels)

    partitions = []
    for shard_id in range(args.shards):
        ids = np.where(labels == shard_id)[0].astype(np.int64)
        if ids.size == 0:
            raise ValueError(f"partition {shard_id} is empty")
        partition_vectors = np.ascontiguousarray(vectors[ids])
        partition_dir = out / f"partition_{shard_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building partition {shard_id}: vectors={partition_vectors.shape[0]}", flush=True)
        index = build_hnswflat_index(
            partition_vectors,
            hnsw_m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            ef_search=args.hnsw_ef_search,
            batch_size=args.hnsw_batch_size,
        )
        faiss.write_index(index, str(partition_dir / "hnsw.index"))
        np.save(partition_dir / "ids.npy", ids)
        write_fvecs(partition_dir / "base.fvecs", partition_vectors)

        meta = {
            "shard_id": shard_id,
            "index_file": "hnsw.index",
            "ids_file": "ids.npy",
            "base_file": "base.fvecs",
            "dimension": int(vectors.shape[1]),
            "vector_count": int(partition_vectors.shape[0]),
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search": args.hnsw_ef_search,
        }
        (partition_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        partitions.append({"shard_id": shard_id, "path": partition_dir.name, **meta})

    manifest = {
        "source": str(Path(args.src).resolve()),
        "index_type": "faiss_hnswflat_kmeans_sharded",
        "partitioning": "faiss_kmeans",
        "metric": args.hnsw_space,
        "dimension": int(vectors.shape[1]),
        "vector_count": int(vectors.shape[0]),
        "shard_count": args.shards,
        "train_size": min(args.train_size, vectors.shape[0]),
        "kmeans_iterations": args.kmeans_iterations,
        "seed": args.seed,
        "partitions": partitions,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {out / 'manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build k-means sharded Faiss HNSW-Flat indexes")
    parser.add_argument("--src", required=True, help="input .fvecs base vectors")
    parser.add_argument("--dst", required=True, help="output sharding directory")
    parser.add_argument("--max-vectors", type=int, help="optional cap for debugging")
    parser.add_argument("--shards", type=int, default=8, help="number of k-means partitions")
    parser.add_argument("--train-size", type=int, default=200000, help="vectors sampled for k-means training")
    parser.add_argument("--kmeans-iterations", type=int, default=25, help="Faiss k-means iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--assign-batch-size", type=int, default=50000, help="vectors assigned per batch")
    parser.add_argument("--hnsw-space", default="l2", choices=["l2"], help="distance metric")
    parser.add_argument("--hnsw-m", type=int, required=True, help="HNSW graph connectivity")
    parser.add_argument("--hnsw-ef-construction", type=int, required=True, help="HNSW build-time search width")
    parser.add_argument("--hnsw-ef-search", type=int, required=True, help="default HNSW query-time search width")
    parser.add_argument("--hnsw-batch-size", type=int, required=True, help="vectors added per batch")
    return parser.parse_args()


if __name__ == "__main__":
    save_indexes(parse_args())
