"""Build a Faiss HNSW-PQ index with an exact-L2-built HNSW graph.

This is an experimental variant of ``build_pq_index.py``:

1. Build an ``IndexHNSWFlat`` on the original float vectors, so HNSW links are
   selected using exact L2 distances.
2. Build an ``IndexHNSWPQ`` with the same insertion order, so vectors are stored
   as PQ codes.
3. Copy the exact HNSW graph from the flat index onto the PQ index.

The resulting index searches with PQ distances, but its graph topology comes
from exact-vector construction. It is intended to test whether HNSWPQ recall is
limited by PQ-distance graph construction quality.

Example:

python data_generator/hnsw/build_exact_graph_pq_index.py \
  --src data/gist/gist_base.fvecs \
  --dst data/gist/index/full/pq_exact_graph \
  --subspaces 120 \
  --codebook-size 256 \
  --train-size 1000000 \
  --iterations 50 \
  --seed 0 \
  --hnsw-space l2 \
  --hnsw-m 48 \
  --hnsw-ef-construction 1250 \
  --hnsw-ef-search 1000 \
  --hnsw-batch-size 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import faiss
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_generator.hnsw.build_pq_index import INDEX_NAME, read_fvecs


def _validate_args(
    vectors: np.ndarray,
    *,
    subspaces: int,
    codebook_size: int,
    hnsw_batch_size: int,
    metric: str,
) -> None:
    if vectors.ndim != 2 or vectors.shape[0] <= 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if vectors.shape[1] % subspaces != 0:
        raise ValueError(f"dimension={vectors.shape[1]} is not divisible by subspaces={subspaces}")
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook-size must be a power of two")
    if metric != "l2":
        raise ValueError("exact-graph HNSW-PQ currently supports hnsw-space=l2")
    if hnsw_batch_size <= 0:
        raise ValueError("hnsw-batch-size must be positive")


def _add_in_batches(index: faiss.Index, vectors: np.ndarray, batch_size: int, *, label: str) -> None:
    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        index.add(np.ascontiguousarray(vectors[start:end]))
        print(f"{label}: added vectors {end}/{vectors.shape[0]}", flush=True)


def _set_pq_training_params(index: faiss.Index, *, iterations: int, seed: int) -> None:
    storage = faiss.downcast_index(index.storage)
    if not isinstance(storage, faiss.IndexPQ):
        raise TypeError(f"expected HNSWPQ storage to be IndexPQ, got {type(storage).__name__}")
    storage.pq.cp.niter = iterations
    storage.pq.cp.seed = seed


def copy_hnsw_graph(source: faiss.IndexHNSW, target: faiss.IndexHNSW) -> None:
    """Copy HNSW topology and search/build parameters from source to target.

    Source and target must contain the same number of vectors in the same id
    order. The target storage is intentionally left untouched.
    """

    if source.ntotal != target.ntotal:
        raise ValueError(f"source.ntotal={source.ntotal} does not match target.ntotal={target.ntotal}")

    for name in ("levels", "offsets", "neighbors", "cum_nneighbor_per_level", "assign_probas"):
        values = faiss.vector_to_array(getattr(source.hnsw, name))
        faiss.copy_array_to_vector(values, getattr(target.hnsw, name))

    for name in (
        "entry_point",
        "max_level",
        "efConstruction",
        "efSearch",
        "search_bounded_queue",
        "check_relative_distance",
    ):
        setattr(target.hnsw, name, getattr(source.hnsw, name))


def build_exact_graph_hnswpq_index(
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
) -> faiss.Index:
    _validate_args(
        vectors,
        subspaces=subspaces,
        codebook_size=codebook_size,
        hnsw_batch_size=hnsw_batch_size,
        metric=metric,
    )

    nbits = codebook_size.bit_length() - 1
    print(
        f"Building exact graph: IndexHNSWFlat(d={vectors.shape[1]}, M={hnsw_m}), "
        f"vectors={vectors.shape[0]}",
        flush=True,
    )
    exact_index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_L2)
    exact_index.hnsw.efConstruction = ef_construction
    exact_index.hnsw.efSearch = ef_search
    _add_in_batches(exact_index, vectors, hnsw_batch_size, label="exact graph")

    print(
        f"Building PQ storage: IndexHNSWPQ(d={vectors.shape[1]}, PQ{subspaces}x{nbits}, HNSW{hnsw_m})",
        flush=True,
    )
    pq_index = faiss.IndexHNSWPQ(vectors.shape[1], subspaces, hnsw_m, nbits, faiss.METRIC_L2)
    pq_index.hnsw.efConstruction = ef_construction
    pq_index.hnsw.efSearch = ef_search
    _set_pq_training_params(pq_index, iterations=iterations, seed=seed)

    rng = np.random.default_rng(seed)
    train_count = min(train_size, vectors.shape[0])
    train = np.ascontiguousarray(vectors[rng.choice(vectors.shape[0], size=train_count, replace=False)])
    print(f"Training PQ{subspaces}x{nbits}: vectors={train_count}", flush=True)
    pq_index.train(train)
    _add_in_batches(pq_index, vectors, hnsw_batch_size, label="pq storage")

    print("Copying exact HNSW graph onto PQ index", flush=True)
    copy_hnsw_graph(exact_index, pq_index)
    pq_index.hnsw.efSearch = ef_search
    return pq_index


def save_index(args: argparse.Namespace) -> None:
    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / INDEX_NAME
    if index_path.is_file():
        index_path.unlink()

    vectors = read_fvecs(args.src, args.max_vectors)
    index = build_exact_graph_hnswpq_index(
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
    faiss.write_index(index, str(index_path))
    print(f"Saved {index_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW-PQ with exact-L2 HNSW graph topology")
    parser.add_argument("--src", required=True, help="input .fvecs base vectors")
    parser.add_argument("--dst", required=True, help="output directory for faiss_hnswpq.index")
    parser.add_argument("--max-vectors", type=int, help="optional cap for debugging with a smaller subset")
    parser.add_argument("--subspaces", type=int, required=True, help="PQ subquantizers M; dimension must divide this value")
    parser.add_argument("--codebook-size", type=int, required=True, help="centroids per PQ subspace; 256 means 8 bits/code")
    parser.add_argument("--train-size", type=int, required=True, help="number of vectors sampled to train PQ codebooks")
    parser.add_argument("--iterations", type=int, required=True, help="k-means iterations for PQ training")
    parser.add_argument("--seed", type=int, required=True, help="random seed for deterministic training sampling")
    parser.add_argument("--hnsw-space", required=True, choices=["l2"], help="distance metric for the HNSW graph")
    parser.add_argument("--hnsw-m", type=int, required=True, help="HNSW graph connectivity")
    parser.add_argument("--hnsw-ef-construction", type=int, required=True, help="HNSW build-time search width")
    parser.add_argument("--hnsw-ef-search", type=int, required=True, help="default HNSW query-time search width")
    parser.add_argument("--hnsw-batch-size", type=int, required=True, help="vectors added per batch")
    return parser.parse_args()


if __name__ == "__main__":
    save_index(parse_args())
