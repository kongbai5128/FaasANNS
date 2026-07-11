"""Build a Faiss HNSW-Flat index for full-search baselines.

This generator is the Faiss counterpart of ``constrained_kmeans_w_clusters.py``
when that script is used with ``--k 1``. It keeps the full-search baseline in
the same Faiss library family as the HNSW-PQ baseline, making full vs PQ
comparisons less affected by hnswlib-vs-Faiss implementation differences.

Output files:
  faiss_hnswflat.index
  faiss_hnswflat.meta.json

Examples:

python data_generator/hnsw/build_faiss_hnsw_index.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/sift100w/index/full \
  --hnsw-m 32 \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 99 \
  --hnsw-batch-size 50000

python data_generator/hnsw/build_faiss_hnsw_index.py \
  --src data/gist/gist_base.fvecs \
  --dst data/gist/index/full \
  --hnsw-m 48 \
  --hnsw-ef-construction 1250 \
  --hnsw-ef-search 100 \
  --hnsw-batch-size 50000
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


INDEX_NAME = "faiss_hnswflat.index"
META_NAME = "faiss_hnswflat.meta.json"


def build_hnswflat_index(
    vectors: np.ndarray,
    *,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    hnsw_batch_size: int,
    metric: str,
) -> faiss.IndexHNSWFlat:
    """Build an exact-vector Faiss HNSW index from contiguous float32 vectors."""

    if vectors.ndim != 2 or vectors.shape[0] <= 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if hnsw_m <= 0:
        raise ValueError("hnsw-m must be positive")
    if ef_construction <= 0:
        raise ValueError("hnsw-ef-construction must be positive")
    if ef_search <= 0:
        raise ValueError("hnsw-ef-search must be positive")
    if hnsw_batch_size <= 0:
        raise ValueError("hnsw-batch-size must be positive")
    if metric != "l2":
        raise ValueError("Faiss HNSW-Flat generator currently supports hnsw-space=l2")

    index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print(
        f"Building IndexHNSWFlat: vectors={vectors.shape[0]}, dim={vectors.shape[1]}, "
        f"M={hnsw_m}, efConstruction={ef_construction}",
        flush=True,
    )
    for start in range(0, vectors.shape[0], hnsw_batch_size):
        end = min(start + hnsw_batch_size, vectors.shape[0])
        index.add(np.ascontiguousarray(vectors[start:end]))
        print(f"Added vectors {end}/{vectors.shape[0]}", flush=True)
    return index


def save_index(args: argparse.Namespace) -> None:
    """Read vectors, build the Faiss HNSW-Flat index, and write index metadata."""

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / args.index_name
    meta_path = out / args.meta_name
    for path in (index_path, meta_path):
        if path.is_file():
            path.unlink()

    vectors = read_fvecs(args.src, args.max_vectors)
    index = build_hnswflat_index(
        vectors,
        hnsw_m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
        hnsw_batch_size=args.hnsw_batch_size,
        metric=args.hnsw_space,
    )
    faiss.write_index(index, str(index_path))
    print(f"Saved {index_path}", flush=True)

    meta = {
        "source": str(Path(args.src).resolve()),
        "index_type": "faiss_hnswflat",
        "index_file": index_path.name,
        "vector_count": int(index.ntotal),
        "dimension": int(index.d),
        "metric": args.hnsw_space,
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.hnsw_ef_construction,
        "hnsw_ef_search": args.hnsw_ef_search,
        "max_vectors": args.max_vectors,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {meta_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Faiss HNSW-Flat full-search index")
    parser.add_argument("--src", required=True, help="input .fvecs base vectors")
    parser.add_argument("--dst", required=True, help="output directory for faiss_hnswflat.index")
    parser.add_argument("--max-vectors", type=int, help="optional cap for debugging with a smaller subset")
    parser.add_argument("--index-name", default=INDEX_NAME, help="output Faiss index file name")
    parser.add_argument("--meta-name", default=META_NAME, help="output metadata file name")
    parser.add_argument("--hnsw-space", default="l2", choices=["l2"], help="distance metric")
    parser.add_argument("--hnsw-m", type=int, required=True, help="HNSW graph connectivity")
    parser.add_argument("--hnsw-ef-construction", type=int, required=True, help="HNSW build-time search width")
    parser.add_argument("--hnsw-ef-search", type=int, required=True, help="default HNSW query-time search width")
    parser.add_argument("--hnsw-batch-size", type=int, required=True, help="vectors added per batch")
    return parser.parse_args()


if __name__ == "__main__":
    save_index(parse_args())
