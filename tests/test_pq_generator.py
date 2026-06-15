"""Faiss HNSW-PQ generator tests."""

from __future__ import annotations

import numpy as np

from data_generator.hnsw.build_pq_index import build_hnswpq_index


def test_build_hnswpq_index_uses_configured_codebook_size() -> None:
    vectors = np.random.default_rng(0).random((160, 4), dtype=np.float32)

    index = build_hnswpq_index(
        vectors,
        subspaces=1,
        codebook_size=4,
        train_size=160,
        iterations=1,
        seed=0,
        hnsw_batch_size=64,
        hnsw_m=4,
        ef_construction=20,
        ef_search=20,
        metric="l2",
    )

    assert index.d == 4
    assert index.ntotal == 160
