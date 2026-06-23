"""Faiss HNSW-PQ generator tests."""

from __future__ import annotations

import numpy as np
import faiss

from data_generator.hnsw.build_exact_graph_pq_index import build_exact_graph_hnswpq_index
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


def test_build_exact_graph_hnswpq_copies_exact_hnsw_topology() -> None:
    vectors = np.random.default_rng(1).random((180, 4), dtype=np.float32)

    index = build_exact_graph_hnswpq_index(
        vectors,
        subspaces=1,
        codebook_size=4,
        train_size=180,
        iterations=1,
        seed=0,
        hnsw_batch_size=64,
        hnsw_m=4,
        ef_construction=20,
        ef_search=20,
        metric="l2",
    )

    exact_index = faiss.IndexHNSWFlat(4, 4, faiss.METRIC_L2)
    exact_index.hnsw.efConstruction = 20
    exact_index.hnsw.efSearch = 20
    exact_index.add(vectors[:64])
    exact_index.add(vectors[64:128])
    exact_index.add(vectors[128:])

    assert index.d == 4
    assert index.ntotal == 180
    assert isinstance(index, faiss.IndexHNSWPQ)
    storage = faiss.downcast_index(index.storage)
    assert isinstance(storage, faiss.IndexPQ)
    assert storage.pq.nbits == 2
    assert index.hnsw.nb_neighbors(0) == 8
    assert np.array_equal(faiss.vector_to_array(index.hnsw.levels), faiss.vector_to_array(exact_index.hnsw.levels))
    assert np.array_equal(faiss.vector_to_array(index.hnsw.offsets), faiss.vector_to_array(exact_index.hnsw.offsets))
    assert np.array_equal(faiss.vector_to_array(index.hnsw.neighbors), faiss.vector_to_array(exact_index.hnsw.neighbors))
