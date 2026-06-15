# -*- coding: utf-8 -*-
"""云函数侧 Faiss HNSW-PQ 候选搜索。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


INDEX_PATH = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann")) / "index" / "full" / "pq" / "faiss_hnswpq.index"

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    index: faiss.Index
    dimension: int
    vector_count: int


def warmup() -> None:
    load_state()


def index_status() -> dict:
    state = _state
    return {
        "loaded": state is not None,
        "index_path": str(INDEX_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
    }


def search(query: list[float], candidate_k: int, ef_search: int) -> list[dict]:
    state = load_state()
    query_vector = np.asarray(query, dtype="float32")
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    params = faiss.SearchParametersHNSW(efSearch=ef_search)
    distances, ids = state.index.search(query_vector.reshape(1, -1), min(candidate_k, state.vector_count), params=params)
    return [
        {"id": int(vector_id), "approx_score": float(score)}
        for vector_id, score in zip(ids[0], distances[0], strict=True)
        if int(vector_id) >= 0 and float(score) < 3.4028235e38
    ]


def load_state() -> State:
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Faiss HNSW-PQ index file not found: {INDEX_PATH}")
        index = faiss.read_index(str(INDEX_PATH))
        _state = State(index=index, dimension=int(index.d), vector_count=int(index.ntotal))
        return _state
