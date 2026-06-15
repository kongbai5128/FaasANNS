# -*- coding: utf-8 -*-
"""云函数侧两阶段搜索：Faiss HNSW-PQ candidate search + raw vector exact rerank。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
INDEX_PATH = Path(
    os.environ.get("FAASANN_PQ_INDEX_PATH", str(DATA_ROOT / "index" / "full" / "pq" / "faiss_hnswpq.index"))
)
BASE_PATH = Path(os.environ.get("FAASANN_BASE_PATH", str(DATA_ROOT / "sift100w" / "sift_base.fvecs")))

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    index: faiss.Index
    vectors: np.memmap
    index_path: Path
    base_path: Path
    dimension: int
    vector_count: int


def warmup() -> None:
    load_state()


def index_status() -> dict:
    state = _state
    return {
        "loaded": state is not None,
        "index_type": "faiss_hnswpq",
        "index_path": str(INDEX_PATH),
        "base_path": str(BASE_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
    }


def search(query: list[float], k: int, candidate_k: int, ef_search: int) -> list[dict]:
    state = load_state()
    query_vector = np.asarray(query, dtype="float32")
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    candidates = _pq_candidates(state, query_vector, candidate_k, ef_search)
    return _rerank(state, query_vector, candidates, k)


def load_state() -> State:
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        _state = _load_state(INDEX_PATH, BASE_PATH)
        return _state


def _load_state(index_path: Path, base_path: Path) -> State:
    if not index_path.exists():
        raise FileNotFoundError(f"Faiss HNSW-PQ index file not found: {index_path}")

    index = faiss.read_index(str(index_path))
    dimension = int(index.d)
    vector_count = int(index.ntotal)
    vectors = _load_fvecs_memmap(base_path, dimension, vector_count)
    return State(
        index=index,
        vectors=vectors,
        index_path=index_path,
        base_path=base_path,
        dimension=dimension,
        vector_count=vector_count,
    )


def _load_fvecs_memmap(path: Path, dimension: int, vector_count: int) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"base fvecs file not found: {path}")
    raw = np.memmap(path, dtype=np.int32, mode="r")
    record_width = dimension + 1
    expected_size = vector_count * record_width
    if raw.size < expected_size:
        raise ValueError(f"{path} has too few records for vector_count={vector_count}, dimension={dimension}")

    records = raw[:expected_size].reshape(vector_count, record_width)
    if not np.all(records[:, 0] == dimension):
        raise ValueError(f"dimension mismatch in fvecs file: {path}")
    return records[:, 1:].view(np.float32)


def _pq_candidates(state: State, query: np.ndarray, candidate_k: int, ef_search: int) -> np.ndarray:
    params = faiss.SearchParametersHNSW(efSearch=ef_search)
    _, ids = state.index.search(query.reshape(1, -1), min(candidate_k, state.vector_count), params=params)
    ids = ids[0].astype(np.int64, copy=False)
    return ids[ids >= 0]


def _rerank(state: State, query: np.ndarray, candidate_ids: np.ndarray, k: int) -> list[dict]:
    if candidate_ids.size == 0:
        return []
    ids = _dedupe_ids(candidate_ids, state.vector_count)
    subset = state.vectors[ids]
    diff = subset - query.reshape(1, -1)
    scores = np.sum(diff * diff, axis=1)
    top_k = min(k, scores.shape[0])
    order = np.argsort(scores)[:top_k]
    return [{"id": int(ids[i]), "score": float(scores[i])} for i in order]


def _dedupe_ids(ids: np.ndarray, vector_count: int) -> np.ndarray:
    seen: dict[int, None] = {}
    for value in ids:
        vector_id = int(value)
        if 0 <= vector_id < vector_count:
            seen.setdefault(vector_id, None)
    return np.array(list(seen), dtype=np.int64)
