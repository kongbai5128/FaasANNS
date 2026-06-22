# -*- coding: utf-8 -*-
"""云函数侧 Faiss HNSW-PQ 候选搜索。"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


DEFAULT_DATASET = "sift100w"
DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
DATASET = os.environ.get("FAASANN_DATASET", DEFAULT_DATASET).strip().strip("/")
if not DATASET or "/" in DATASET or DATASET in {".", ".."}:
    raise ValueError(f"invalid FAASANN_DATASET={DATASET!r}")

DATASET_ROOT = DATA_ROOT / DATASET
INDEX_PATH = Path(
    os.environ.get("FAASANN_PQ_INDEX_PATH", str(DATASET_ROOT / "index" / "full" / "pq" / "faiss_hnswpq.index"))
)

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    index: faiss.Index
    dimension: int
    vector_count: int
    cold_start_id: str
    index_loaded_at: float
    index_load_ms: float
    index_file_size_bytes: int


def warmup() -> None:
    load_state()


def index_status() -> dict:
    state = _state
    return {
        "loaded": state is not None,
        "dataset": DATASET,
        "index_path": str(INDEX_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "cold_start_id": state.cold_start_id if state else None,
        "index_loaded_at": state.index_loaded_at if state else None,
        "index_load_ms": state.index_load_ms if state else None,
        "index_file_size_bytes": state.index_file_size_bytes if state else None,
    }


def search(query: list[float], candidate_k: int, ef_search: int) -> list[dict]:
    candidates, _ = search_with_timings(query=query, candidate_k=candidate_k, ef_search=ef_search)
    return candidates


def search_with_timings(query: list[float], candidate_k: int, ef_search: int) -> tuple[list[dict], dict[str, float]]:
    total_start = time.perf_counter()
    state, timings = load_state_with_timings()
    parse_start = time.perf_counter()
    query_vector = np.asarray(query, dtype="float32")
    timings["query_parse"] = _elapsed_ms(parse_start)
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    faiss_start = time.perf_counter()
    params = faiss.SearchParametersHNSW(efSearch=ef_search)
    distances, ids = state.index.search(query_vector.reshape(1, -1), min(candidate_k, state.vector_count), params=params)
    timings["faiss_search"] = _elapsed_ms(faiss_start)

    format_start = time.perf_counter()
    candidates = [
        {"id": int(vector_id), "approx_score": float(score)}
        for vector_id, score in zip(ids[0], distances[0], strict=True)
        if int(vector_id) >= 0 and float(score) < 3.4028235e38
    ]
    timings["format_candidates"] = _elapsed_ms(format_start)
    timings["search_total"] = _elapsed_ms(total_start)
    return candidates, timings


def _search_without_extra_timings(query: list[float], candidate_k: int, ef_search: int) -> list[dict]:
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
    state, _ = load_state_with_timings()
    return state


def load_state_with_timings() -> tuple[State, dict[str, float]]:
    global _state
    total_start = time.perf_counter()
    timings: dict[str, float] = {
        "load_wait": 0.0,
        "index_load": 0.0,
    }
    if _state is not None:
        timings["load_state"] = _elapsed_ms(total_start)
        return _state, timings
    wait_start = time.perf_counter()
    with _lock:
        timings["load_wait"] = _elapsed_ms(wait_start)
        if _state is not None:
            timings["load_state"] = _elapsed_ms(total_start)
            return _state, timings
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Faiss HNSW-PQ index file not found: {INDEX_PATH}")
        load_start = time.perf_counter()
        stat_start = time.perf_counter()
        index_file_size_bytes = INDEX_PATH.stat().st_size
        timings["index_stat"] = _elapsed_ms(stat_start)
        read_start = time.perf_counter()
        index = faiss.read_index(str(INDEX_PATH))
        timings["faiss_read_index"] = _elapsed_ms(read_start)
        index_load_ms = _elapsed_ms(load_start)
        _state = State(
            index=index,
            dimension=int(index.d),
            vector_count=int(index.ntotal),
            cold_start_id=uuid.uuid4().hex,
            index_loaded_at=time.time(),
            index_load_ms=index_load_ms,
            index_file_size_bytes=index_file_size_bytes,
        )
        timings["index_load"] = index_load_ms
        timings["load_state"] = _elapsed_ms(total_start)
        return _state, timings


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
