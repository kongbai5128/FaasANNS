# -*- coding: utf-8 -*-
"""云函数侧全量 HNSW 搜索。"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np


DEFAULT_DATASET = "sift100w"
DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
DATASET = os.environ.get("FAASANN_DATASET", DEFAULT_DATASET).strip().strip("/")
if not DATASET or "/" in DATASET or DATASET in {".", ".."}:
    raise ValueError(f"invalid FAASANN_DATASET={DATASET!r}")

DATASET_ROOT = DATA_ROOT / DATASET
INDEX_PATH = Path(os.environ.get("FAASANN_HNSW_INDEX_PATH", str(DATASET_ROOT / "index" / "full" / "full_hnsw.bin")))

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    index: hnswlib.Index
    index_path: Path
    dimension: int
    vector_count: int
    space: str
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
        "space": state.space if state else None,
        "cold_start_id": state.cold_start_id if state else None,
        "index_loaded_at": state.index_loaded_at if state else None,
        "index_load_ms": state.index_load_ms if state else None,
        "index_file_size_bytes": state.index_file_size_bytes if state else None,
    }


def search(query: list[float], candidate_k: int, ef_search: int | None = None) -> list[dict]:
    candidates, _ = search_with_timings(query=query, candidate_k=candidate_k, ef_search=ef_search)
    return candidates


def search_with_timings(query: list[float], candidate_k: int, ef_search: int | None = None) -> tuple[list[dict], dict[str, float]]:
    total_start = time.perf_counter()
    state, timings = load_state_with_timings()
    parse_start = time.perf_counter()
    query_vector = np.asarray(query, dtype="float32")
    timings["query_parse"] = _elapsed_ms(parse_start)
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    if ef_search is not None and ef_search > 0:
        set_ef_start = time.perf_counter()
        state.index.set_ef(ef_search)
        timings["hnsw_set_ef"] = _elapsed_ms(set_ef_start)
    k = min(candidate_k, state.vector_count)
    hnsw_start = time.perf_counter()
    ids, distances = state.index.knn_query(query_vector.reshape(1, -1), k=k)
    timings["hnsw_knn_query"] = _elapsed_ms(hnsw_start)

    format_start = time.perf_counter()
    candidates = [
        {"id": int(vector_id), "approx_score": float(score)}
        for vector_id, score in zip(ids[0], distances[0])
        if vector_id >= 0
    ]
    timings["format_candidates"] = _elapsed_ms(format_start)
    timings["search_total"] = _elapsed_ms(total_start)
    return candidates, timings


def _search_without_extra_timings(query: list[float], candidate_k: int, ef_search: int | None = None) -> list[dict]:
    state = load_state()
    query_vector = np.asarray(query, dtype="float32")
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    if ef_search is not None and ef_search > 0:
        state.index.set_ef(ef_search)
    k = min(candidate_k, state.vector_count)
    ids, distances = state.index.knn_query(query_vector.reshape(1, -1), k=k)
    return [
        {"id": int(vector_id), "approx_score": float(score)}
        for vector_id, score in zip(ids[0], distances[0])
        if vector_id >= 0
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
        load_start = time.perf_counter()
        _state = _load_hnsw_index(INDEX_PATH)
        timings["index_load"] = _state.index_load_ms
        timings["load_state"] = _elapsed_ms(total_start)
        return _state, timings


def _load_hnsw_index(index_path: Path) -> State:
    load_start = time.perf_counter()
    if not index_path.exists():
        raise FileNotFoundError(f"hnswlib index file not found: {index_path}")
    meta = _load_meta(index_path)
    index_file_size_bytes = index_path.stat().st_size

    dimension = int(meta["dimension"])
    vector_count = int(meta["vector_count"])
    space = str(meta["space"])
    index = hnswlib.Index(space=space, dim=dimension)
    index.load_index(str(index_path), max_elements=vector_count)
    index.set_ef(int(meta["ef_search"]))
    return State(
        index=index,
        index_path=index_path,
        dimension=dimension,
        vector_count=vector_count,
        space=space,
        cold_start_id=uuid.uuid4().hex,
        index_loaded_at=time.time(),
        index_load_ms=_elapsed_ms(load_start),
        index_file_size_bytes=index_file_size_bytes,
    )


def _load_meta(index_path: Path) -> dict:
    meta_path = index_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"hnswlib metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required_keys = {"dimension", "vector_count", "space", "ef_search"}
    missing = sorted(required_keys - set(meta))
    if missing:
        raise ValueError(f"hnswlib metadata missing keys {missing}: {meta_path}")
    return meta


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
