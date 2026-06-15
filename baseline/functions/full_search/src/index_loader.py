# -*- coding: utf-8 -*-
"""云函数侧全量 HNSW 搜索。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np


DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
INDEX_PATH = Path(os.environ.get("FAASANN_HNSW_INDEX_PATH", str(DATA_ROOT / "index" / "full" / "full_hnsw.bin")))

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    index: hnswlib.Index
    index_path: Path
    dimension: int
    vector_count: int
    space: str


def warmup() -> None:
    load_state()


def index_status() -> dict:
    state = _state
    return {
        "loaded": state is not None,
        "index_path": str(INDEX_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "space": state.space if state else None,
    }


def search(query: list[float], candidate_k: int, ef_search: int | None = None) -> list[dict]:
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
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        _state = _load_hnsw_index(INDEX_PATH)
        return _state


def _load_hnsw_index(index_path: Path) -> State:
    if not index_path.exists():
        raise FileNotFoundError(f"hnswlib index file not found: {index_path}")
    meta = _load_meta(index_path)

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
