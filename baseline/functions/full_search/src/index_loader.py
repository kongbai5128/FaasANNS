# -*- coding: utf-8 -*-
"""云函数侧 Faiss HNSW-Flat 全量搜索。"""

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
# DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/home/qian/Code/FaasANNS/data"))
DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
DATASET = os.environ.get("FAASANN_DATASET", DEFAULT_DATASET).strip().strip("/")
if not DATASET or "/" in DATASET or DATASET in {".", ".."}:
    raise ValueError(f"invalid FAASANN_DATASET={DATASET!r}")

DATASET_ROOT = DATA_ROOT / DATASET
DEFAULT_INDEX_PATH = DATASET_ROOT / "index" / "full" / "faiss_hnswflat.index"
INDEX_PATH = Path(
    os.environ.get(
        "FAASANN_FAISS_HNSW_INDEX_PATH",
        os.environ.get("FAASANN_HNSW_INDEX_PATH", str(DEFAULT_INDEX_PATH)),
    )
)

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    """热实例中缓存的 Faiss HNSW-Flat 索引状态。"""

    index: faiss.Index
    index_path: Path
    dimension: int
    vector_count: int
    metric_type: int
    cold_start_id: str
    index_loaded_at: float
    index_load_ms: float
    index_file_size_bytes: int


def warmup() -> None:
    """在服务监听前加载索引，避免首个查询承担加载成本。"""

    load_state()


def index_status() -> dict:
    """返回可 JSON 序列化的索引加载状态和实例指标。"""

    state = _state
    return {
        "loaded": state is not None,
        "dataset": DATASET,
        "index_path": str(INDEX_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "index_type": "faiss_hnswflat" if state else None,
        "metric_type": state.metric_type if state else None,
        "cold_start_id": state.cold_start_id if state else None,
        "index_loaded_at": state.index_loaded_at if state else None,
        "index_load_ms": state.index_load_ms if state else None,
        "index_file_size_bytes": state.index_file_size_bytes if state else None,
    }


def search(query: list[float], candidate_k: int, ef_search: int | None = None) -> list[dict]:
    """执行一次 Faiss HNSW 查询，只返回候选结果，不暴露内部计时。"""

    candidates, _ = search_with_timings(query=query, candidate_k=candidate_k, ef_search=ef_search)
    return candidates


def search_with_timings(query: list[float], candidate_k: int, ef_search: int | None = None) -> tuple[list[dict], dict[str, float]]:
    """执行一次 full HNSW 查询，并返回函数内部各阶段耗时。"""

    total_start = time.perf_counter()
    state, timings = load_state_with_timings()
    parse_start = time.perf_counter()
    query_vector = np.asarray(query, dtype="float32")
    timings["query_parse"] = _elapsed_ms(parse_start)
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    k = min(candidate_k, state.vector_count)
    faiss_start = time.perf_counter()
    if ef_search is not None and ef_search > 0:
        params = faiss.SearchParametersHNSW(efSearch=ef_search)
        distances, ids = state.index.search(query_vector.reshape(1, -1), k, params=params)
    else:
        distances, ids = state.index.search(query_vector.reshape(1, -1), k)
    timings["faiss_search"] = _elapsed_ms(faiss_start)

    format_start = time.perf_counter()
    candidates = [
        {"id": int(vector_id), "approx_score": float(score)}
        for vector_id, score in zip(ids[0], distances[0])
        if int(vector_id) >= 0 and float(score) < 3.4028235e38
    ]
    timings["format_candidates"] = _elapsed_ms(format_start)
    timings["search_total"] = _elapsed_ms(total_start)
    return candidates, timings


def load_state() -> State:
    """返回已加载的索引状态；必要时触发一次加载。"""

    state, _ = load_state_with_timings()
    return state


def load_state_with_timings() -> tuple[State, dict[str, float]]:
    """线程安全地加载或复用索引，并记录加载等待和加载耗时。"""

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
    """从磁盘读取 Faiss HNSW-Flat 索引。"""

    load_start = time.perf_counter()
    if not index_path.exists():
        raise FileNotFoundError(f"Faiss HNSW-Flat index file not found: {index_path}")
    index_file_size_bytes = index_path.stat().st_size

    index = faiss.read_index(str(index_path))
    dimension = int(index.d)
    vector_count = int(index.ntotal)
    return State(
        index=index,
        index_path=index_path,
        dimension=dimension,
        vector_count=vector_count,
        metric_type=int(index.metric_type),
        cold_start_id=uuid.uuid4().hex,
        index_loaded_at=time.time(),
        index_load_ms=_elapsed_ms(load_start),
        index_file_size_bytes=index_file_size_bytes,
    )


def _elapsed_ms(start: float) -> float:
    """返回从 start 到当前时刻的毫秒数。"""

    return round((time.perf_counter() - start) * 1000.0, 3)
