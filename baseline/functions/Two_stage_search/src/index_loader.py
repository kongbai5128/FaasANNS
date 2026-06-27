# -*- coding: utf-8 -*-
"""云函数侧两阶段搜索：Faiss HNSW-PQ candidate search + raw vector exact rerank。"""

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
INDEX_PATH = Path(
    os.environ.get("FAASANN_PQ_INDEX_PATH", str(DATASET_ROOT / "index" / "full" / "pq" / "faiss_hnswpq.index"))
)
BASE_FILE = os.environ.get("FAASANN_BASE_FILE", "sift_base.fvecs" if DATASET == "sift100w" else f"{DATASET}_base.fvecs")
BASE_PATH = Path(os.environ.get("FAASANN_BASE_PATH", str(DATASET_ROOT / BASE_FILE)))
NOMEMORY_MODE = os.environ.get("FAASANN_NOMEMORY", "").strip().lower() in {"1", "true", "yes", "on", "nomemory"}

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    """热实例中缓存的 PQ 索引、raw vector 访问方式和加载指标。"""

    index: faiss.Index
    vectors: np.memmap | None
    index_path: Path
    base_path: Path
    dimension: int
    vector_count: int
    cold_start_id: str
    index_loaded_at: float
    index_load_ms: float
    vector_memmap_load_ms: float
    index_file_size_bytes: int
    base_file_size_bytes: int


def warmup() -> None:
    """在服务监听前加载 PQ 索引；memory 模式也会建立 raw vector memmap。"""

    load_state()


def index_status() -> dict:
    """返回可 JSON 序列化的索引、raw vector 和实例状态。"""

    state = _state
    return {
        "loaded": state is not None,
        "dataset": DATASET,
        "index_type": "faiss_hnswpq",
        "index_path": str(INDEX_PATH),
        "base_path": str(BASE_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "vector_memmap_loaded": state.vectors is not None if state else False,
        "cold_start_id": state.cold_start_id if state else None,
        "index_loaded_at": state.index_loaded_at if state else None,
        "index_load_ms": state.index_load_ms if state else None,
        "vector_memmap_load_ms": state.vector_memmap_load_ms if state else None,
        "index_file_size_bytes": state.index_file_size_bytes if state else None,
        "base_file_size_bytes": state.base_file_size_bytes if state else None,
    }


def search(query: list[float], k: int, candidate_k: int, ef_search: int, nomemory: bool = False) -> list[dict]:
    """执行两阶段搜索，只返回最终 top-k，不暴露内部计时。"""

    results, _ = search_with_timings(
        query=query,
        k=k,
        candidate_k=candidate_k,
        ef_search=ef_search,
        nomemory=nomemory,
    )
    return results


def search_with_timings(
    query: list[float],
    k: int,
    candidate_k: int,
    ef_search: int,
    nomemory: bool = False,
) -> tuple[list[dict], dict[str, float]]:
    """执行 HNSW-PQ 候选搜索和 raw vector exact rerank，并返回阶段耗时。"""

    total_start = time.perf_counter()
    state, timings = load_state_with_timings()
    parse_start = time.perf_counter()
    query_vector = np.asarray(query, dtype="float32")
    timings["query_parse"] = _elapsed_ms(parse_start)
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    candidates = _pq_candidates(state, query_vector, candidate_k, ef_search, timings)
    results = _rerank(state, query_vector, candidates, k, timings, nomemory=nomemory)
    timings["search_total"] = _elapsed_ms(total_start)
    return results, timings


def load_state() -> State:
    """返回已加载的索引状态；必要时触发一次加载。"""

    state, _ = load_state_with_timings()
    return state


def load_state_with_timings() -> tuple[State, dict[str, float]]:
    """线程安全地加载或复用 PQ 索引，并记录加载等待和加载耗时。"""

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
        _state = _load_state(INDEX_PATH, BASE_PATH)
        timings["index_load"] = _state.index_load_ms
        timings["load_state"] = _elapsed_ms(total_start)
        return _state, timings


def _load_state(index_path: Path, base_path: Path) -> State:
    """从磁盘读取 Faiss HNSW-PQ 索引，并按运行模式准备 raw vectors。"""

    load_start = time.perf_counter()
    if not index_path.exists():
        raise FileNotFoundError(f"Faiss HNSW-PQ index file not found: {index_path}")
    index_file_size_bytes = index_path.stat().st_size
    base_file_size_bytes = base_path.stat().st_size if base_path.exists() else 0

    index = faiss.read_index(str(index_path))
    dimension = int(index.d)
    vector_count = int(index.ntotal)
    vector_memmap_load_ms = 0.0
    vectors = None
    if not NOMEMORY_MODE:
        memmap_start = time.perf_counter()
        vectors = _load_fvecs_memmap(base_path, dimension, vector_count)
        vector_memmap_load_ms = _elapsed_ms(memmap_start)
    return State(
        index=index,
        vectors=vectors,
        index_path=index_path,
        base_path=base_path,
        dimension=dimension,
        vector_count=vector_count,
        cold_start_id=uuid.uuid4().hex,
        index_loaded_at=time.time(),
        index_load_ms=_elapsed_ms(load_start),
        vector_memmap_load_ms=vector_memmap_load_ms,
        index_file_size_bytes=index_file_size_bytes,
        base_file_size_bytes=base_file_size_bytes,
    )


def _load_fvecs_memmap(path: Path, dimension: int, vector_count: int) -> np.memmap:
    """建立 raw fvecs 的全量 memmap 视图，供 memory 模式精排随机访问。"""

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


def _pq_candidates(
    state: State,
    query: np.ndarray,
    candidate_k: int,
    ef_search: int,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """用 Faiss HNSW-PQ 搜索第一阶段候选 id。"""

    search_start = time.perf_counter()
    params = faiss.SearchParametersHNSW(efSearch=ef_search)
    _, ids = state.index.search(query.reshape(1, -1), min(candidate_k, state.vector_count), params=params)
    ids = ids[0].astype(np.int64, copy=False)
    candidates = ids[ids >= 0]
    if timings is not None:
        timings["faiss_search"] = _elapsed_ms(search_start)
    return candidates


def _rerank(
    state: State,
    query: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    timings: dict[str, float] | None = None,
    nomemory: bool = False,
) -> list[dict]:
    """对候选 id 做 raw vector exact L2 rerank，并返回最终 top-k。"""

    rerank_start = time.perf_counter()
    if candidate_ids.size == 0:
        if timings is not None:
            timings["rerank_total"] = _elapsed_ms(rerank_start)
        return []
    dedupe_start = time.perf_counter()
    ids = _dedupe_ids(candidate_ids, state.vector_count)
    if timings is not None:
        timings["dedupe_ids"] = _elapsed_ms(dedupe_start)
    if nomemory:
        gather_start = time.perf_counter()
        subset = _read_fvecs_rows(state.base_path, ids, state.dimension)
        if timings is not None:
            timings["direct_fvecs_gather"] = _elapsed_ms(gather_start)
            timings["memmap_load"] = 0.0
    else:
        vectors, memmap_load_ms = _load_memmap_vectors(state)
        if timings is not None:
            timings["memmap_load"] = memmap_load_ms
        gather_start = time.perf_counter()
        subset = vectors[ids]
        if timings is not None:
            timings["memmap_gather"] = _elapsed_ms(gather_start)
    l2_start = time.perf_counter()
    diff = subset - query.reshape(1, -1)
    scores = np.sum(diff * diff, axis=1)
    if timings is not None:
        timings["l2_scores"] = _elapsed_ms(l2_start)
    top_k = min(k, scores.shape[0])
    argsort_start = time.perf_counter()
    order = np.argsort(scores)[:top_k]
    if timings is not None:
        timings["argsort"] = _elapsed_ms(argsort_start)
    format_start = time.perf_counter()
    results = [{"id": int(ids[i]), "score": float(scores[i])} for i in order]
    if timings is not None:
        timings["format_results"] = _elapsed_ms(format_start)
        timings["rerank_total"] = _elapsed_ms(rerank_start)
    return results


def _load_memmap_vectors(state: State) -> tuple[np.memmap, float]:
    """懒加载 raw vector memmap；通常只有 memory 模式的兜底路径会触发。"""

    if state.vectors is not None:
        return state.vectors, 0.0

    load_start = time.perf_counter()
    with _lock:
        if state.vectors is None:
            state.vectors = _load_fvecs_memmap(state.base_path, state.dimension, state.vector_count)
            state.vector_memmap_load_ms = _elapsed_ms(load_start)
            return state.vectors, state.vector_memmap_load_ms
        return state.vectors, 0.0


def _read_fvecs_rows(path: Path, ids: np.ndarray, dimension: int) -> np.ndarray:
    """nomemory 模式下按候选 id 逐条读取 fvecs 记录。"""

    record_bytes = (dimension + 1) * np.dtype(np.float32).itemsize
    subset = np.empty((ids.size, dimension), dtype=np.float32)
    with path.open("rb") as fp:
        for row, vector_id in enumerate(ids):
            fp.seek(int(vector_id) * record_bytes)
            record = fp.read(record_bytes)
            if len(record) != record_bytes:
                raise ValueError(f"{path} has no complete vector record for id={int(vector_id)}")
            stored_dimension = int(np.frombuffer(record, dtype=np.int32, count=1)[0])
            if stored_dimension != dimension:
                raise ValueError(
                    f"dimension mismatch in fvecs file: id={int(vector_id)}, "
                    f"expected={dimension}, got={stored_dimension}"
                )
            subset[row] = np.frombuffer(record, dtype=np.float32, count=dimension, offset=4)
    return subset


def _dedupe_ids(ids: np.ndarray, vector_count: int) -> np.ndarray:
    """保留候选顺序，去重并过滤越界 id。"""

    seen: dict[int, None] = {}
    for value in ids:
        vector_id = int(value)
        if 0 <= vector_id < vector_count:
            seen.setdefault(vector_id, None)
    return np.array(list(seen), dtype=np.int64)


def _elapsed_ms(start: float) -> float:
    """返回从 start 到当前时刻的毫秒数。"""

    return round((time.perf_counter() - start) * 1000.0, 3)
