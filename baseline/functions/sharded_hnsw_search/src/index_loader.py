# -*- coding: utf-8 -*-
"""Per-server k-means partition HNSW loader.

Each server loads exactly one partition selected by ``FAASANN_SHARD_ID``. The
same upload package can therefore be used by all shards; deployment commands
only differ by the shard id argument passed to ``app.py``.
"""

from __future__ import annotations

import json
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


SHARD_ID = int(os.environ.get("FAASANN_SHARD_ID", "0"))
SHARDING_DIR = Path(os.environ.get("FAASANN_SHARDING_DIR", str(DATA_ROOT / DATASET / "sharding_8")))
MANIFEST_PATH = Path(os.environ.get("FAASANN_SHARDING_MANIFEST_PATH", str(SHARDING_DIR / "manifest.json")))

_lock = threading.Lock()
_state: "State | None" = None


@dataclass(slots=True)
class State:
    """Hot-instance cache for one partition index."""

    index: faiss.Index
    ids: np.ndarray
    shard_id: int
    shard_count: int
    partition_dir: Path
    index_path: Path
    ids_path: Path
    dimension: int
    vector_count: int
    global_vector_count: int
    metric_type: int
    cold_start_id: str
    index_loaded_at: float
    index_load_ms: float
    index_file_size_bytes: int
    ids_file_size_bytes: int


def warmup() -> None:
    """Load this server's partition index before serving traffic."""

    load_state()


def index_status() -> dict:
    """Return JSON-serializable status for this shard."""

    state = _state
    return {
        "loaded": state is not None,
        "dataset": DATASET,
        "index_type": "faiss_hnswflat_kmeans_partition" if state else None,
        "shard_id": SHARD_ID,
        "shard_count": state.shard_count if state else None,
        "sharding_dir": str(SHARDING_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "partition_dir": str(state.partition_dir) if state else None,
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "global_vector_count": state.global_vector_count if state else None,
        "metric_type": state.metric_type if state else None,
        "cold_start_id": state.cold_start_id if state else None,
        "index_loaded_at": state.index_loaded_at if state else None,
        "index_load_ms": state.index_load_ms if state else None,
        "index_file_size_bytes": state.index_file_size_bytes if state else None,
        "ids_file_size_bytes": state.ids_file_size_bytes if state else None,
    }


def local_search(query: list[float], candidate_k: int, ef_search: int | None = None) -> tuple[list[dict], dict[str, float]]:
    """Search this server's local partition and return global ids."""

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
    search_start = time.perf_counter()
    if ef_search is not None and ef_search > 0:
        params = faiss.SearchParametersHNSW(efSearch=ef_search)
        distances, local_ids = state.index.search(query_vector.reshape(1, -1), k, params=params)
    else:
        distances, local_ids = state.index.search(query_vector.reshape(1, -1), k)
    timings["faiss_search"] = _elapsed_ms(search_start)

    format_start = time.perf_counter()
    candidates = []
    for local_id, score in zip(local_ids[0], distances[0]):
        local_id_int = int(local_id)
        score_float = float(score)
        if local_id_int < 0 or score_float >= 3.4028235e38:
            continue
        candidates.append(
            {
                "id": int(state.ids[local_id_int]),
                "approx_score": score_float,
                "shard_id": state.shard_id,
            }
        )
    timings["format_candidates"] = _elapsed_ms(format_start)
    timings["search_total"] = _elapsed_ms(total_start)
    return candidates, timings


def load_state() -> State:
    """Return cached partition state, loading it if needed."""

    state, _ = load_state_with_timings()
    return state


def load_state_with_timings() -> tuple[State, dict[str, float]]:
    """Thread-safe partition loading with timing fields."""

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
        _state = _load_state(SHARDING_DIR, MANIFEST_PATH, SHARD_ID)
        timings["index_load"] = _state.index_load_ms
        timings["load_state"] = _elapsed_ms(total_start)
        return _state, timings


def _load_state(sharding_dir: Path, manifest_path: Path, shard_id: int) -> State:
    """Load one partition from a k-means sharding manifest."""

    load_start = time.perf_counter()
    if not manifest_path.exists():
        raise FileNotFoundError(f"sharding manifest file not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError(f"manifest has no partitions: {manifest_path}")
    if shard_id < 0 or shard_id >= len(partitions):
        raise ValueError(f"FAASANN_SHARD_ID={shard_id} out of range for {len(partitions)} partitions")

    item = next((partition for partition in partitions if int(partition["shard_id"]) == shard_id), None)
    if item is None:
        raise ValueError(f"manifest has no partition for shard_id={shard_id}")

    partition_dir = sharding_dir / str(item["path"])
    index_path = partition_dir / str(item["index_file"])
    ids_path = partition_dir / str(item["ids_file"])
    if not index_path.exists():
        raise FileNotFoundError(f"partition index file not found: {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"partition ids file not found: {ids_path}")

    index_file_size = index_path.stat().st_size
    ids_file_size = ids_path.stat().st_size
    index = faiss.read_index(str(index_path))
    ids = np.load(ids_path, mmap_mode="r")
    if ids.shape != (int(index.ntotal),):
        raise ValueError(f"ids shape {ids.shape} does not match index.ntotal={index.ntotal}")

    return State(
        index=index,
        ids=ids,
        shard_id=shard_id,
        shard_count=int(manifest["shard_count"]),
        partition_dir=partition_dir,
        index_path=index_path,
        ids_path=ids_path,
        dimension=int(index.d),
        vector_count=int(index.ntotal),
        global_vector_count=int(manifest["vector_count"]),
        metric_type=int(index.metric_type),
        cold_start_id=uuid.uuid4().hex,
        index_loaded_at=time.time(),
        index_load_ms=_elapsed_ms(load_start),
        index_file_size_bytes=index_file_size,
        ids_file_size_bytes=ids_file_size,
    )


def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds from start to now."""

    return round((time.perf_counter() - start) * 1000.0, 3)
