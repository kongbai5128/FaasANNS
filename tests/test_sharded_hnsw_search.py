"""Distributed sharded HNSW baseline tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "baseline" / "functions" / "sharded_hnsw_search" / "src" / "index_loader.py"
HANDLER = ROOT / "baseline" / "functions" / "sharded_hnsw_search" / "src" / "handler.py"


def test_partition_loader_maps_local_ids_to_global_ids(tmp_path, monkeypatch) -> None:
    sharding_dir = tmp_path / "sharding_8"
    vectors = np.array([[8.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    global_ids = np.array([20, 30], dtype=np.int64)
    _write_partition(sharding_dir, shard_id=3, vectors=vectors, global_ids=global_ids, shard_count=8)

    monkeypatch.setenv("FAASANN_SHARDING_DIR", str(sharding_dir))
    monkeypatch.setenv("FAASANN_SHARD_ID", "3")
    loader = _load_module(INDEX_LOADER)

    candidates, timings = loader.local_search(query=[9.5, 0.0], candidate_k=2, ef_search=10)

    assert [item["id"] for item in candidates] == [30, 20]
    assert all(item["shard_id"] == 3 for item in candidates)
    assert timings["faiss_search"] >= 0.0
    assert loader.index_status()["shard_id"] == 3


def test_merge_candidates_sorts_global_results(monkeypatch) -> None:
    monkeypatch.setenv("FAASANN_SHARD_ID", "0")
    monkeypatch.setenv("FAASANN_PEER_ENDPOINTS", "")
    sys.path.insert(0, str(HANDLER.parent))
    try:
        handler = _load_module(HANDLER)
    finally:
        sys.path.remove(str(HANDLER.parent))

    merged = handler._merge_candidates(
        [
            {"candidates": [{"id": 1, "approx_score": 5.0}, {"id": 2, "approx_score": 2.0}]},
            {"candidates": [{"id": 3, "approx_score": 1.0}]},
        ],
        candidate_k=2,
    )

    assert [item["id"] for item in merged] == [3, 2]


def test_coordinator_timings_report_peer_and_all_shard_metrics(monkeypatch) -> None:
    monkeypatch.setenv("FAASANN_SHARD_ID", "0")
    monkeypatch.setenv("FAASANN_PEER_ENDPOINTS", "")
    sys.path.insert(0, str(HANDLER.parent))
    try:
        handler = _load_module(HANDLER)
    finally:
        sys.path.remove(str(HANDLER.parent))

    timings = handler._coordinator_timings(
        {"timings_ms": {"faiss_search": 2.0, "search_total": 3.0, "handler_total": 4.0}},
        [
            {"_peer_request_ms": 20.0, "timings_ms": {"faiss_search": 5.0}},
            {"_peer_request_ms": 30.0, "timings_ms": {"faiss_search": 7.0}},
        ],
    )

    assert timings["shard0_faiss_search"] == 2.0
    assert timings["peer_request_max"] == 30.0
    assert timings["peer_faiss_search_max"] == 7.0
    assert timings["all_shard_faiss_search_max"] == 7.0
    assert timings["all_shard_faiss_search_sum"] == 14.0


def _write_partition(sharding_dir: Path, *, shard_id: int, vectors: np.ndarray, global_ids: np.ndarray, shard_count: int) -> None:
    partition_dir = sharding_dir / f"partition_{shard_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(vectors.shape[1], 4, faiss.METRIC_L2)
    index.hnsw.efConstruction = 20
    index.hnsw.efSearch = 10
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(partition_dir / "hnsw.index"))
    np.save(partition_dir / "ids.npy", global_ids)
    (partition_dir / "meta.json").write_text(
        json.dumps(
            {
                "shard_id": shard_id,
                "path": partition_dir.name,
                "index_file": "hnsw.index",
                "ids_file": "ids.npy",
                "base_file": "base.fvecs",
                "dimension": int(vectors.shape[1]),
                "vector_count": int(vectors.shape[0]),
            }
        ),
        encoding="utf-8",
    )
    partitions = [
        {
            "shard_id": item,
            "path": f"partition_{item}",
            "index_file": "hnsw.index",
            "ids_file": "ids.npy",
            "base_file": "base.fvecs",
            "dimension": int(vectors.shape[1]),
            "vector_count": int(vectors.shape[0]) if item == shard_id else 0,
        }
        for item in range(shard_count)
    ]
    (sharding_dir / "manifest.json").write_text(
        json.dumps(
            {
                "index_type": "faiss_hnswflat_kmeans_sharded",
                "dimension": int(vectors.shape[1]),
                "vector_count": int(vectors.shape[0]),
                "shard_count": shard_count,
                "partitions": partitions,
            }
        ),
        encoding="utf-8",
    )


def _load_module(path: Path):
    name = f"sharded_hnsw_{uuid.uuid4().hex}"
    old_index_loader = sys.modules.pop("index_loader", None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("index_loader", None)
        if old_index_loader is not None:
            sys.modules["index_loader"] = old_index_loader
