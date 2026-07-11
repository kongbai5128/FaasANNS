"""Baseline Faiss HNSW full-search function tests."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "baseline" / "functions" / "full_search" / "src" / "index_loader.py"


def test_baseline_faiss_hnsw_search_returns_candidates(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "faiss_hnswflat.index"
    vectors = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    _write_faiss_hnsw_index(index_path, vectors)

    monkeypatch.setenv("FAASANN_FAISS_HNSW_INDEX_PATH", str(index_path))
    module = _load_index_loader()

    candidates = module.search(query=[1.0, 0.0], candidate_k=3, ef_search=10)

    assert [item["id"] for item in candidates] == [0, 1, 2]
    assert [item["approx_score"] for item in candidates] == [1.0, 1.0, 10.0]

    status = module.index_status()
    assert status["index_type"] == "faiss_hnswflat"
    assert status["cold_start_id"]
    assert isinstance(status["index_loaded_at"], float)


def _write_faiss_hnsw_index(path: Path, vectors: np.ndarray) -> None:
    index = faiss.IndexHNSWFlat(vectors.shape[1], 16, faiss.METRIC_L2)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 10
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(path))


def _load_index_loader():
    name = f"baseline_hnsw_index_loader_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, INDEX_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
