"""Baseline two-stage function tests."""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "baseline" / "functions" / "Two_stage_search" / "src" / "index_loader.py"


def test_two_stage_search_reranks_pq_candidates(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "faiss_hnswpq.index"
    base_path = tmp_path / "vectors_base.fvecs"
    index_path.write_bytes(b"fake")
    vectors = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    _write_fvecs(base_path, vectors)
    _install_fake_faiss(
        monkeypatch,
        vector_count=len(vectors),
        dimension=vectors.shape[1],
        candidate_ids=[2, 1, 0],
    )

    monkeypatch.setenv("FAASANN_PQ_INDEX_PATH", str(index_path))
    monkeypatch.setenv("FAASANN_BASE_PATH", str(base_path))
    module = _load_index_loader()

    results = module.search(query=[1.0, 0.0], k=2, candidate_k=3, ef_search=10)

    assert [item["id"] for item in results] == [1, 0]
    assert [item["score"] for item in results] == [1.0, 1.0]

    status = module.index_status()
    assert status["cold_start_id"]
    assert isinstance(status["index_loaded_at"], float)


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    dimension = vectors.shape[1]
    with path.open("wb") as fp:
        for vector in vectors:
            np.array([dimension], dtype=np.int32).tofile(fp)
            vector.astype(np.float32, copy=False).tofile(fp)


def _install_fake_faiss(monkeypatch, vector_count: int, dimension: int, candidate_ids: list[int]) -> None:
    class FakeIndex:
        d = dimension
        ntotal = vector_count

        def search(self, query, k, params=None):
            ids = np.array(candidate_ids[:k], dtype=np.int64).reshape(1, -1)
            distances = np.arange(ids.shape[1], dtype=np.float32).reshape(1, -1)
            return distances, ids

    class SearchParametersHNSW:
        def __init__(self, efSearch: int):
            self.efSearch = efSearch

    fake_faiss = types.SimpleNamespace(
        Index=FakeIndex,
        SearchParametersHNSW=SearchParametersHNSW,
        read_index=lambda path: FakeIndex(),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)


def _load_index_loader():
    name = f"baseline_two_stage_index_loader_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, INDEX_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
