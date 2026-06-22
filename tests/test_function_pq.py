"""Function-side Faiss HNSW-PQ candidate search tests."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "functions" / "ann_candidate_search" / "index_loader.py"


def test_function_faiss_hnswpq_search_returns_candidates(tmp_path, monkeypatch) -> None:
    pq_dir = tmp_path / "index" / "full" / "pq"
    pq_dir.mkdir(parents=True)

    vectors = np.random.default_rng(0).random((160, 4), dtype=np.float32)
    index = faiss.index_factory(4, "HNSW4,PQ1x2", faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(pq_dir / "faiss_hnswpq.index"))

    monkeypatch.setenv("FAASANN_PQ_INDEX_PATH", str(pq_dir / "faiss_hnswpq.index"))
    module = _load_index_loader()

    candidates = module.search(query=vectors[0].tolist(), candidate_k=5, ef_search=20)

    assert len(candidates) == 5
    assert candidates[0]["id"] >= 0
    assert candidates[0]["approx_score"] >= 0.0

    status = module.index_status()
    assert status["cold_start_id"]
    assert isinstance(status["index_loaded_at"], float)


def test_function_faiss_hnswpq_uses_dataset_from_env(tmp_path, monkeypatch) -> None:
    pq_dir = tmp_path / "gist" / "index" / "full" / "pq"
    pq_dir.mkdir(parents=True)

    vectors = np.random.default_rng(1).random((160, 4), dtype=np.float32)
    index = faiss.index_factory(4, "HNSW4,PQ1x2", faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(pq_dir / "faiss_hnswpq.index"))

    monkeypatch.setenv("FAASANN_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FAASANN_DATASET", "gist")
    module = _load_index_loader()

    assert module.index_status()["dataset"] == "gist"
    assert module.index_status()["index_path"] == str(pq_dir / "faiss_hnswpq.index")


def _load_index_loader():
    name = f"function_faiss_index_loader_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, INDEX_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
