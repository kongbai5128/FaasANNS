"""Function-side PQ candidate search tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "functions" / "ann_candidate_search" / "index_loader.py"


def test_function_pq_search_returns_candidates(tmp_path, monkeypatch) -> None:
    pq_dir = tmp_path / "pq"
    pq_dir.mkdir()
    meta_path = pq_dir / "pq_meta.json"
    codebooks_path = pq_dir / "pq_codebooks.npy"
    codes_path = pq_dir / "pq_codes.npy"
    ids_path = pq_dir / "pq_ids.npy"

    meta_path.write_text(
        json.dumps({"dimension": 2, "vector_count": 3, "subspace_count": 2}),
        encoding="utf-8",
    )
    np.save(codebooks_path, np.array([[[0.0], [10.0]], [[0.0], [10.0]]], dtype="float32"))
    np.save(codes_path, np.array([[0, 0], [1, 1], [0, 1]], dtype="uint8"))
    np.save(ids_path, np.array([100, 101, 102], dtype="int64"))

    monkeypatch.setenv("FAASANN_PQ_META_PATH", str(meta_path))
    monkeypatch.setenv("FAASANN_PQ_CODEBOOKS_PATH", str(codebooks_path))
    monkeypatch.setenv("FAASANN_PQ_CODES_PATH", str(codes_path))
    monkeypatch.setenv("FAASANN_PQ_IDS_PATH", str(ids_path))

    module = _load_index_loader()

    candidates = module.search(query=[0.0, 0.0], candidate_k=2)

    assert candidates == [
        {"id": 100, "approx_score": 0.0},
        {"id": 102, "approx_score": 100.0},
    ]


def _load_index_loader():
    name = f"function_pq_index_loader_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, INDEX_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
