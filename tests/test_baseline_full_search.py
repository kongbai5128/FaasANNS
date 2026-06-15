"""Baseline HNSW function tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import hnswlib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX_LOADER = ROOT / "baseline" / "functions" / "full_search" / "src" / "index_loader.py"


def test_baseline_hnsw_search_returns_candidates(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "full_hnsw.bin"
    vectors = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    _write_hnsw_index(index_path, vectors)

    monkeypatch.setenv("FAASANN_HNSW_INDEX_PATH", str(index_path))
    module = _load_index_loader()

    candidates = module.search(query=[1.0, 0.0], candidate_k=3, ef_search=10)

    assert [item["id"] for item in candidates] == [0, 1, 2]
    assert [item["approx_score"] for item in candidates] == [1.0, 1.0, 10.0]


def _write_hnsw_index(path: Path, vectors: np.ndarray) -> None:
    index = hnswlib.Index(space="l2", dim=vectors.shape[1])
    index.init_index(max_elements=vectors.shape[0], ef_construction=100, M=16)
    index.add_items(vectors, np.arange(vectors.shape[0]))
    index.set_ef(10)
    index.save_index(str(path))
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "vector_count": int(vectors.shape[0]),
                "dimension": int(vectors.shape[1]),
                "space": "l2",
                "ef_search": 10,
            }
        ),
        encoding="utf-8",
    )


def _load_index_loader():
    name = f"baseline_hnsw_index_loader_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, INDEX_LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
