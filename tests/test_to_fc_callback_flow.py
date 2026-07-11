"""to_FC callback flow tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FUNCTION_HANDLER = ROOT / "to_FC" / "function" / "handler.py"


def test_to_fc_function_posts_candidates_to_vm_callback(monkeypatch) -> None:
    module = _load_function_handler(monkeypatch)

    candidates = [{"id": 3, "approx_score": 0.3}, {"id": 0, "approx_score": 0.1}]
    monkeypatch.setattr(module, "search_with_timings", lambda **kwargs: (candidates, {"faiss_search": 1.2}))
    monkeypatch.setattr(
        module,
        "index_status",
        lambda: {
            "cold_start_id": "cold-1",
            "index_loaded_at": 123.0,
            "index_file_size_bytes": 456,
            "index_load_ms": 7.8,
        },
    )

    posted = {}

    def fake_post_json(url: str, payload: dict, timeout: float) -> dict:
        posted["url"] = url
        posted["payload"] = payload
        posted["timeout"] = timeout
        return {"status": "accepted", "request_id": payload["request_id"]}

    monkeypatch.setattr(module, "_post_json", fake_post_json)

    response = module.handler(
        json.dumps(
            {
                "type": "search_callback",
                "request_id": "r1",
                "query": np.zeros(4, dtype=np.float32).tolist(),
                "candidate_k": 2,
                "ef_search": 20,
                "k": 1,
                "rerank_server_url": "http://server.test/",
                "client_callback_url": "http://client.test/callback",
                "rerank_accept_timeout": 5,
            }
        )
    )

    assert posted["url"] == "http://server.test/rerank_callback"
    assert posted["payload"]["candidates"] == candidates
    assert posted["payload"]["client_callback_url"] == "http://client.test/callback"
    assert posted["payload"]["k"] == 1
    assert posted["timeout"] == 5.0
    assert response["status"] == "accepted"
    assert response["plan"] == {"mode": "fc_to_vm_callback"}
    assert "results" not in response
    assert response["function_metrics"]["candidate_count"] == 2


def _load_function_handler(monkeypatch):
    name = f"to_fc_callback_handler_{uuid.uuid4().hex}"
    old_path = list(sys.path)
    old_index_loader = sys.modules.pop("index_loader", None)
    sys.path.insert(0, str(FUNCTION_HANDLER.parent))
    monkeypatch.setenv("FAASANN_PQ_INDEX_PATH", "/tmp/not-needed-for-mocked-test.index")
    spec = importlib.util.spec_from_file_location(name, FUNCTION_HANDLER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.modules.pop("index_loader", None)
        if old_index_loader is not None:
            sys.modules["index_loader"] = old_index_loader
