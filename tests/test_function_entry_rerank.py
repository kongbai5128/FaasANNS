"""Function-entry search flow tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np

from server.routes import create_router
from server.schemas import RerankRequest
from vectors.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent.parent
FUNCTION_HANDLER = ROOT / "functions" / "ann_candidate_search" / "handler.py"


class DummySearchService:
    def __init__(self, store: VectorStore):
        self.store = store

    async def rerank(self, query, candidates, k):
        return self.store.rerank(query, candidates, k)


def test_rerank_route_returns_exact_topk() -> None:
    store = VectorStore.synthetic(dimension=4, count=10)
    router = create_router(DummySearchService(store), store)
    route = next(route for route in router.routes if getattr(route, "path", None) == "/rerank")
    request = RerankRequest(
        request_id="r1",
        vector=store.get(0).tolist(),
        candidates=[{"id": 3}, {"id": 0}, {"id": 2}],
        k=2,
    )

    response = _run(route.endpoint(request))

    assert response["request_id"] == "r1"
    assert [item["id"] for item in response["results"]] == [0, 2]
    assert response["timings_ms"]["total"] >= response["timings_ms"]["rerank"] >= 0.0
    assert response["rerank_metrics"] == {"candidate_count": 3, "result_count": 2}


def test_function_handler_can_call_server_rerank(monkeypatch) -> None:
    module = _load_function_handler(monkeypatch)

    candidates = [{"id": 3, "approx_score": 0.3}, {"id": 0, "approx_score": 0.1}]
    monkeypatch.setattr(module, "search_with_timings", lambda **kwargs: (candidates, {"search_total": 1.2}))
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
        return {
            "request_id": payload["request_id"],
            "results": [{"id": 0, "score": 0.0}],
            "timings_ms": {"total": 0.9, "rerank": 0.7},
            "rerank_metrics": {"candidate_count": 2, "result_count": 1},
        }

    monkeypatch.setattr(module, "_post_json", fake_post_json)

    response = module.handler(
        json.dumps(
            {
                "request_id": "r1",
                "query": np.zeros(4, dtype=np.float32).tolist(),
                "candidate_k": 2,
                "ef_search": 20,
                "k": 1,
                "rerank_server_url": "http://server.test/",
                "rerank_timeout": 5,
            }
        )
    )

    assert posted["url"] == "http://server.test/rerank"
    assert posted["payload"]["candidates"] == candidates
    assert posted["payload"]["k"] == 1
    assert posted["timeout"] == 5.0
    assert response["results"] == [{"id": 0, "score": 0.0}]
    assert response["plan"] == {"mode": "function_entry"}
    assert response["function_metrics"]["candidate_count"] == 2
    assert response["function_timings_ms"]["server_rerank_request"] >= 0.0
    assert response["server_timings_ms"] == {"total": 0.9, "rerank": 0.7}


def _load_function_handler(monkeypatch):
    name = f"function_entry_handler_{uuid.uuid4().hex}"
    sys.path.insert(0, str(FUNCTION_HANDLER.parent))
    monkeypatch.setenv("FAASANN_PQ_INDEX_PATH", "/tmp/not-needed-for-mocked-test.index")
    spec = importlib.util.spec_from_file_location(name, FUNCTION_HANDLER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(coro):
    import asyncio

    return asyncio.run(coro)
