"""SearchService PQ offload pipeline tests."""

from __future__ import annotations

import asyncio

import pytest

from scaling.metrics import RuntimeMetrics
from scaling.planner import OffloadPlan
from search.service import SearchService
from utils.config import HNSWConfig, PipelineConfig, SearchConfig
from vectors.vector_store import VectorStore


class DummyProvider:
    def __init__(self) -> None:
        self.payload_json = None

    async def invoke(self, payload) -> list[dict]:
        self.payload_json = payload.to_json()
        return [
            {
                "id": 3,
                "_cold_start_id": "cold-1",
                "_index_loaded_at": 123.5,
                "_function_timings_ms": {"handler_total": 1.0, "faiss_search": 0.5},
                "_function_metrics": {"candidate_count": 3},
            },
            {"id": 0},
            {"id": 2},
        ]

    async def warmup(self) -> None:
        return None


class FailingProvider(DummyProvider):
    async def invoke(self, payload) -> list[dict]:
        raise RuntimeError("FC unavailable")


class DummyPlanner:
    def plan(self, qps: float, force_faas: bool | None = None) -> OffloadPlan:
        return OffloadPlan(
            mode="faas",
            reason="test",
            query_qps=qps,
            candidate_k=3,
            warm_function_target=0,
            estimated_function_cost=0.0,
        )


class DummyWarmupManager:
    def observe_query(self, qps: float, warm_target: int) -> None:
        return None


def search_config() -> SearchConfig:
    return SearchConfig(
        hnsw=HNSWConfig(
            default_k=10,
            candidate_k=120,
            hnsw_index_path="data/test/index/full/full_hnsw.bin",
            hnsw_m=32,
            hnsw_ef_construction=200,
            hnsw_ef_search=80,
        ),
        pipeline=PipelineConfig(local_search_workers=1, faas_invoke_workers=1, rerank_workers=1),
        offload_qps_threshold=20.0,
        force_faas=False,
    )


def test_search_service_reranks_remote_pq_candidates_on_vm() -> None:
    async def scenario() -> None:
        vectors = VectorStore.synthetic(dimension=4, count=10)
        provider = DummyProvider()
        metrics = RuntimeMetrics()
        service = SearchService(
            vectors=vectors,
            local_index=None,
            provider=provider,
            warmup_manager=DummyWarmupManager(),
            planner=DummyPlanner(),
            metrics=metrics,
            config=search_config(),
        )
        try:
            result = await service.search(query=vectors.get(0), k=2, candidate_k=3, ef_search=80)
        finally:
            service.close()

        assert provider.payload_json == {
            "request_id": result.request_id,
            "query": vectors.get(0).tolist(),
            "candidate_k": 3,
            "ef_search": 80,
        }
        assert result.results[0].id == 0
        assert len(result.results) == 2
        assert result.to_json()["cold_start_id"] == "cold-1"
        assert result.to_json()["index_loaded_at"] == 123.5
        assert result.to_json()["function_timings_ms"]["handler_total"] == 1.0
        assert result.to_json()["function_timings_ms"]["remote_queue_estimate"] >= 0.0
        assert result.to_json()["function_metrics"] == {"candidate_count": 3}
        snapshot = metrics.snapshot()
        assert snapshot["request_inflight"] == 0
        assert snapshot["faas_candidate_inflight"] == 0
        assert snapshot["faas_candidate_samples"] == 1
        assert snapshot["faas_error_count_total"] == 0
        assert snapshot["rerank_samples"] == 1

    asyncio.run(scenario())


def test_search_service_releases_metrics_after_faas_error() -> None:
    async def scenario() -> None:
        vectors = VectorStore.synthetic(dimension=4, count=10)
        metrics = RuntimeMetrics()
        service = SearchService(
            vectors=vectors,
            local_index=None,
            provider=FailingProvider(),
            warmup_manager=DummyWarmupManager(),
            planner=DummyPlanner(),
            metrics=metrics,
            config=search_config(),
        )
        try:
            with pytest.raises(RuntimeError, match="FC unavailable"):
                await service.search(query=vectors.get(0), k=2, candidate_k=3, ef_search=80)
        finally:
            service.close()

        snapshot = metrics.snapshot()
        assert snapshot["request_inflight"] == 0
        assert snapshot["faas_candidate_inflight"] == 0
        assert snapshot["faas_error_count_total"] == 1
        assert snapshot["faas_consecutive_errors"] == 1
        assert snapshot["rerank_samples"] == 0

    asyncio.run(scenario())
