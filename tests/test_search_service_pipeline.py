"""SearchService PQ offload pipeline tests."""

from __future__ import annotations

import asyncio

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
        return [{"id": 3}, {"id": 0}, {"id": 2}]

    async def warmup(self) -> None:
        return None


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
            hnsw_index_path="data/index/full/full_hnsw.bin",
            hnsw_m=32,
            hnsw_ef_construction=200,
            hnsw_ef_search=80,
        ),
        pipeline=PipelineConfig(local_search_workers=1, rerank_workers=1),
        offload_qps_threshold=20.0,
        force_faas=False,
    )


def test_search_service_reranks_remote_pq_candidates_on_vm() -> None:
    async def scenario() -> None:
        vectors = VectorStore.synthetic(dimension=4, count=10)
        provider = DummyProvider()
        service = SearchService(
            vectors=vectors,
            local_index=None,
            provider=provider,
            warmup_manager=DummyWarmupManager(),
            planner=DummyPlanner(),
            metrics=RuntimeMetrics(),
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
        }
        assert result.results[0].id == 0
        assert len(result.results) == 2

    asyncio.run(scenario())
