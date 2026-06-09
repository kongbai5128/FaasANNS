"""扩容 planner 和预热管理器单元测试。"""

from __future__ import annotations

import asyncio

from scaling.planner import OffloadPlanner
from scaling.prewarm import WarmupManager
from utils.config import HNSWConfig, PipelineConfig, ScalingConfig, SearchConfig


def search_config(**overrides) -> SearchConfig:
    hnsw = HNSWConfig(
        default_k=10,
        candidate_k=120,
        hnsw_index_path="data/index/full/full_hnsw.bin",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=80,
    )
    data = {
        "hnsw": hnsw,
        "pipeline": PipelineConfig(local_search_workers=2, rerank_workers=4),
        "offload_qps_threshold": 20.0,
        "force_faas": False,
    }
    data.update(overrides)
    return SearchConfig(**data)


def scaling_config(**overrides) -> ScalingConfig:
    data = {
        "prewarm_check_seconds": 0.1,
        "load_index_timeout_seconds": 3.0,
        "enable_prewarm": True,
        "local_candidate_ms": 8.0,
        "remote_candidate_ms": 20.0,
        "function_concurrency": 1,
        "max_warm_functions": 32,
        "function_memory_mb": 512,
        "cost_per_gb_second": 0.0000167,
    }
    data.update(overrides)
    return ScalingConfig(**data)


def test_planner_uses_local_below_threshold() -> None:
    planner = OffloadPlanner(search_config(offload_qps_threshold=10.0), scaling_config())
    assert planner.plan(qps=1.0).mode == "local"


def test_planner_uses_faas_above_threshold() -> None:
    planner = OffloadPlanner(search_config(offload_qps_threshold=10.0), scaling_config())
    assert planner.plan(qps=20.0).mode == "faas"


class DummyProvider:
    def __init__(self) -> None:
        self.warmup_count = 0

    async def invoke(self, payload) -> list[dict]:
        return []

    async def warmup(self) -> None:
        self.warmup_count += 1
        return None


def test_warmup_manager_sends_warmup_without_slots() -> None:
    async def scenario() -> None:
        provider = DummyProvider()
        manager = WarmupManager(
            provider=provider,
            config=scaling_config(enable_prewarm=True, max_warm_functions=4),
        )

        await manager.trigger_warmup(2)

        assert provider.warmup_count == 2
        assert manager.snapshot()["warmup_requests"] == 2

    asyncio.run(scenario())
