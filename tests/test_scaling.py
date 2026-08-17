"""扩容 planner 和预热管理器单元测试。"""

from __future__ import annotations

import asyncio

from scaling.planner import OffloadPlanner
from scaling.metrics import RuntimeMetrics
from scaling.prewarm import WarmupManager
from utils.config import HNSWConfig, PipelineConfig, ScalingConfig, SearchConfig


def search_config(**overrides) -> SearchConfig:
    hnsw = HNSWConfig(
        default_k=10,
        candidate_k=120,
        hnsw_index_path="data/test/index/full/full_hnsw.bin",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=80,
    )
    data = {
        "hnsw": hnsw,
        "pipeline": PipelineConfig(local_search_workers=2, faas_invoke_workers=100, rerank_workers=4),
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


def test_adaptive_planner_routes_only_overflow_to_faas() -> None:
    metrics = RuntimeMetrics()
    planner = OffloadPlanner(
        search_config(),
        scaling_config(
            adaptive_offload_enabled=True,
            adaptive_target_local_utilization=0.5,
            local_candidate_ms=100.0,
        ),
        metrics=metrics,
    )

    modes = [planner.plan(qps=50.0).mode for _ in range(100)]

    # Two local workers at 100 ms and 50% target utilization provide 10 QPS,
    # so the remaining 40/50 requests should use FaaS.
    assert planner.snapshot()["faas_ratio"] == 0.8
    assert 79 <= modes.count("faas") <= 80


def test_adaptive_planner_spills_when_local_slots_are_busy() -> None:
    metrics = RuntimeMetrics()
    planner = OffloadPlanner(
        search_config(),
        scaling_config(adaptive_offload_enabled=True),
        metrics=metrics,
    )
    metrics.candidate_started("local")
    metrics.candidate_started("local")
    try:
        plan = planner.plan(qps=1.0)
    finally:
        metrics.candidate_finished("local", 0.01, success=True)
        metrics.candidate_finished("local", 0.01, success=True)

    assert plan.mode == "faas"
    assert "pressure-adjusted proportional route" in plan.reason
    assert plan.faas_ratio == 1.0


def test_adaptive_planner_opens_circuit_after_faas_error(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("scaling.planner.time.monotonic", lambda: now[0])
    metrics = RuntimeMetrics()
    planner = OffloadPlanner(
        search_config(),
        scaling_config(
            adaptive_offload_enabled=True,
            local_candidate_ms=100.0,
            adaptive_circuit_breaker_seconds=30.0,
        ),
        metrics=metrics,
    )
    metrics.candidate_started("faas")
    metrics.candidate_finished("faas", 0.2, success=False)

    plan = planner.plan(qps=100.0)

    assert plan.mode == "local"
    assert "circuit open" in plan.reason
    assert planner.snapshot()["circuit_open"] is True

    now[0] = 131.0
    recovery_modes = [planner.plan(qps=100.0).mode for _ in range(10)]

    assert planner.snapshot()["circuit_open"] is False
    assert planner.snapshot()["faas_ratio"] == 0.1
    assert recovery_modes.count("faas") == 1


def test_runtime_metrics_track_paths_and_release_inflight() -> None:
    metrics = RuntimeMetrics()
    metrics.request_started()
    metrics.candidate_started("local")
    metrics.candidate_finished("local", 0.04, success=True)
    metrics.rerank_started()
    metrics.rerank_finished(0.01, success=True)
    metrics.request_finished()

    snapshot = metrics.snapshot()

    assert snapshot["request_inflight"] == 0
    assert snapshot["local_candidate_inflight"] == 0
    assert snapshot["local_candidate_p95_ms"] == 40.0
    assert snapshot["rerank_p95_ms"] == 10.0


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
