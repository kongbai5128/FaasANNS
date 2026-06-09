"""local/FaaS 执行计划与 warm target 规划。

OffloadPlanner 根据实时 QPS、配置阈值和请求级 override 决定候选搜索走本地 HNSW 还是 FaaS。
它还估算当前建议预热多少个 FC 实例，供 WarmupManager 发送 warmup ping。
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.config import ScalingConfig, SearchConfig


@dataclass(slots=True)
class OffloadPlan:
    mode: str
    reason: str
    query_qps: float
    candidate_k: int
    warm_function_target: int
    estimated_function_cost: float


class OffloadPlanner:
    def __init__(self, search: SearchConfig, scaling: ScalingConfig):
        self.search = search
        self.scaling = scaling

    def plan(self, qps: float, force_faas: bool | None = None) -> OffloadPlan:
        use_faas = self.search.force_faas
        reason = "configured force_faas"
        if force_faas is not None:
            use_faas = force_faas
            reason = "request override"
        elif qps >= self.search.offload_qps_threshold:
            use_faas = True
            reason = f"qps {qps:.2f} >= threshold {self.search.offload_qps_threshold:.2f}"
        else:
            reason = f"qps {qps:.2f} < threshold {self.search.offload_qps_threshold:.2f}"

        return OffloadPlan(
            mode="faas" if use_faas else "local",
            reason=reason,
            query_qps=qps,
            candidate_k=self.search.hnsw.candidate_k,
            warm_function_target=self.warm_target(qps),
            estimated_function_cost=self.estimate_function_cost(self.scaling.remote_candidate_ms / 1000.0),
        )

    def warm_target(self, qps: float) -> int:
        concurrency = max(1, self.scaling.function_concurrency)
        remote_seconds = max(self.scaling.remote_candidate_ms, 1.0) / 1000.0
        needed = int(qps * remote_seconds / concurrency) + 1
        return max(0, min(self.scaling.max_warm_functions, needed))

    def estimate_function_cost(self, duration_seconds: float) -> float:
        memory_gb = self.scaling.function_memory_mb / 1024.0
        return memory_gb * duration_seconds * self.scaling.cost_per_gb_second
