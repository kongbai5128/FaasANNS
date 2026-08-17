"""Local/FaaS execution planning with optional adaptive proportional routing."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from scaling.metrics import RuntimeMetrics
from utils.config import ScalingConfig, SearchConfig


@dataclass(slots=True)
class OffloadPlan:
    mode: str
    reason: str
    query_qps: float
    candidate_k: int
    warm_function_target: int
    estimated_function_cost: float
    faas_ratio: float = 0.0


class OffloadPlanner:
    def __init__(
        self,
        search: SearchConfig,
        scaling: ScalingConfig,
        metrics: RuntimeMetrics | None = None,
    ):
        self.search = search
        self.scaling = scaling
        self.metrics = metrics
        self._faas_ratio = 0.0
        self._desired_faas_ratio = 0.0
        self._routing_credit = 0.0
        self._last_control_at = 0.0
        self._circuit_open_until = 0.0
        self._last_seen_faas_outcomes = 0
        self._last_seen_faas_errors = 0
        self._local_safe_qps = 0.0
        self._remote_degraded = False
        self._last_reason = "not evaluated"
        self._route_count = 0
        self._faas_route_count = 0
        self._recent_routes: deque[tuple[float, bool]] = deque()

    def plan(self, qps: float, force_faas: bool | None = None) -> OffloadPlan:
        metrics: dict = {}
        if self.scaling.adaptive_offload_enabled and self.metrics is not None:
            now = time.monotonic()
            interval = max(0.05, self.scaling.adaptive_control_interval_seconds)
            include_latencies = (
                self._last_control_at == 0.0
                or now - self._last_control_at >= interval
            )
            metrics = self.metrics.routing_snapshot(include_latencies=include_latencies)

        if force_faas is not None:
            use_faas = force_faas
            ratio = 1.0 if use_faas else 0.0
            reason = "request override"
        elif self.search.force_faas:
            use_faas = True
            ratio = 1.0
            reason = "configured force_faas"
        elif self.scaling.adaptive_offload_enabled:
            self._update_adaptive_ratio(qps, metrics)
            use_faas, dispatch_reason, ratio = self._adaptive_dispatch(metrics)
            reason = (
                f"adaptive {dispatch_reason}; target_ratio={self._faas_ratio:.3f}; "
                f"dispatch_ratio={ratio:.3f}; "
                f"local_safe_qps={self._local_safe_qps:.2f}"
            )
        elif qps >= self.search.offload_qps_threshold:
            use_faas = True
            ratio = 1.0
            reason = f"qps {qps:.2f} >= threshold {self.search.offload_qps_threshold:.2f}"
        else:
            use_faas = False
            ratio = 0.0
            reason = f"qps {qps:.2f} < threshold {self.search.offload_qps_threshold:.2f}"

        self._last_reason = reason
        self._route_count += 1
        if use_faas:
            self._faas_route_count += 1
        now = time.monotonic()
        self._recent_routes.append((now, use_faas))
        self._trim_recent_routes(now)
        faas_qps = qps * ratio
        return OffloadPlan(
            mode="faas" if use_faas else "local",
            reason=reason,
            query_qps=qps,
            candidate_k=self.search.hnsw.candidate_k,
            warm_function_target=self.warm_target(faas_qps),
            estimated_function_cost=self.estimate_function_cost(
                self.scaling.remote_candidate_ms / 1000.0
            ),
            faas_ratio=ratio,
        )

    def warm_target(self, faas_qps: float) -> int:
        if faas_qps <= 0:
            return 0
        concurrency = max(1, self.scaling.function_concurrency)
        remote_seconds = max(self.scaling.remote_candidate_ms, 1.0) / 1000.0
        needed = max(1, math.ceil(faas_qps * remote_seconds / concurrency))
        return min(self.scaling.max_warm_functions, needed)

    def estimate_function_cost(self, duration_seconds: float) -> float:
        memory_gb = self.scaling.function_memory_mb / 1024.0
        return memory_gb * duration_seconds * self.scaling.cost_per_gb_second

    def snapshot(self) -> dict:
        now = time.monotonic()
        self._trim_recent_routes(now)
        recent_count = len(self._recent_routes)
        recent_faas_count = sum(1 for _, use_faas in self._recent_routes if use_faas)
        return {
            "adaptive_enabled": self.scaling.adaptive_offload_enabled,
            "faas_ratio": round(self._faas_ratio, 6),
            "desired_faas_ratio": round(self._desired_faas_ratio, 6),
            "local_safe_qps": round(self._local_safe_qps, 3),
            "remote_degraded": self._remote_degraded,
            "circuit_open": now < self._circuit_open_until,
            "circuit_remaining_seconds": round(max(0.0, self._circuit_open_until - now), 3),
            "route_count": self._route_count,
            "faas_route_count": self._faas_route_count,
            "cumulative_faas_ratio": round(
                self._faas_route_count / self._route_count if self._route_count else 0.0,
                6,
            ),
            "recent_route_count_30s": recent_count,
            "recent_faas_ratio_30s": round(
                recent_faas_count / recent_count if recent_count else 0.0,
                6,
            ),
            "last_reason": self._last_reason,
        }

    def _trim_recent_routes(self, now: float) -> None:
        cutoff = now - 30.0
        while self._recent_routes and self._recent_routes[0][0] < cutoff:
            self._recent_routes.popleft()

    def _update_adaptive_ratio(self, qps: float, metrics: dict) -> None:
        now = time.monotonic()
        self._update_circuit_breaker(now, metrics)
        if now < self._circuit_open_until:
            self._faas_ratio = 0.0
            self._desired_faas_ratio = 0.0
            self._last_control_at = now
            return

        interval = max(0.05, self.scaling.adaptive_control_interval_seconds)
        if self._last_control_at and now - self._last_control_at < interval:
            return

        local_ms = self._observed_or_configured_local_ms(metrics)
        local_workers = max(1, self.search.pipeline.local_search_workers)
        utilization = _clamp(self.scaling.adaptive_target_local_utilization, 0.05, 1.0)
        self._local_safe_qps = local_workers * 1000.0 * utilization / max(local_ms, 0.1)

        desired = 0.0 if qps <= 0 else max(0.0, (qps - self._local_safe_qps) / qps)
        local_inflight = int(metrics.get("local_candidate_inflight", 0))
        if local_inflight > local_workers:
            queue_ratio = (local_inflight - local_workers) / max(local_inflight, 1)
            desired = max(desired, min(1.0, queue_ratio))

        min_samples = max(1, self.scaling.adaptive_min_latency_samples)
        remote_samples = int(metrics.get("faas_candidate_samples", 0))
        remote_p95 = float(metrics.get("faas_candidate_p95_ms", 0.0))
        self._remote_degraded = (
            remote_samples >= min_samples
            and remote_p95 > self.scaling.adaptive_remote_p95_limit_ms
        )
        if self._remote_degraded:
            desired = min(
                desired,
                max(0.0, self._faas_ratio - self.scaling.adaptive_ratio_step_down),
            )

        rerank_workers = max(1, self.search.pipeline.rerank_workers)
        if int(metrics.get("rerank_inflight", 0)) > rerank_workers:
            desired = min(desired, self._faas_ratio)

        max_ratio = _clamp(self.scaling.adaptive_max_faas_ratio, 0.0, 1.0)
        desired = _clamp(desired, 0.0, max_ratio)
        self._desired_faas_ratio = desired
        self._apply_slew_limit(desired)
        self._last_control_at = now

    def _update_circuit_breaker(self, now: float, metrics: dict) -> None:
        outcomes = int(metrics.get("faas_outcome_count_total", 0))
        errors = int(metrics.get("faas_error_count_total", 0))
        has_new_outcomes = outcomes > self._last_seen_faas_outcomes
        has_new_errors = errors > self._last_seen_faas_errors
        if has_new_outcomes:
            error_rate = float(metrics.get("faas_error_rate", 0.0))
            consecutive = int(metrics.get("faas_consecutive_errors", 0))
            if has_new_errors and (
                consecutive > 0
                or error_rate >= self.scaling.adaptive_faas_error_rate_threshold
            ):
                self._circuit_open_until = now + max(
                    0.0, self.scaling.adaptive_circuit_breaker_seconds
                )
            self._last_seen_faas_outcomes = outcomes
            self._last_seen_faas_errors = errors

    def _observed_or_configured_local_ms(self, metrics: dict) -> float:
        samples = int(metrics.get("local_candidate_samples", 0))
        if samples >= max(1, self.scaling.adaptive_min_latency_samples):
            return max(0.1, float(metrics.get("local_candidate_p95_ms", 0.0)))
        return max(0.1, self.scaling.local_candidate_ms)

    def _apply_slew_limit(self, desired: float) -> None:
        if self._last_control_at == 0.0:
            self._faas_ratio = desired
            return
        difference = desired - self._faas_ratio
        if abs(difference) < max(0.0, self.scaling.adaptive_ratio_hysteresis):
            return
        if difference > 0:
            self._faas_ratio += min(difference, max(0.0, self.scaling.adaptive_ratio_step_up))
        else:
            self._faas_ratio -= min(-difference, max(0.0, self.scaling.adaptive_ratio_step_down))
        self._faas_ratio = _clamp(
            self._faas_ratio,
            0.0,
            _clamp(self.scaling.adaptive_max_faas_ratio, 0.0, 1.0),
        )

    def _adaptive_dispatch(self, metrics: dict) -> tuple[bool, str, float]:
        now = time.monotonic()
        if now < self._circuit_open_until:
            return False, "FC circuit open", 0.0

        local_slots = max(1, self.search.pipeline.local_search_workers)
        faas_slots = max(1, self.scaling.function_concurrency)
        local_inflight = int(metrics.get("local_candidate_inflight", 0))
        faas_inflight = int(metrics.get("faas_candidate_inflight", 0))
        rerank_workers = max(1, self.search.pipeline.rerank_workers)
        rerank_queued = int(metrics.get("rerank_inflight", 0)) > rerank_workers

        local_pressure = local_inflight / local_slots
        faas_pressure = faas_inflight / faas_slots
        if self._remote_degraded and local_pressure < 2.0:
            return False, "FC P95 degraded", 0.0
        effective_ratio = self._faas_ratio
        if not rerank_queued:
            total_pressure = local_pressure + faas_pressure
            if total_pressure > 0 and (local_pressure >= 1.0 or faas_pressure >= 1.0):
                pressure_ratio = local_pressure / total_pressure
                effective_ratio = max(effective_ratio, pressure_ratio)
                if faas_pressure > local_pressure:
                    effective_ratio = min(effective_ratio, pressure_ratio)

        effective_ratio = _clamp(
            effective_ratio,
            0.0,
            _clamp(self.scaling.adaptive_max_faas_ratio, 0.0, 1.0),
        )
        self._routing_credit += effective_ratio
        if self._routing_credit + 1e-12 >= 1.0:
            self._routing_credit -= 1.0
            return True, "pressure-adjusted proportional route", effective_ratio
        return False, "pressure-adjusted proportional route", effective_ratio


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
