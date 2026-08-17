"""运行时指标统计。

这里提供滑动窗口 QPS 和阶段延迟统计。SearchService 会在每次查询时更新这些指标，
OffloadPlanner 和 WarmupManager 使用这些指标判断是否切换到 FaaS 或提前预热。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


class QPSWindow:
    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def mark(self, count: int = 1) -> None:
        now = time.monotonic()
        for _ in range(count):
            self._timestamps.append(now)
        self._trim(now)

    def qps(self) -> float:
        now = time.monotonic()
        self._trim(now)
        return len(self._timestamps) / self.window_seconds

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class LatencyTracker:
    def __init__(self, max_samples: int = 1024, ewma_alpha: float = 0.2):
        self.samples: deque[float] = deque(maxlen=max_samples)
        self.ewma_alpha = ewma_alpha
        self._ewma_seconds: float | None = None

    def record(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self.samples.append(seconds)
        if self._ewma_seconds is None:
            self._ewma_seconds = seconds
        else:
            self._ewma_seconds += self.ewma_alpha * (seconds - self._ewma_seconds)

    def average_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples) * 1000.0

    def percentile_ms(self, quantile: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
        return ordered[index] * 1000.0

    def ewma_ms(self) -> float:
        return (self._ewma_seconds or 0.0) * 1000.0

    @property
    def count(self) -> int:
        return len(self.samples)


class OutcomeWindow:
    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self._outcomes: deque[tuple[float, bool]] = deque()
        self._error_count = 0

    def record(self, success: bool) -> None:
        now = time.monotonic()
        self._outcomes.append((now, success))
        if not success:
            self._error_count += 1
        self._trim(now)

    def snapshot(self) -> tuple[int, float]:
        now = time.monotonic()
        self._trim(now)
        count = len(self._outcomes)
        return count, self._error_count / count if count else 0.0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._outcomes and self._outcomes[0][0] < cutoff:
            _, success = self._outcomes.popleft()
            if not success:
                self._error_count -= 1


@dataclass
class RuntimeMetrics:
    qps_window: QPSWindow = field(default_factory=lambda: QPSWindow(window_seconds=1.0))
    candidate_latency: LatencyTracker = field(default_factory=LatencyTracker)
    local_candidate_latency: LatencyTracker = field(default_factory=LatencyTracker)
    faas_candidate_latency: LatencyTracker = field(default_factory=LatencyTracker)
    rerank_latency: LatencyTracker = field(default_factory=LatencyTracker)
    faas_outcomes: OutcomeWindow = field(default_factory=OutcomeWindow)
    request_inflight: int = 0
    local_candidate_inflight: int = 0
    faas_candidate_inflight: int = 0
    rerank_inflight: int = 0
    faas_outcome_count_total: int = 0
    faas_error_count_total: int = 0
    faas_consecutive_errors: int = 0

    def mark_query(self) -> None:
        self.qps_window.mark()

    def request_started(self) -> None:
        self.request_inflight += 1

    def request_finished(self) -> None:
        self.request_inflight = max(0, self.request_inflight - 1)

    def candidate_started(self, mode: str) -> None:
        if mode == "local":
            self.local_candidate_inflight += 1
        elif mode == "faas":
            self.faas_candidate_inflight += 1
        else:
            raise ValueError(f"unsupported candidate mode: {mode}")

    def candidate_finished(self, mode: str, seconds: float, *, success: bool) -> None:
        if mode == "local":
            self.local_candidate_inflight = max(0, self.local_candidate_inflight - 1)
            if success:
                self.local_candidate_latency.record(seconds)
        elif mode == "faas":
            self.faas_candidate_inflight = max(0, self.faas_candidate_inflight - 1)
            self.faas_candidate_latency.record(seconds)
            self.faas_outcomes.record(success)
            self.faas_outcome_count_total += 1
            if success:
                self.faas_consecutive_errors = 0
            else:
                self.faas_error_count_total += 1
                self.faas_consecutive_errors += 1
        else:
            raise ValueError(f"unsupported candidate mode: {mode}")

        if success:
            self.candidate_latency.record(seconds)

    def rerank_started(self) -> None:
        self.rerank_inflight += 1

    def rerank_finished(self, seconds: float, *, success: bool) -> None:
        self.rerank_inflight = max(0, self.rerank_inflight - 1)
        if success:
            self.rerank_latency.record(seconds)

    @property
    def qps(self) -> float:
        return self.qps_window.qps()

    def snapshot(self) -> dict:
        snapshot = {
            "qps": self.qps,
            "candidate_avg_ms": self.candidate_latency.average_ms(),
            "local_candidate_avg_ms": self.local_candidate_latency.average_ms(),
            "local_candidate_ewma_ms": self.local_candidate_latency.ewma_ms(),
            "local_candidate_p95_ms": self.local_candidate_latency.percentile_ms(0.95),
            "local_candidate_samples": self.local_candidate_latency.count,
            "faas_candidate_avg_ms": self.faas_candidate_latency.average_ms(),
            "faas_candidate_ewma_ms": self.faas_candidate_latency.ewma_ms(),
            "faas_candidate_p95_ms": self.faas_candidate_latency.percentile_ms(0.95),
            "faas_candidate_samples": self.faas_candidate_latency.count,
            "rerank_avg_ms": self.rerank_latency.average_ms(),
            "rerank_p95_ms": self.rerank_latency.percentile_ms(0.95),
            "rerank_samples": self.rerank_latency.count,
        }
        snapshot.update(self.routing_snapshot())
        return snapshot

    def routing_snapshot(self, *, include_latencies: bool = False) -> dict:
        faas_outcome_samples, faas_error_rate = self.faas_outcomes.snapshot()
        snapshot = {
            "request_inflight": self.request_inflight,
            "local_candidate_inflight": self.local_candidate_inflight,
            "faas_candidate_inflight": self.faas_candidate_inflight,
            "rerank_inflight": self.rerank_inflight,
            "faas_outcome_samples": faas_outcome_samples,
            "faas_error_rate": round(faas_error_rate, 6),
            "faas_outcome_count_total": self.faas_outcome_count_total,
            "faas_error_count_total": self.faas_error_count_total,
            "faas_consecutive_errors": self.faas_consecutive_errors,
        }
        if include_latencies:
            snapshot.update(
                {
                    "local_candidate_p95_ms": self.local_candidate_latency.percentile_ms(0.95),
                    "local_candidate_samples": self.local_candidate_latency.count,
                    "faas_candidate_p95_ms": self.faas_candidate_latency.percentile_ms(0.95),
                    "faas_candidate_samples": self.faas_candidate_latency.count,
                    "rerank_p95_ms": self.rerank_latency.percentile_ms(0.95),
                    "rerank_samples": self.rerank_latency.count,
                }
            )
        return snapshot
