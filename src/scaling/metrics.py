"""运行时指标统计。

这里提供滑动窗口 QPS 和阶段延迟统计。SearchService 会在每次查询时更新这些指标，
OffloadPlanner 和 WarmupManager 使用这些指标判断是否切换到 FaaS 或提前预热。
"""

from __future__ import annotations

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
    def __init__(self, max_samples: int = 1024):
        self.samples: deque[float] = deque(maxlen=max_samples)

    def record(self, seconds: float) -> None:
        self.samples.append(seconds)

    def average_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples) * 1000.0


@dataclass
class RuntimeMetrics:
    qps_window: QPSWindow = field(default_factory=lambda: QPSWindow(window_seconds=1.0))
    candidate_latency: LatencyTracker = field(default_factory=LatencyTracker)
    rerank_latency: LatencyTracker = field(default_factory=LatencyTracker)

    def mark_query(self) -> None:
        self.qps_window.mark()

    @property
    def qps(self) -> float:
        return self.qps_window.qps()

    def snapshot(self) -> dict:
        return {
            "qps": self.qps,
            "candidate_avg_ms": self.candidate_latency.average_ms(),
            "rerank_avg_ms": self.rerank_latency.average_ms(),
        }
