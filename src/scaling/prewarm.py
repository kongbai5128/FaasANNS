"""FaaS 预热管理器。

阿里云 FC 已经负责真实实例并发、排队和扩缩容，因此本地服务器不再维护 function slot 队列。
这个模块只做一件事：根据最近 QPS 和 planner 给出的 warm target，异步发送 warmup ping，尽量降低
后续请求遇到冷启动的概率。所有真实查询请求都会直接调用云函数 HTTP endpoint。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time

from utils.config import ScalingConfig

logger = logging.getLogger(__name__)


class WarmupManager:
    def __init__(self, provider, config: ScalingConfig):
        self.provider = provider
        self.config = config
        self._closed = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._query_counter = 0
        self._last_query_counter = 0
        self._recent_qps = 0.0
        self._warm_target = 0
        self._warmup_in_flight = 0
        self._warmup_requests = 0
        self._last_warmup_at = 0.0
        self._last_warmup_seconds = 0.0
        self._last_error: str | None = None

    async def start(self) -> None:
        if not self.config.enable_prewarm:
            return
        self._task = asyncio.create_task(self._prewarm_loop())
        asyncio.create_task(self.trigger_warmup(1))

    async def close(self) -> None:
        self._closed.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def observe_query(self, qps: float, warm_target: int) -> None:
        self._query_counter += 1
        self._recent_qps = qps
        self._warm_target = max(0, min(self.config.max_warm_functions, warm_target))

    async def trigger_warmup(self, count: int = 1) -> None:
        if not self.config.enable_prewarm:
            return

        count = max(0, min(count, self.config.max_warm_functions))
        if count <= 0:
            return

        async with self._lock:
            room = max(0, self.config.max_warm_functions - self._warmup_in_flight)
            count = min(count, room)
            if count <= 0:
                return
            self._warmup_in_flight += count

        start = time.perf_counter()
        try:
            results = await asyncio.gather(
                *(self.provider.warmup() for _ in range(count)),
                return_exceptions=True,
            )
            errors = [item for item in results if isinstance(item, Exception)]
            self._last_error = str(errors[0]) if errors else None
            if errors:
                logger.warning("warmup failed for %d/%d requests: %s", len(errors), count, errors[0])
        finally:
            elapsed = time.perf_counter() - start
            async with self._lock:
                self._warmup_in_flight -= count
                self._warmup_requests += count
                self._last_warmup_at = time.monotonic()
                self._last_warmup_seconds = elapsed

    def snapshot(self) -> dict:
        return {
            "enabled": self.config.enable_prewarm,
            "recent_qps": self._recent_qps,
            "warm_target": self._warm_target,
            "warmup_in_flight": self._warmup_in_flight,
            "warmup_requests": self._warmup_requests,
            "last_warmup_ms": round(self._last_warmup_seconds * 1000.0, 3),
            "last_error": self._last_error,
            "function_concurrency": self.config.function_concurrency,
            "max_warm_functions": self.config.max_warm_functions,
        }

    async def _prewarm_loop(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(self.config.prewarm_check_seconds)
            current = self._query_counter
            diff = current - self._last_query_counter
            self._last_query_counter = current

            recent_qps = max(
                self._recent_qps,
                diff / max(self.config.prewarm_check_seconds, 0.001),
            )
            if recent_qps <= 0:
                continue

            target = max(self._warm_target, self._estimate_target_from_qps(recent_qps))
            target = min(target, self.config.max_warm_functions)
            if target <= 0:
                continue

            now = time.monotonic()
            cooldown = max(1.0, self.config.load_index_timeout_seconds)
            if now - self._last_warmup_at < cooldown:
                continue

            asyncio.create_task(self.trigger_warmup(target))

    def _estimate_target_from_qps(self, qps: float) -> int:
        concurrency = max(1, self.config.function_concurrency)
        remote_seconds = max(self.config.remote_candidate_ms, 1.0) / 1000.0
        return max(1, math.ceil(qps * remote_seconds / concurrency))
