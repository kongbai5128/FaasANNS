"""两阶段搜索服务。

SearchService 根据 QPS 和请求参数选择本地 HNSW 或云函数 PQ 候选召回，拿到 candidate ids 后回到
VM 侧 VectorStore 做 exact rerank，并用线程池隔离本地搜索和精排计算。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from faas.payload import CandidateSearchPayload
from scaling.metrics import RuntimeMetrics
from scaling.planner import OffloadPlan, OffloadPlanner
from scaling.prewarm import WarmupManager
from search.hnsw import HNSWIndex
from utils.config import SearchConfig
from utils.timer import measure
from vectors.vector_store import ScoredVector, VectorStore


@dataclass(slots=True)
class SearchResult:
    request_id: str
    results: list[ScoredVector]
    plan: OffloadPlan
    timings: dict[str, float]

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "results": [asdict(item) for item in self.results],
            "plan": asdict(self.plan),
            "timings_ms": {key: round(value * 1000.0, 3) for key, value in self.timings.items()},
        }


class SearchService:
    def __init__(
        self,
        vectors: VectorStore,
        local_index: HNSWIndex,
        provider,
        warmup_manager: WarmupManager,
        planner: OffloadPlanner,
        metrics: RuntimeMetrics,
        config: SearchConfig,
    ):
        self.vectors = vectors
        self.local_index = local_index
        self.provider = provider
        self.warmup_manager = warmup_manager
        self.planner = planner
        self.metrics = metrics
        self.config = config
        self.local_search_executor = ThreadPoolExecutor(
            max_workers=config.pipeline.local_search_workers,
            thread_name_prefix="faasann-local-search",
        )
        self.rerank_executor = ThreadPoolExecutor(
            max_workers=config.pipeline.rerank_workers,
            thread_name_prefix="faasann-rerank",
        )

    def close(self) -> None:
        self.local_search_executor.shutdown(wait=True, cancel_futures=True)
        self.rerank_executor.shutdown(wait=True, cancel_futures=True)
        close_provider = getattr(self.provider, "close", None)
        if close_provider is not None:
            close_provider()

    async def search(
        self,
        query: np.ndarray,
        k: int | None = None,
        request_id: str | None = None,
        use_faas: bool | None = None,
        candidate_k: int | None = None,
        ef_search: int | None = None,
    ) -> SearchResult:
        total_start = time.perf_counter()

        # 请求没有显式传参时，使用配置里的默认 top-k、候选数量和 HNSW 搜索深度。
        request_id = request_id or uuid.uuid4().hex
        k = k or self.config.hnsw.default_k
        candidate_k = candidate_k or self.config.hnsw.candidate_k
        ef_search = ef_search or self.config.hnsw.hnsw_ef_search

        self.metrics.mark_query()

        # 根据当前 QPS 或请求级 use_faas 决定第一阶段候选召回走 VM 本地 HNSW 还是云函数 HNSW-PQ。
        timings: dict[str, float] = {}
        with measure() as plan_elapsed:
            # 主要获取plan.mode，它是当前请求的路由结果：local 表示 VM 本地 HNSW，faas 表示调用云函数 HNSW-PQ。
            plan = self.planner.plan(self.metrics.qps, force_faas=use_faas)
            # 每个请求都会调用一次：这里只更新 query 计数、最近 QPS 和建议 warm target。
            # 真正发送 warmup ping 的后台循环在 WarmupManager 里按 prewarm_check_seconds 周期执行。
            self.warmup_manager.observe_query(self.metrics.qps, plan.warm_function_target)
        timings["plan"] = plan_elapsed.seconds

        # 第一阶段只返回候选 id 和近似分数；云函数路径不会做 raw vector 精排。
        with measure() as candidate_elapsed:
            if plan.mode == "faas":
                candidates = await self._search_faas(request_id, query, candidate_k, ef_search, timings)
            else:
                candidates = await self._run_local_candidates(query, candidate_k, ef_search)
        timings["candidates"] = candidate_elapsed.seconds
        self.metrics.candidate_latency.record(candidate_elapsed.seconds)

        # 第二阶段始终在 VM 上用原始向量 exact rerank，最终只返回 top-k。
        with measure() as rerank_elapsed:
            results = await self._rerank(query, candidates, k)
        timings["rerank"] = rerank_elapsed.seconds
        self.metrics.rerank_latency.record(rerank_elapsed.seconds)
        timings["total"] = time.perf_counter() - total_start

        return SearchResult(request_id=request_id, results=results, plan=plan, timings=timings)

    async def _search_faas(
        self,
        request_id: str,
        query: np.ndarray,
        candidate_k: int,
        ef_search: int,
        timings: dict[str, float],
    ) -> list[dict]:
        start = time.perf_counter()
        payload = CandidateSearchPayload(
            request_id=request_id,
            query=query,
            candidate_k=candidate_k,
            ef_search=ef_search,
        )
        try:
            return await self.provider.invoke(payload)
        finally:
            timings["remote_invoke"] = time.perf_counter() - start

    async def _run_local_candidates(self, query: np.ndarray, candidate_k: int, ef_search: int) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.local_search_executor,
            self.local_index.search,
            query,
            candidate_k,
            ef_search,
        )

    async def _rerank(self, query: np.ndarray, candidates: list[dict], k: int) -> list[ScoredVector]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.rerank_executor,
            self.vectors.rerank,
            query,
            candidates,
            k,
        )
