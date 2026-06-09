"""FastAPI 路由。

`/search` 负责把请求转换成 query vector，然后交给 search.SearchService。
`/stats` 返回当前 QPS、阶段延迟和 FaaS 预热状态，便于观察 offload 策略。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from urllib.error import HTTPError, URLError

from search.service import SearchService
from server.schemas import SearchRequest
from vectors.vector_store import VectorStore


def create_router(search_service: SearchService, vector_store: VectorStore) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "vectors": vector_store.size,
            "dimension": vector_store.dimension,
        }

    @router.get("/stats")
    async def stats() -> dict:
        return {
            "metrics": search_service.metrics.snapshot(),
            "prewarm": search_service.warmup_manager.snapshot(),
        }

    @router.post("/search")
    async def search(request: SearchRequest) -> dict:
        query = _resolve_query(request, vector_store)
        try:
            result = await search_service.search(
                query=query,
                k=request.k,
                request_id=request.request_id,
                use_faas=request.use_faas,
                candidate_k=request.candidate_k,
                ef_search=request.ef_search,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            raise HTTPException(status_code=502, detail=f"faas invoke failed: {exc}") from exc
        return result.to_json()

    return router


def _resolve_query(request: SearchRequest, vector_store: VectorStore) -> np.ndarray:
    if request.vector is not None:
        query = np.asarray(request.vector, dtype="float32")
        if query.shape != (vector_store.dimension,):
            raise HTTPException(status_code=400, detail=f"vector dimension must be {vector_store.dimension}")
        return query

    if request.query_id is None:
        raise HTTPException(status_code=400, detail="either vector or query_id is required")
    if request.query_id < 0 or request.query_id >= vector_store.size:
        raise HTTPException(status_code=400, detail="query_id out of range")
    return vector_store.get(request.query_id).copy()
