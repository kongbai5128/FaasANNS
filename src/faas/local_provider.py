"""本地候选搜索模拟器。"""

from __future__ import annotations

import asyncio

from faas.payload import CandidateSearchPayload
from search.hnsw import HNSWIndex


class LocalFaaSProvider:
    def __init__(self, index: HNSWIndex):
        self.index = index

    async def invoke(self, payload: CandidateSearchPayload) -> list[dict]:
        return await asyncio.to_thread(
            self.index.search,
            payload.query,
            payload.candidate_k,
        )

    async def warmup(self) -> None:
        return None
