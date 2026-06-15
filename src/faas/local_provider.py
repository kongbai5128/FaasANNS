"""本地候选搜索模拟器。"""

from __future__ import annotations

from faas.payload import CandidateSearchPayload
from search.hnsw import HNSWIndex


class LocalFaaSProvider:
    def __init__(self, index: HNSWIndex):
        self.index = index

    async def invoke(self, payload: CandidateSearchPayload) -> list[dict]:
        return self.index.search(
            payload.query,
            payload.candidate_k,
            payload.ef_search,
        )

    async def warmup(self) -> None:
        return None
