"""Local FaaS provider tests."""

from __future__ import annotations

import asyncio

import numpy as np

from faas.local_provider import LocalFaaSProvider
from faas.payload import CandidateSearchPayload


class DummyIndex:
    def search(self, query, candidate_k, ef_search=None):
        return [{"id": int(query[0]), "candidate_k": candidate_k, "ef_search": ef_search}]


def test_local_provider_invokes_index_search() -> None:
    async def scenario() -> None:
        provider = LocalFaaSProvider(DummyIndex())
        results = await provider.invoke(
            CandidateSearchPayload(
                request_id="r1",
                query=np.array([3], dtype="float32"),
                candidate_k=5,
            )
        )

        assert results == [{"id": 3, "candidate_k": 5, "ef_search": None}]

    asyncio.run(scenario())
