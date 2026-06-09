"""FaaS PQ 候选搜索请求载荷。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CandidateSearchPayload:
    request_id: str
    query: np.ndarray
    candidate_k: int

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "query": self.query.astype("float32").tolist(),
            "candidate_k": self.candidate_k,
        }
