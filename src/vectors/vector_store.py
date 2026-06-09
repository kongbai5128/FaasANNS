"""VM 侧 raw vector 存储与 exact rerank。

VectorStore 持有完整原始向量矩阵，并提供按 id 访问、L2 精确打分和最终 top-k rerank。
候选搜索可以在本地 HNSW 或云函数 PQ 中完成，但最终精排仍回到这里使用 raw vectors。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vectors.fvecs import read_fvecs


@dataclass(slots=True)
class ScoredVector:
    id: int
    score: float


class VectorStore:
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.vectors = np.ascontiguousarray(vectors.astype("float32", copy=False))
        self.dimension = int(self.vectors.shape[1])

    @classmethod
    def from_fvecs(cls, path: str, dimension: int, max_vectors: int | None = None) -> "VectorStore":
        return cls(read_fvecs(path, dimension=dimension, max_vectors=max_vectors))

    @classmethod
    def synthetic(cls, dimension: int, count: int = 10000) -> "VectorStore":
        ids = np.arange(count, dtype=np.float32)[:, None]
        dims = np.arange(dimension, dtype=np.float32)[None, :]
        vectors = ((ids + 1.0) * (dims + 3.0)) % 997.0
        return cls((vectors / 997.0).astype("float32"))

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def get(self, vector_id: int) -> np.ndarray:
        return self.vectors[vector_id]

    def l2_scores(self, query: np.ndarray, ids: np.ndarray) -> np.ndarray:
        subset = self.vectors[ids]
        diff = subset - query.reshape(1, -1)
        return np.sum(diff * diff, axis=1)

    def rerank(self, query: np.ndarray, candidates: list[dict], k: int) -> list[ScoredVector]:
        if query.shape[-1] != self.dimension:
            raise ValueError(f"query dimension={query.shape[-1]} does not match store dimension={self.dimension}")
        if not candidates:
            return []

        seen: dict[int, None] = {}
        for item in candidates:
            vector_id = int(item["id"])
            if 0 <= vector_id < self.size:
                seen.setdefault(vector_id, None)

        ids = np.array(list(seen.keys()), dtype=np.int64)
        scores = self.l2_scores(query.astype("float32", copy=False), ids)
        order = np.argsort(scores)[:k]
        return [ScoredVector(id=int(ids[i]), score=float(scores[i])) for i in order]
