"""HNSW 候选索引。

这个文件实现第一版本地候选搜索。优先加载 data/full_hnsw.bin 这种 hnswlib 落盘索引；
如果索引文件或依赖缺失，直接报错，避免用不真实的后端掩盖环境问题。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import hnswlib  # type: ignore
except ImportError:
    hnswlib = None

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


class HNSWIndex:
    def __init__(
        self,
        vectors: np.ndarray,
        index_path: str | Path | None,
        m: int,
        ef_construction: int,
        ef_search: int,
    ):
        self.vectors = np.ascontiguousarray(vectors.astype("float32", copy=False))
        self.dimension = int(self.vectors.shape[1])
        self.default_ef_search = ef_search
        self.backend: str | None = None
        self.faiss_index = None
        self.hnswlib_index = None

        if index_path is not None:
            self._load_hnswlib(Path(index_path), ef_search)
            return

        if faiss is None:
            raise RuntimeError("faiss is required to build an in-memory HNSW index when no index_path is configured")

        index = faiss.IndexHNSWFlat(self.dimension, m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(self.vectors)
        self.faiss_index = index
        self.backend = "faiss"
        logger.info("built Faiss HNSW index: vectors=%d dim=%d m=%d", self.vectors.shape[0], self.dimension, m)

    @property
    def uses_faiss(self) -> bool:
        return self.backend == "faiss"

    @property
    def uses_hnswlib(self) -> bool:
        return self.backend == "hnswlib"

    def search(self, query: np.ndarray, candidate_k: int, ef_search: int | None = None) -> list[dict]:
        candidate_k = min(candidate_k, self.vectors.shape[0])
        if candidate_k <= 0:
            return []
        query = query.astype("float32", copy=False)
        if self.hnswlib_index is not None:
            return self._search_hnswlib(query, candidate_k, ef_search)
        if self.faiss_index is not None:
            return self._search_faiss(query, candidate_k, ef_search)
        raise RuntimeError("HNSW index is not initialized")

    def _load_hnswlib(self, index_path: Path, ef_search: int) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"hnswlib index file not found: {index_path}")
        if hnswlib is None:
            raise RuntimeError(f"hnswlib is required to load configured index: {index_path}")

        self._validate_meta(index_path)
        index = hnswlib.Index(space="l2", dim=self.dimension)
        index.load_index(str(index_path), max_elements=self.vectors.shape[0])
        index.set_ef(ef_search)
        self.hnswlib_index = index
        self.backend = "hnswlib"
        logger.info("loaded hnswlib index from %s", index_path)

    def _validate_meta(self, index_path: Path) -> None:
        meta_path = index_path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"hnswlib metadata file not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "vector_count" not in meta or "dimension" not in meta:
            raise ValueError(f"hnswlib metadata must contain vector_count and dimension: {meta_path}")
        meta_count = int(meta["vector_count"])
        meta_dim = int(meta["dimension"])
        if meta_count != self.vectors.shape[0] or meta_dim != self.dimension:
            raise ValueError(
                "HNSW index metadata does not match loaded raw vectors: "
                f"meta_count={meta_count}, vector_count={self.vectors.shape[0]}, "
                f"meta_dim={meta_dim}, dimension={self.dimension}"
            )

    def _search_hnswlib(self, query: np.ndarray, candidate_k: int, ef_search: int | None) -> list[dict]:
        if ef_search is not None:
            self.hnswlib_index.set_ef(ef_search)
        ids, distances = self.hnswlib_index.knn_query(query.reshape(1, -1), k=candidate_k)
        return [
            {"id": int(vector_id), "approx_score": float(score)}
            for vector_id, score in zip(ids[0], distances[0])
            if vector_id >= 0
        ]

    def _search_faiss(self, query: np.ndarray, candidate_k: int, ef_search: int | None) -> list[dict]:
        if ef_search is not None:
            self.faiss_index.hnsw.efSearch = ef_search
        distances, ids = self.faiss_index.search(query.reshape(1, -1), candidate_k)
        return [
            {"id": int(vector_id), "approx_score": float(score)}
            for vector_id, score in zip(ids[0], distances[0])
            if vector_id >= 0
        ]
