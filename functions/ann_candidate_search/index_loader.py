# -*- coding: utf-8 -*-
"""云函数侧 PQ 候选索引加载与第一阶段召回。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_ROOT = Path(os.environ.get("FAASANN_DATA_ROOT", "/mnt/faasann"))
PQ_DIR = DATA_ROOT / "index" / "full" / "pq"
PQ_META_PATH = Path(os.environ.get("FAASANN_PQ_META_PATH", str(PQ_DIR / "pq_meta.json")))
PQ_CODEBOOKS_PATH = Path(os.environ.get("FAASANN_PQ_CODEBOOKS_PATH", str(PQ_DIR / "pq_codebooks.npy")))
PQ_CODES_PATH = Path(os.environ.get("FAASANN_PQ_CODES_PATH", str(PQ_DIR / "pq_codes.npy")))
PQ_IDS_PATH = Path(os.environ.get("FAASANN_PQ_IDS_PATH", str(PQ_DIR / "pq_ids.npy")))

_load_lock = threading.Lock()
_state: "PQState | None" = None


@dataclass(slots=True)
class PQState:
    codebooks: np.ndarray
    codes: np.ndarray
    ids: np.ndarray
    dimension: int
    vector_count: int
    subspace_count: int
    codebook_size: int


def warmup() -> None:
    load_state()


def index_status() -> dict:
    state = _state
    return {
        "backend": "pq" if state else "not_loaded",
        "loaded": state is not None,
        "meta_available": PQ_META_PATH.exists(),
        "codebooks_available": PQ_CODEBOOKS_PATH.exists(),
        "codes_available": PQ_CODES_PATH.exists(),
        "ids_available": PQ_IDS_PATH.exists(),
        "meta_path": str(PQ_META_PATH),
        "codebooks_path": str(PQ_CODEBOOKS_PATH),
        "codes_path": str(PQ_CODES_PATH),
        "ids_path": str(PQ_IDS_PATH),
        "dimension": state.dimension if state else None,
        "vector_count": state.vector_count if state else None,
        "subspace_count": state.subspace_count if state else None,
        "codebook_size": state.codebook_size if state else None,
    }


def search(query: list[float], candidate_k: int) -> list[dict]:
    state = load_state()
    query_vector = np.asarray(query, dtype="float32")
    if query_vector.shape != (state.dimension,):
        raise ValueError(f"query dimension must be {state.dimension}, got {query_vector.shape}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    candidate_k = min(candidate_k, state.vector_count)
    scores = _pq_scores(state, query_vector)
    top = np.argpartition(scores, candidate_k - 1)[:candidate_k]
    top = top[np.argsort(scores[top])]
    return [
        {"id": int(state.ids[row]), "approx_score": float(scores[row])}
        for row in top
    ]


def load_state() -> PQState:
    global _state
    if _state is not None:
        return _state

    with _load_lock:
        if _state is not None:
            return _state

        meta = _read_meta()
        codebooks = np.asarray(np.load(PQ_CODEBOOKS_PATH), dtype="float32")
        codes = np.load(PQ_CODES_PATH, mmap_mode="r")
        ids = np.load(PQ_IDS_PATH, mmap_mode="r")
        _validate_arrays(meta, codebooks, codes, ids)

        _state = PQState(
            codebooks=codebooks,
            codes=codes,
            ids=ids,
            dimension=int(meta["dimension"]),
            vector_count=int(meta["vector_count"]),
            subspace_count=int(meta["subspace_count"]),
            codebook_size=int(codebooks.shape[1]),
        )
        return _state


def _pq_scores(state: PQState, query: np.ndarray) -> np.ndarray:
    subvector_dim = state.dimension // state.subspace_count
    query_parts = query.reshape(state.subspace_count, subvector_dim)
    scores = np.zeros(state.vector_count, dtype="float32")

    for part_id in range(state.subspace_count):
        centroids = state.codebooks[part_id]
        diff = centroids - query_parts[part_id].reshape(1, -1)
        lookup = np.sum(diff * diff, axis=1)
        scores += lookup[state.codes[:, part_id]]
    return scores


def _read_meta() -> dict:
    _require_file(PQ_META_PATH, "PQ metadata")
    _require_file(PQ_CODEBOOKS_PATH, "PQ codebooks")
    _require_file(PQ_CODES_PATH, "PQ codes")
    _require_file(PQ_IDS_PATH, "PQ ids")

    meta = json.loads(PQ_META_PATH.read_text(encoding="utf-8"))
    required = {"dimension", "vector_count", "subspace_count"}
    missing = sorted(required - set(meta))
    if missing:
        raise ValueError(f"PQ metadata missing required key(s): {', '.join(missing)}")
    return meta


def _validate_arrays(meta: dict, codebooks: np.ndarray, codes: np.ndarray, ids: np.ndarray) -> None:
    dimension = int(meta["dimension"])
    vector_count = int(meta["vector_count"])
    subspace_count = int(meta["subspace_count"])

    if dimension <= 0 or vector_count <= 0 or subspace_count <= 0:
        raise ValueError("PQ metadata dimension, vector_count, and subspace_count must be positive")
    if dimension % subspace_count != 0:
        raise ValueError(f"dimension={dimension} is not divisible by subspace_count={subspace_count}")
    if codebooks.ndim != 3:
        raise ValueError("pq_codebooks.npy must have shape (subspace_count, codebook_size, subvector_dim)")
    if codebooks.shape[0] != subspace_count:
        raise ValueError(f"codebook subspace mismatch: meta={subspace_count}, file={codebooks.shape[0]}")
    if codebooks.shape[2] != dimension // subspace_count:
        raise ValueError("codebook subvector dimension does not match PQ metadata")
    if codes.shape != (vector_count, subspace_count):
        raise ValueError(f"pq_codes.npy must have shape {(vector_count, subspace_count)}, got {codes.shape}")
    if ids.shape != (vector_count,):
        raise ValueError(f"pq_ids.npy must have shape {(vector_count,)}, got {ids.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("pq_codes.npy must use an integer dtype")
    if int(np.max(codes)) >= codebooks.shape[1] or int(np.min(codes)) < 0:
        raise ValueError("pq_codes.npy contains code ids outside the codebook range")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
