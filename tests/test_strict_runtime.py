"""Strict runtime failure tests."""

from __future__ import annotations

import numpy as np
import pytest

from main import load_vector_store
from search.hnsw import HNSWIndex
from utils.config import (
    AppConfig,
    DatasetConfig,
    FaaSConfig,
    HNSWConfig,
    PipelineConfig,
    ScalingConfig,
    SearchConfig,
    ServerConfig,
)


def test_load_vector_store_rejects_missing_dataset(tmp_path) -> None:
    config = AppConfig(
        server=ServerConfig(host="127.0.0.1", port=8080, log_level="info"),
        dataset=DatasetConfig(
            base_path="missing.fvecs",
            dimension=4,
            max_vectors=10,
        ),
        search=SearchConfig(
            hnsw=HNSWConfig(
                default_k=10,
                candidate_k=120,
                hnsw_index_path="data/index/full/full_hnsw.bin",
                hnsw_m=32,
                hnsw_ef_construction=200,
                hnsw_ef_search=80,
            ),
            pipeline=PipelineConfig(local_search_workers=2, rerank_workers=4),
            offload_qps_threshold=20.0,
            force_faas=False,
        ),
        scaling=ScalingConfig(
            prewarm_check_seconds=0.1,
            load_index_timeout_seconds=3.0,
            enable_prewarm=True,
            local_candidate_ms=8.0,
            remote_candidate_ms=20.0,
            function_concurrency=1,
            max_warm_functions=32,
            function_memory_mb=512,
            cost_per_gb_second=0.0000167,
        ),
        faas=FaaSConfig(
            provider="local",
            invoke_timeout_seconds=10.0,
            warmup_timeout_seconds=5.0,
            endpoints={},
        ),
    )

    with pytest.raises(FileNotFoundError, match="dataset file not found"):
        load_vector_store(config, tmp_path / "configs" / "server.json")


def test_hnsw_index_rejects_missing_configured_index(tmp_path) -> None:
    vectors = np.zeros((2, 4), dtype="float32")

    with pytest.raises(FileNotFoundError, match="hnswlib index file not found"):
        HNSWIndex(
            vectors,
            index_path=tmp_path / "missing.bin",
            m=32,
            ef_construction=200,
            ef_search=80,
        )


def test_hnsw_index_rejects_missing_metadata(tmp_path) -> None:
    vectors = np.zeros((2, 4), dtype="float32")
    index_path = tmp_path / "index.bin"
    index_path.write_bytes(b"not a real index")

    with pytest.raises(FileNotFoundError, match="hnswlib metadata file not found"):
        HNSWIndex(
            vectors,
            index_path=index_path,
            m=32,
            ef_construction=200,
            ef_search=80,
        )
