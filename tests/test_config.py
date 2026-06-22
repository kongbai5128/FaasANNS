"""Runtime config loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.config import load_config


def _dataset_paths(dataset: str) -> tuple[str, str]:
    prefix = "sift" if dataset == "sift100w" else dataset
    return (
        f"data/{dataset}/{prefix}_base.fvecs",
        f"data/{dataset}/index/full/full_hnsw.bin",
    )


def _full_config(dataset: str = "sift100w", dimension: int = 128) -> dict:
    base_path, hnsw_index_path = _dataset_paths(dataset)
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "info",
        },
        "dataset": {
            "base_path": base_path,
            "dimension": dimension,
            "max_vectors": 1000000,
        },
        "search": {
            "hnsw": {
                "default_k": 10,
                "candidate_k": 120,
                "hnsw_index_path": hnsw_index_path,
                "hnsw_m": 32,
                "hnsw_ef_construction": 200,
                "hnsw_ef_search": 80,
            },
            "pipeline": {
                "local_search_workers": 2,
                "faas_invoke_workers": 100,
                "rerank_workers": 4,
            },
            "offload_qps_threshold": 20.0,
            "force_faas": False,
        },
        "scaling": {
            "prewarm_check_seconds": 0.1,
            "load_index_timeout_seconds": 3.0,
            "enable_prewarm": True,
            "local_candidate_ms": 8.0,
            "remote_candidate_ms": 20.0,
            "function_concurrency": 1,
            "max_warm_functions": 32,
            "function_memory_mb": 512,
            "cost_per_gb_second": 0.0000167,
        },
        "faas": {
            "provider": "local",
            "invoke_timeout_seconds": 10.0,
            "warmup_timeout_seconds": 5.0,
            "endpoints": {},
        },
    }


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "server.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("dataset", "dimension", "base_path", "hnsw_index_path"),
    [
        (
            "sift100w",
            128,
            "data/sift100w/sift_base.fvecs",
            "data/sift100w/index/full/full_hnsw.bin",
        ),
        (
            "gist",
            960,
            "data/gist/gist_base.fvecs",
            "data/gist/index/full/full_hnsw.bin",
        ),
    ],
)
def test_load_config_accepts_complete_config(
    tmp_path: Path,
    dataset: str,
    dimension: int,
    base_path: str,
    hnsw_index_path: str,
) -> None:
    config = load_config(_write_config(tmp_path, _full_config(dataset=dataset, dimension=dimension)))

    assert config.server.port == 8080
    assert config.dataset.base_path == base_path
    assert config.dataset.dimension == dimension
    assert config.search.hnsw.hnsw_index_path == hnsw_index_path


def test_load_config_rejects_missing_key(tmp_path: Path) -> None:
    data = _full_config()
    del data["dataset"]["dimension"]

    with pytest.raises(ValueError, match="dataset: .*dimension"):
        load_config(_write_config(tmp_path, data))


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    data = _full_config()
    data["search"]["hnsw"]["made_up"] = 123

    with pytest.raises(ValueError, match="search.hnsw: unknown config key"):
        load_config(_write_config(tmp_path, data))


def test_load_config_rejects_flat_search_key(tmp_path: Path) -> None:
    data = _full_config()
    data["search"]["candidate_k"] = 999

    with pytest.raises(ValueError, match="search: unknown config key.*candidate_k"):
        load_config(_write_config(tmp_path, data))
