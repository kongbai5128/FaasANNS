"""Dataset-aware query runner defaults."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_full_search_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "baseline" / "functions" / "full_search" / "test" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "baseline/functions/full_search/test/result/run_queries_gist.csv"


def test_two_stage_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "baseline" / "functions" / "Two_stage_search" / "test" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "baseline/functions/Two_stage_search/test/result/run_queries_gist.csv"


def test_server_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "logs/run_queries_gist.csv"


def _load_module(path: Path):
    name = f"dataset_query_defaults_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_cold_start_count_ignores_warm_instances() -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    responses = [
        {"cold_start_id": "warm", "index_loaded_at": 90.0, "function_metrics": {"index_load_ms": 10.0}},
        {"cold_start_id": "cold-a", "index_loaded_at": 101.0, "function_metrics": {"index_load_ms": 20.0}},
        {"cold_start_id": "cold-a", "index_loaded_at": 101.0, "function_metrics": {"index_load_ms": 20.0}},
        {"cold_start_id": "cold-b", "index_loaded_at": 102.0, "function_metrics": {"index_load_ms": 30.0}},
        {"cold_start_id": None, "index_loaded_at": 103.0},
    ]

    assert module.cold_start_load_times(responses, batch_start_wall_time=100.0) == {
        "cold-a": 20.0,
        "cold-b": 30.0,
    }
