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


def test_server_runner_summary_uses_clear_timing_names() -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    responses = [
        {
            "recall": 1.0,
            "client_elapsed_s": 0.010,
            "plan": {"mode": "faas"},
            "timings_ms": {"total": 8.0, "candidates": 5.0, "rerank": 2.0, "remote_invoke": 4.0},
            "function_timings_ms": {"handler_total": 3.0, "faiss_search": 2.5},
        }
    ]

    summary = module.summarize_run(responses, 0.010, 10, 130, 130, 1, "gist", 100.0)

    assert summary["candidate_k"] == 130
    assert summary["ef_search"] == 130
    assert summary["avg_entry_request_ms"] == 10.0
    assert summary["avg_function_request_ms"] == 4.0
    assert summary["avg_function_handler_ms"] == 3.0
    assert summary["avg_function_ann_search_ms"] == 2.5
    assert summary["avg_server_total_ms"] == 8.0
    assert summary["avg_server_candidate_stage_ms"] == 5.0
    assert summary["avg_server_rerank_ms"] == 2.0


def test_function_entry_summary_uses_clear_timing_names() -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    responses = [
        {
            "recall": 1.0,
            "client_elapsed_s": 0.012,
            "plan": {"mode": "function_entry"},
            "timings_ms": {"total": 11.0, "candidates": 6.0, "rerank": 3.0},
            "function_timings_ms": {
                "handler_total": 11.0,
                "faiss_search": 4.0,
                "server_rerank_request": 3.0,
            },
            "server_timings_ms": {"total": 2.0, "rerank": 1.5},
        }
    ]

    summary = module.summarize_run(responses, 0.012, 10, 130, 130, 1, "gist", 100.0)

    assert summary["entrypoint"] == "function"
    assert summary["candidate_k"] == 130
    assert summary["ef_search"] == 130
    assert summary["avg_entry_request_ms"] == 12.0
    assert summary["avg_function_request_ms"] == 12.0
    assert summary["avg_function_handler_ms"] == 11.0
    assert summary["avg_function_ann_search_ms"] == 4.0
    assert summary["avg_server_total_ms"] == 2.0
    assert summary["avg_server_candidate_stage_ms"] == 0.0
    assert summary["avg_server_rerank_ms"] == 1.5
