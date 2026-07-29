"""Dataset-aware query runner defaults."""

from __future__ import annotations

import importlib.util
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
    assert args.method_label == "full_search"
    assert args.p99_log_file == (
        "baseline/functions/full_search/test/result/P99/7_24/full_search_gist_p99_5s.csv"
    )
    assert args.latency_trace_file == (
        "baseline/functions/full_search/test/result/P99/7_24/full_search_gist_query_trace.csv"
    )


def test_two_stage_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "baseline" / "functions" / "Two_stage_search" / "test" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "baseline/functions/Two_stage_search/test/result/run_queries_gist.csv"
    assert args.method_label == "two_stage_memory"
    assert args.p99_log_file == (
        "baseline/functions/Two_stage_search/test/result/P99/7_24/two_stage_memory_gist_p99_5s.csv"
    )
    assert args.latency_trace_file == (
        "baseline/functions/Two_stage_search/test/result/P99/7_24/two_stage_memory_gist_query_trace.csv"
    )


def test_sharded_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(
        ROOT / "baseline" / "functions" / "sharded_hnsw_search" / "test" / "run_queries.py"
    )

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "baseline/functions/sharded_hnsw_search/test/result/run_queries_gist.csv"
    assert args.method_label == "sharded_hnsw"
    assert args.p99_log_file == (
        "baseline/functions/sharded_hnsw_search/test/result/P99/7_24/sharded_hnsw_gist_p99_5s.csv"
    )
    assert args.latency_trace_file == (
        "baseline/functions/sharded_hnsw_search/test/result/P99/7_24/sharded_hnsw_gist_query_trace.csv"
    )


def test_server_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "logs/run_queries_gist.csv"
    assert args.p99_window_seconds == 5.0
    assert args.p99_log_file == "logs/P99/7_24/faasann_gist_p99_5s.csv"
    assert args.latency_trace_file == "logs/P99/7_24/faasann_gist_query_trace.csv"


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


def test_server_runner_timing_diagnostics_reports_entry_workers(tmp_path) -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")
    responses = [
        {
            "client_elapsed_s": 0.100,
            "_client_started_at_s": 100.01,
            "_planned_offset_s": 0.0,
            "_client_to_asgi_ms": 20.0,
            "_vm_http_process_ms": 60.0,
            "_server_to_client_ms": 20.0,
            "_worker_pid": 101,
            "_worker_active_at_entry": 3.0,
            "_worker_max_active": 4.0,
            "plan": {"mode": "faas"},
            "timings_ms": {"total": 50.0, "plan": 1.0, "candidates": 40.0, "rerank": 9.0},
        },
        {
            "client_elapsed_s": 0.120,
            "_client_started_at_s": 100.02,
            "_planned_offset_s": 0.0,
            "_client_to_asgi_ms": 30.0,
            "_vm_http_process_ms": 70.0,
            "_server_to_client_ms": 20.0,
            "_worker_pid": 202,
            "_worker_active_at_entry": 2.0,
            "_worker_max_active": 5.0,
            "plan": {"mode": "faas"},
            "timings_ms": {"total": 60.0, "plan": 1.0, "candidates": 50.0, "rerank": 9.0},
        },
    ]
    summary = {
        "avg_entry_request_ms": 110.0,
        "avg_server_total_ms": 55.0,
        "avg_function_request_ms": 45.0,
    }
    path = tmp_path / "timing.log"

    module.write_timing_diagnostics(path, responses, summary, batch_start_wall_time=100.0)

    output = path.read_text(encoding="utf-8")
    assert "client_dispatch_delay=samples:2 avg:15.000" in output
    assert "client_to_asgi=samples:2 avg:25.000" in output
    assert "worker_count=2" in output
    assert "worker pid=101 requests=1 local=0 faas=1 peak_active=4" in output
    assert "worker pid=202 requests=1 local=0 faas=1 peak_active=5" in output


def test_server_runner_builds_fixed_window_p99() -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")
    responses = [
        {
            "query_id": 0,
            "request_id": "query-0",
            "client_elapsed_s": 0.010,
            "_client_started_at_s": 100.5,
            "_client_finished_at_s": 100.51,
            "_planned_offset_s": 0.4,
            "plan": {"mode": "faas"},
        },
        {
            "query_id": 1,
            "request_id": "query-1",
            "client_elapsed_s": 0.030,
            "_client_started_at_s": 104.0,
            "_client_finished_at_s": 104.03,
            "_planned_offset_s": 3.8,
            "plan": {"mode": "faas"},
        },
        {
            "query_id": 2,
            "request_id": "query-2",
            "client_elapsed_s": 0.020,
            "_client_started_at_s": 106.0,
            "_client_finished_at_s": 106.02,
            "_planned_offset_s": 5.9,
            "plan": {"mode": "faas"},
            "error": "timeout",
        },
    ]

    trace = module.build_latency_trace(
        responses,
        batch_start_wall_time=100.0,
        dataset="gist",
        method="faasann",
    )
    windows = module.build_p99_windows(trace, window_seconds=5.0)

    assert len(windows) == 2
    assert windows[0]["method"] == "faasann"
    assert windows[0]["request_count"] == 2
    assert windows[0]["success_count"] == 2
    assert windows[0]["p99_client_latency_ms"] == 30.0
    assert windows[0]["p99_schedule_slip_ms"] == 200.0
    assert windows[1]["request_count"] == 1
    assert windows[1]["success_count"] == 0
    assert windows[1]["error_count"] == 1
    assert windows[1]["p99_client_latency_ms"] == ""


def test_server_runner_stops_plan_after_first_query_error(monkeypatch) -> None:
    module = _load_module(ROOT / "tests" / "hnsw" / "run_queries.py")

    def fail_query(**_kwargs):
        raise RuntimeError("synthetic query failure")

    monkeypatch.setattr(module, "send_one_query", fail_query)
    plan = SimpleNamespace(timestamps=[0.0, 0.5, 1.0], offset=0.0)
    started_at = time.perf_counter()
    responses, _batch_start = module.run_queries(
        server_url="http://127.0.0.1:1",
        query_vectors=np.array([[0.0, 0.0]], dtype=np.float32),
        groundtruth=np.array([[0]], dtype=np.int32),
        count=3,
        k=1,
        candidate_k=1,
        ef_search=1,
        use_faas=True,
        send_vectors=True,
        concurrent_requests=1,
        timeout=1.0,
        continue_on_error=False,
        plan=plan,
        show_progress=False,
    )

    assert time.perf_counter() - started_at < 0.4
    assert len(responses) == 1
    assert responses[0]["query_id"] == 0
    assert responses[0]["error"] == "synthetic query failure"


def test_to_fc_runner_defaults_to_gist_files(monkeypatch) -> None:
    module = _load_module(ROOT / "to_FC" / "test" / "run_queries.py")

    monkeypatch.setenv("FAASANN_DATASET", "gist")
    monkeypatch.setattr(sys, "argv", ["run_queries.py"])
    args = module.parse_args()

    assert args.dataset == "gist"
    assert args.query_file == "data/gist/gist_query.fvecs"
    assert args.groundtruth_file == "data/gist/gist_groundtruth.ivecs"
    assert args.log_file == "logs/run_queries_to_FC_gist.csv"


def test_to_fc_summary_uses_callback_timing_names() -> None:
    module = _load_module(ROOT / "to_FC" / "test" / "run_queries.py")

    class Args:
        dataset = "gist"
        concurrent_requests = 1
        k = 10
        candidate_k = 130
        ef_search = 130

    responses = [
        {
            "recall": 1.0,
            "results": [{"id": 1}],
            "client_final_result_elapsed_s": 0.030,
            "function_timings_ms": {"handler_total": 6.0, "faiss_search": 3.0, "vm_accept_request": 2.0},
            "server_timings_ms": {"rerank": 1.2, "total_before_callback": 1.5},
            "function_metrics": {"candidate_count": 130},
        }
    ]
    acks = [{"fc_accept_elapsed_s": 0.010}]

    summary = module.summarize(Args(), responses, acks, 0.030, 100.0)

    assert summary["entrypoint"] == "function_callback"
    assert summary["qps_client_final_result"] == 33.33
    assert summary["avg_fc_accept_request_ms"] == 10.0
    assert summary["avg_client_final_result_ms"] == 30.0
    assert summary["avg_function_handler_ms"] == 6.0
    assert summary["avg_function_ann_search_ms"] == 3.0
    assert summary["avg_function_to_vm_accept_ms"] == 2.0
    assert summary["avg_server_rerank_ms"] == 1.2
