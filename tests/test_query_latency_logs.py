"""Tests for per-run P99 and query latency log files."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_query_latency_module():
    path = ROOT / "workload-generator" / "query_latency.py"
    spec = importlib.util.spec_from_file_location("query_latency_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_p99_logs_creates_new_files_for_each_run(tmp_path) -> None:
    module = _load_query_latency_module()
    p99_base = tmp_path / "faasann_gist_p99_1s.csv"
    trace_base = tmp_path / "faasann_gist_query_trace.csv"
    batch_start = 1_786_689_000.123
    responses = [
        {
            "query_id": 0,
            "request_id": "query-0",
            "client_elapsed_s": 0.025,
            "_client_started_at_s": batch_start + 0.2,
            "_client_finished_at_s": batch_start + 0.225,
            "_planned_offset_s": 0.1,
        }
    ]

    first_p99, first_trace = module.write_p99_logs(
        p99_base,
        trace_base,
        responses,
        batch_start,
        "gist",
        1.0,
        method="faasann",
    )
    second_p99, second_trace = module.write_p99_logs(
        p99_base,
        trace_base,
        responses,
        batch_start,
        "gist",
        1.0,
        method="faasann",
    )

    assert first_p99.exists()
    assert first_trace.exists()
    assert second_p99.exists()
    assert second_trace.exists()
    assert first_p99 != second_p99
    assert first_trace != second_trace
    assert second_p99.stem == f"{first_p99.stem}-2"
    assert second_trace.stem == f"{first_trace.stem}-2"
    assert not p99_base.exists()
    assert not trace_base.exists()

    first_run_id = first_p99.stem.removeprefix(f"{p99_base.stem}_")
    assert first_trace.stem.endswith(first_run_id)


def test_latency_trace_separates_each_client_queue_stage() -> None:
    module = _load_query_latency_module()
    batch_start = 100.0
    rows = module.build_latency_trace(
        [
            {
                "query_id": 0,
                "request_id": "query-0",
                "_planned_offset_s": 0.1,
                "_client_future_submitted_at_s": 100.12,
                "_client_worker_started_at_s": 100.15,
                "_client_started_at_s": 100.17,
                "_client_finished_at_s": 100.20,
                "client_elapsed_s": 0.03,
            }
        ],
        batch_start,
        "gist",
        method="full_search",
    )

    assert rows[0]["scheduler_submit_slip_ms"] == 20.0
    assert rows[0]["executor_queue_ms"] == 30.0
    assert rows[0]["request_prepare_ms"] == 20.0
    assert rows[0]["schedule_slip_ms"] == 70.0
    assert rows[0]["planned_to_completion_ms"] == 100.0
