"""Tests for timestamp-based workload replay."""

from __future__ import annotations

import io
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workload-generator"))

from workload_plan import QueryProgress, WorkloadPlan, submit_with_plan


def test_query_progress_separates_executor_queue_and_http_inflight() -> None:
    stream = io.StringIO()
    progress = QueryProgress(1, stream=stream)

    progress.mark_submitted()
    progress.mark_sent()
    progress.mark_received()
    progress.close()

    output = stream.getvalue()
    assert progress.submitted == 1
    assert progress.sent == 1
    assert progress.received == 1
    assert "submitted=1/1" in output
    assert "executor_queue=0, in_flight=0" in output


def test_high_rate_plan_does_not_starve_worker_threads() -> None:
    rate = 500
    count = 500
    starts = [0.0] * count
    plan = WorkloadPlan(Path("synthetic.bin"), [i / rate for i in range(count)])

    def submit(i: int) -> dict:
        work_until = time.perf_counter() + 0.001
        while time.perf_counter() < work_until:
            pass
        starts[i] = time.time()
        time.sleep(0.005)
        return {"query_id": i}

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures, batch_start = submit_with_plan(executor, count, submit, plan)
        responses = [future.result() for future in futures]

    slips_ms = sorted(
        (starts[i] - batch_start - plan.timestamps[i]) * 1000.0
        for i in range(count)
    )
    p99_slip_ms = slips_ms[math.ceil(len(slips_ms) * 0.99) - 1]

    assert p99_slip_ms < 20.0
    assert all("_client_future_submitted_at_s" in response for response in responses)
    assert all("_client_worker_started_at_s" in response for response in responses)
