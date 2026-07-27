"""Shared per-query trace and fixed-window latency statistics."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path


def write_p99_logs(
    p99_path: Path,
    trace_path: Path,
    responses: list[dict],
    batch_start_wall_time: float,
    dataset: str,
    window_seconds: float,
    *,
    method: str = "",
) -> None:
    """Append one run's per-query trace and fixed-window P99 rows."""

    trace_rows = build_latency_trace(
        responses,
        batch_start_wall_time,
        dataset,
        method=method,
    )
    p99_rows = build_p99_windows(trace_rows, window_seconds)
    write_rows(trace_path, trace_rows)
    write_rows(p99_path, p99_rows)


def build_latency_trace(
    responses: list[dict],
    batch_start_wall_time: float,
    dataset: str,
    *,
    method: str = "",
) -> list[dict]:
    run_id = (
        time.strftime("%Y%m%dT%H%M%S", time.localtime(batch_start_wall_time))
        + f"-{int(batch_start_wall_time * 1000) % 1000:03d}"
    )
    run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(batch_start_wall_time))
    rows: list[dict] = []
    for item in sorted(responses, key=lambda response: int(response.get("query_id", 0))):
        started_at = _as_float(item.get("_client_started_at_s"))
        finished_at = _as_float(item.get("_client_finished_at_s"))
        elapsed_s = _as_float(item.get("client_elapsed_s"))
        planned_offset_s = _as_float(item.get("_planned_offset_s"))
        actual_offset_s = started_at - batch_start_wall_time if started_at is not None else None
        completion_offset_s = finished_at - batch_start_wall_time if finished_at is not None else None
        schedule_slip_ms = (
            (actual_offset_s - planned_offset_s) * 1000.0
            if actual_offset_s is not None and planned_offset_s is not None
            else None
        )
        timings = item.get("timings_ms", {})
        function_timings = item.get("function_timings_ms", {})
        if not isinstance(function_timings, dict) or not function_timings:
            function_timings = timings
        plan = item.get("plan", {})
        rows.append(
            {
                "run_id": run_id,
                "run_start_time": run_start_time,
                "method": method,
                "dataset": dataset,
                "query_id": item.get("query_id", ""),
                "request_id": item.get("request_id", ""),
                "planned_offset_s": _rounded_or_blank(planned_offset_s, 6),
                "actual_start_offset_s": _rounded_or_blank(actual_offset_s, 6),
                "completion_offset_s": _rounded_or_blank(completion_offset_s, 6),
                "schedule_slip_ms": _rounded_or_blank(schedule_slip_ms, 3),
                "client_latency_ms": _rounded_or_blank(elapsed_s * 1000.0 if elapsed_s is not None else None, 3),
                "success": "error" not in item,
                "error": item.get("error", ""),
                "plan_mode": plan.get("mode", "") if isinstance(plan, dict) else "",
                "cold_start_id": item.get("cold_start_id", ""),
                "vm_http_process_ms": _rounded_or_blank(_as_float(item.get("_vm_http_process_ms")), 3),
                "search_service_total_ms": _rounded_or_blank(
                    _as_float(timings.get("total")) if isinstance(timings, dict) else None, 3
                ),
                "function_handler_ms": _rounded_or_blank(
                    _as_float(function_timings.get("handler_total"))
                    if isinstance(function_timings, dict)
                    else None,
                    3,
                ),
            }
        )
    return rows


def build_p99_windows(trace_rows: list[dict], window_seconds: float) -> list[dict]:
    if window_seconds <= 0:
        raise ValueError("P99 window must be positive")
    if not trace_rows:
        return []

    by_bin: dict[int, list[dict]] = {}
    for row in trace_rows:
        actual_offset = _as_float(row.get("actual_start_offset_s"))
        if actual_offset is None:
            continue
        bin_index = max(0, int(math.floor(actual_offset / window_seconds)))
        by_bin.setdefault(bin_index, []).append(row)
    if not by_bin:
        return []

    first = trace_rows[0]
    result: list[dict] = []
    for bin_index in range(max(by_bin) + 1):
        items = by_bin.get(bin_index, [])
        successful = [item for item in items if item.get("success") is True]
        latencies = [_as_float(item.get("client_latency_ms")) for item in successful]
        latencies = [value for value in latencies if value is not None]
        slips = [_as_float(item.get("schedule_slip_ms")) for item in items]
        slips = [value for value in slips if value is not None]
        result.append(
            {
                "run_id": first["run_id"],
                "run_start_time": first["run_start_time"],
                "method": first.get("method", ""),
                "dataset": first["dataset"],
                "window_seconds": window_seconds,
                "bin_index": bin_index,
                "bin_start_s": round(bin_index * window_seconds, 6),
                "bin_end_s": round((bin_index + 1) * window_seconds, 6),
                "request_count": len(items),
                "success_count": len(successful),
                "error_count": len(items) - len(successful),
                "actual_qps": round(len(items) / window_seconds, 3),
                "p99_client_latency_ms": _rounded_or_blank(_nearest_rank(latencies, 0.99), 3),
                "max_client_latency_ms": _rounded_or_blank(max(latencies) if latencies else None, 3),
                "p99_schedule_slip_ms": _rounded_or_blank(_nearest_rank(slips, 0.99), 3),
            }
        )
    return result


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _rounded_or_blank(value: float | None, digits: int):
    return "" if value is None else round(value, digits)


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists and _existing_header(path) != fieldnames:
        archive_path = path.with_suffix(path.suffix + f".old-{int(time.time())}")
        path.rename(archive_path)
        exists = False
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _existing_header(path: Path) -> list[str] | None:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return next(csv.reader(fp), None)
