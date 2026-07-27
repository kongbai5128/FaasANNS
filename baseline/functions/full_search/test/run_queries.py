"""Send dataset queries directly to the full-search Function Compute baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workload-generator"))

from vectors.fvecs import read_fvecs
from vectors.ivecs import read_ivecs
from query_latency import write_p99_logs
from workload_plan import QueryProgress, describe_workload_plan, load_workload_plan, resolve_request_count, submit_with_plan


DEFAULT_DATASET = "sift100w"


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().strip("/")
    if not dataset or "/" in dataset or dataset in {".", ".."}:
        raise SystemExit("dataset must be one directory name, for example: sift100w or gist")
    return dataset


def dataset_file_prefix(dataset: str) -> str:
    if dataset == "sift100w":
        return "sift"
    return dataset


def default_query_file(dataset: str) -> str:
    prefix = dataset_file_prefix(dataset)
    return f"data/{dataset}/{prefix}_query.fvecs"


def default_groundtruth_file(dataset: str) -> str:
    prefix = dataset_file_prefix(dataset)
    return f"data/{dataset}/{prefix}_groundtruth.ivecs"


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def send_one_query(
    *,
    endpoint: str,
    query_id: int,
    request_id: str,
    vector: list[float],
    k: int,
    candidate_k: int,
    ef_search: int,
    timeout: float,
) -> dict:
    started_at = time.time()
    start = time.perf_counter()
    payload = {
        "request_id": request_id,
        "query": vector,
        "candidate_k": candidate_k,
        "ef_search": ef_search,
    }
    response = post_json(endpoint, payload, timeout)
    if response.get("error"):
        raise RuntimeError(f"Function error: {response['error']}")
    result_ids = [int(item["id"]) for item in response.get("candidates", [])[:k]]
    elapsed_s = time.perf_counter() - start
    finished_at = time.time()

    return {
        "query_id": query_id,
        "request_id": request_id,
        "source_query_id": query_id,
        "result_ids": result_ids,
        "client_elapsed_s": elapsed_s,
        "_client_started_at_s": started_at,
        "_client_finished_at_s": finished_at,
        "cold_start_id": response.get("cold_start_id"),
        "index_loaded_at": response.get("index_loaded_at"),
        "timings_ms": response.get("timings_ms", {}),
        "function_metrics": response.get("function_metrics", {}),
    }


def calculate_recall(result_ids: list[int], truth_ids: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(result_ids[:k]).intersection(map(int, truth_ids[:k]))) / k


def run_queries(args: argparse.Namespace) -> list[dict]:
    query_file = ROOT / args.query_file
    groundtruth_file = ROOT / args.groundtruth_file
    print(f"Endpoint: {args.endpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Query file: {query_file}")
    print(f"Groundtruth file: {groundtruth_file}")

    if args.candidate_k < args.k:
        raise SystemExit("candidate_k must be greater than or equal to k")

    plan = load_workload_plan(args.plan_file, root=ROOT) if args.plan_file else None
    try:
        request_count = resolve_request_count(args.query_num, plan)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    queries = read_fvecs(query_file, max_vectors=request_count)
    groundtruth = read_ivecs(groundtruth_file, max_vectors=len(queries))
    if len(queries) == 0:
        raise SystemExit("query file has no vectors")
    if len(groundtruth) < len(queries):
        raise SystemExit("groundtruth row count is smaller than query count")
    print(
        f"Sending {request_count} requests directly to FC with client_workers={args.concurrent_requests} "
        f"(cycling {len(queries)} query vectors, "
        f"rough per-worker requests={request_count / max(1, args.concurrent_requests):.1f})"
    )
    print(describe_workload_plan(plan, request_count))
    if request_count > len(queries):
        print(f"Replaying {request_count} requests by cycling {len(queries)} query vectors")

    progress = QueryProgress(request_count, enabled=not args.no_progress)
    stop_event = threading.Event()

    def submit(i: int) -> dict:
        source_query_id = i % len(queries)
        progress.mark_sent()
        vector = queries[source_query_id].astype("float32").tolist()
        attempt_started_at = time.time()
        attempt_start = time.perf_counter()
        try:
            response = send_one_query(
                endpoint=args.endpoint,
                query_id=source_query_id,
                request_id=f"query-{i}",
                vector=vector,
                k=args.k,
                candidate_k=args.candidate_k,
                ef_search=args.ef_search,
                timeout=args.timeout,
            )
        except Exception as exc:
            return {
                "query_id": i,
                "request_id": f"query-{i}",
                "source_query_id": source_query_id,
                "result_ids": [],
                "recall": 0.0,
                "error": str(exc),
                "client_elapsed_s": time.perf_counter() - attempt_start,
                "_client_started_at_s": attempt_started_at,
                "_client_finished_at_s": time.time(),
                "_planned_offset_s": plan.timestamps[i] if plan is not None else None,
            }
        response["query_id"] = i
        response["source_query_id"] = source_query_id
        response["recall"] = calculate_recall(response["result_ids"], groundtruth[source_query_id].tolist(), args.k)
        response["_planned_offset_s"] = plan.timestamps[i] if plan is not None else None
        return response

    def track_future(future: Future) -> None:
        def on_done(completed: Future) -> None:
            if completed.cancelled():
                return
            progress.mark_received()
            if args.continue_on_error:
                return
            try:
                result = completed.result()
            except Exception:
                stop_event.set()
                return
            if isinstance(result, dict) and "error" in result:
                stop_event.set()

        future.add_done_callback(on_done)

    args.batch_start_wall_time = time.time()
    results: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
            futures, args.batch_start_wall_time = submit_with_plan(
                executor,
                request_count,
                submit,
                plan,
                on_submit=track_future,
                stop_event=stop_event,
            )
            if stop_event.is_set():
                for future in futures:
                    future.cancel()
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                results.append(future.result())
    finally:
        progress.close()
    return results


def cold_start_load_times(responses: list[dict], batch_start_wall_time: float) -> dict[str, float]:
    cold_start_load_ms: dict[str, float] = {}
    for item in responses:
        cold_start_id = item.get("cold_start_id")
        if cold_start_id is None:
            continue
        loaded_at = item.get("index_loaded_at")
        if loaded_at is not None:
            try:
                if float(loaded_at) < batch_start_wall_time:
                    continue
            except (TypeError, ValueError):
                pass
        metrics = item.get("function_metrics", {})
        index_load_ms = _as_float(metrics.get("index_load_ms")) if isinstance(metrics, dict) else None
        cold_start_load_ms.setdefault(str(cold_start_id), index_load_ms or 0.0)
    return cold_start_load_ms


def summarize_run(
    args: argparse.Namespace,
    responses: list[dict],
    elapsed: float,
    batch_start_wall_time: float,
) -> dict:
    query_count = len(responses)
    error_count = sum(1 for item in responses if "error" in item)
    success_count = query_count - error_count

    avg_entry_request_ms = (
        sum(item.get("client_elapsed_s", 0.0) for item in responses if "error" not in item) * 1000.0 / success_count
        if success_count
        else 0.0
    )
    function_timings = [item.get("timings_ms", {}) for item in responses]
    cold_start_load_ms = cold_start_load_times(responses, batch_start_wall_time)
    avg_cold_start_load_ms = sum(cold_start_load_ms.values()) / len(cold_start_load_ms) if cold_start_load_ms else 0.0

    def avg_function_timing(name: str) -> float:
        return _avg_metric(function_timings, name)

    recall = sum(item.get("recall", 0.0) for item in responses if "error" not in item) / success_count if success_count else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "query_count": query_count,
        "success_count": success_count,
        "error_count": error_count,
        "concurrent_requests": args.concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "ef_search": args.ef_search,
        "cold_start_num": len(cold_start_load_ms),
        "avg_cold_start_load_ms": round(avg_cold_start_load_ms, 3),
        "avg_entry_request_ms": round(avg_entry_request_ms, 3),
        "avg_function_request_ms": round(avg_entry_request_ms, 3),
        "avg_function_handler_ms": round(avg_function_timing("handler_total"), 3),
        "avg_function_ann_search_ms": round(avg_function_timing("faiss_search"), 3),
        "avg_function_rerank_ms": 0.0,
        "avg_server_total_ms": 0.0,
        "avg_server_candidate_stage_ms": 0.0,
        "avg_server_rerank_ms": 0.0,
    }


def _avg_metric(items: list[dict], name: str) -> float:
    values = []
    for item in items:
        if name not in item:
            continue
        try:
            values.append(float(item[name]))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


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
        reader = csv.reader(fp)
        return next(reader, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send dataset queries to the full-search baseline")
    parser.add_argument("--endpoint", default="http://base-lil-search-mbidkbhcit.cn-hongkong.fcapp.run")
    parser.add_argument("--dataset", default=os.environ.get("FAASANN_DATASET", DEFAULT_DATASET))
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--groundtruth-file", default=None)
    parser.add_argument("--query-num", type=int, default=10)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="maximum client worker threads; in plan mode this is only a capacity cap, not fixed concurrency",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-file", default=None, help="replay query arrivals from workload-generator plan.bin")
    parser.add_argument("--no-progress", action="store_true", help="disable live sent/received progress output")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--method-label", default="full_search")
    parser.add_argument("--p99-window-seconds", type=float, default=5.0)
    parser.add_argument("--p99-log-file", default=None)
    parser.add_argument("--latency-trace-file", default=None)
    args = parser.parse_args()
    args.dataset = normalize_dataset(args.dataset)
    if args.query_file is None:
        args.query_file = default_query_file(args.dataset)
    if args.groundtruth_file is None:
        args.groundtruth_file = default_groundtruth_file(args.dataset)
    if args.log_file is None:
        args.log_file = f"baseline/functions/full_search/test/result/run_queries_{args.dataset}.csv"
    if args.p99_window_seconds <= 0:
        raise SystemExit("--p99-window-seconds must be positive")
    window_label = f"{args.p99_window_seconds:g}"
    if args.p99_log_file is None:
        args.p99_log_file = (
            "baseline/functions/full_search/test/result/P99/7_24/"
            f"{args.method_label}_{args.dataset}_p99_{window_label}s.csv"
        )
    if args.latency_trace_file is None:
        args.latency_trace_file = (
            "baseline/functions/full_search/test/result/P99/7_24/"
            f"{args.method_label}_{args.dataset}_query_trace.csv"
        )
    return args


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        responses = run_queries(args)
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc

    elapsed = time.perf_counter() - start
    errors = [item for item in responses if "error" in item]
    if errors and not args.continue_on_error:
        raise SystemExit(f"Run stopped after query error: {errors[0]['error']}")
    summary = summarize_run(args, responses, elapsed, args.batch_start_wall_time)
    write_rows(ROOT / args.log_file, [summary])
    write_p99_logs(
        ROOT / args.p99_log_file,
        ROOT / args.latency_trace_file,
        responses,
        args.batch_start_wall_time,
        args.dataset,
        args.p99_window_seconds,
        method=args.method_label,
    )
    print(
        f"Finished: queries={summary['query_count']}, "
        f"success={summary['success_count']}, errors={summary['error_count']}, "
        f"concurrency={summary['concurrent_requests']}, elapsed={summary['client_elapsed_s']:.3f}s, "
        f"qps={summary['qps_client']:.2f}, recall@{args.k}={summary['recall']:.4f}, "
        f"cold_start_num={summary['cold_start_num']}"
    )
    for item in errors[:5]:
        print(f"Error query_id={item['query_id']}: {item['error']}")
    print(f"P99 time series saved to {ROOT / args.p99_log_file}")
    print(f"Per-query latency trace saved to {ROOT / args.latency_trace_file}")


if __name__ == "__main__":
    main()
