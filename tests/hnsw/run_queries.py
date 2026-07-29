"""向 FaasANN 服务器发送数据集查询并计算 Recall。

这个脚本参考旧项目 /home/qian/faasann/test/hnsw/run.py 的实验方式：
读取查询向量和 groundtruth，按并发度向当前服务器的 `/search` 接口发送请求，
统计客户端 QPS、Recall@K、local/FaaS 路径数量和阶段耗时，并把最终汇总结果写入 CSV。

./run_queries.sh  --dataset gist

"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workload-generator"))

from vectors.fvecs import read_fvecs
from vectors.ivecs import read_ivecs
from query_latency import build_latency_trace, build_p99_windows, write_p99_logs
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


def post_json(url: str, payload: dict, timeout: float, client_started_ns: int) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-faasann-client-started-ns": str(client_started_ns),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            process_time = response.headers.get("x-process-time-ms")
            if process_time is not None and isinstance(data, dict):
                try:
                    data["_vm_http_process_ms"] = float(process_time)
                except ValueError:
                    pass
            if isinstance(data, dict):
                _copy_diagnostic_headers(response.headers, data)
            return data
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _copy_diagnostic_headers(headers, data: dict) -> None:
    float_headers = {
        "x-faasann-client-to-asgi-ms": "_client_to_asgi_ms",
        "x-faasann-worker-active": "_worker_active_at_entry",
        "x-faasann-worker-max-active": "_worker_max_active",
    }
    int_headers = {
        "x-faasann-worker-pid": "_worker_pid",
        "x-faasann-asgi-started-ns": "_server_asgi_started_ns",
        "x-faasann-response-ready-ns": "_server_response_ready_ns",
    }
    for header, key in float_headers.items():
        value = headers.get(header)
        if value is not None:
            try:
                data[key] = float(value)
            except ValueError:
                pass
    for header, key in int_headers.items():
        value = headers.get(header)
        if value is not None:
            try:
                data[key] = int(value)
            except ValueError:
                pass


def send_one_query(
    server_url: str,
    query_id: int,
    request_id: str,
    vector: list[float] | None,
    k: int,
    candidate_k: int | None,
    ef_search: int | None,
    use_faas: bool | None,
    timeout: float,
) -> dict:
    payload: dict = {
        "request_id": request_id,
        "k": k,
    }
    if vector is None:
        payload["query_id"] = query_id
    else:
        payload["vector"] = vector
    if candidate_k is not None:
        payload["candidate_k"] = candidate_k
    if ef_search is not None:
        payload["ef_search"] = ef_search
    if use_faas is not None:
        payload["use_faas"] = use_faas
    url = f"{server_url.rstrip('/')}/search"

    started_at_ns = time.time_ns()
    started_at = started_at_ns / 1_000_000_000.0
    start = time.perf_counter()
    response = post_json(url, payload, timeout, started_at_ns)
    finished_at_ns = time.time_ns()
    response["client_elapsed_s"] = time.perf_counter() - start
    response["_client_started_at_s"] = started_at
    response["_client_finished_at_s"] = finished_at_ns / 1_000_000_000.0
    response_ready_ns = response.get("_server_response_ready_ns")
    if isinstance(response_ready_ns, int):
        response["_server_to_client_ms"] = (finished_at_ns - response_ready_ns) / 1_000_000.0
    return response


def calculate_recall(result_ids: list[int], truth_ids: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    found = set(result_ids[:k])
    truth = set(truth_ids[:k])
    return len(found.intersection(truth)) / k


def run_queries(
    server_url: str,
    query_vectors,
    groundtruth,
    count: int,
    k: int,
    candidate_k: int | None,
    ef_search: int | None,
    use_faas: bool | None,
    send_vectors: bool,
    concurrent_requests: int,
    timeout: float,
    continue_on_error: bool,
    plan,
    show_progress: bool = True,
) -> tuple[list[dict], float]:
    if len(query_vectors) == 0:
        raise ValueError("no query vectors loaded")

    progress = QueryProgress(count, enabled=show_progress)
    stop_event = threading.Event()

    def submit(i: int) -> dict:
        source_query_id = i % len(query_vectors)
        progress.mark_sent()
        vector = query_vectors[source_query_id].astype("float32").tolist() if send_vectors else None
        attempt_started_at = time.time()
        attempt_start = time.perf_counter()
        try:
            response = send_one_query(
                server_url=server_url,
                query_id=source_query_id,
                request_id=f"query-{i}",
                vector=vector,
                k=k,
                candidate_k=candidate_k,
                ef_search=ef_search,
                use_faas=use_faas,
                timeout=timeout,
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
        result_ids = [int(item["id"]) for item in response.get("results", [])]
        response["query_id"] = i
        response["source_query_id"] = source_query_id
        response["result_ids"] = result_ids
        response["recall"] = calculate_recall(result_ids, groundtruth[source_query_id].tolist(), k)
        response["_planned_offset_s"] = plan.timestamps[i] if plan is not None else None
        return response

    def track_future(future: Future) -> None:
        def on_done(completed: Future) -> None:
            if completed.cancelled():
                return
            progress.mark_received()
            if continue_on_error:
                return
            try:
                result = completed.result()
            except Exception:
                stop_event.set()
                return
            if isinstance(result, dict) and "error" in result:
                stop_event.set()

        future.add_done_callback(on_done)

    results: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures, batch_start_wall_time = submit_with_plan(
                executor,
                count,
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
    return results, batch_start_wall_time


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
    responses: list[dict],
    elapsed: float,
    k: int,
    candidate_k: int | None,
    ef_search: int | None,
    concurrent_requests: int,
    dataset: str,
    batch_start_wall_time: float,
) -> dict:
    query_count = len(responses)
    timings = [item.get("timings_ms", {}) for item in responses]
    function_timings = [item.get("function_timings_ms", {}) for item in responses]
    server_timings = [item.get("server_timings_ms", {}) for item in responses]
    plans = [item.get("plan", {}) for item in responses]
    cold_start_load_ms = cold_start_load_times(responses, batch_start_wall_time)
    avg_cold_start_load_ms = sum(cold_start_load_ms.values()) / len(cold_start_load_ms) if cold_start_load_ms else 0.0
    error_count = sum(1 for item in responses if "error" in item)
    success_count = query_count - error_count
    avg_entry_request_ms = (
        sum(item.get("client_elapsed_s", 0.0) for item in responses if "error" not in item) * 1000.0 / success_count
        if success_count
        else 0.0
    )
    def avg_timing(name: str) -> float:
        return _avg_metric(timings, name)

    def avg_function_timing(name: str) -> float:
        return _avg_metric(function_timings, name)

    def avg_server_timing(name: str) -> float:
        return _avg_metric(server_timings, name)

    recall = sum(item.get("recall", 0.0) for item in responses if "error" not in item) / success_count if success_count else 0.0
    function_request_ms = avg_timing("remote_invoke")
    function_ann_search_ms = avg_function_timing("faiss_search") or avg_function_timing("ann_search_and_format")
    server_total_ms = avg_timing("total")
    server_candidate_stage_ms = avg_timing("candidates")
    server_rerank_ms = avg_timing("rerank")
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "query_count": query_count,
        "success_count": success_count,
        "error_count": error_count,
        "concurrent_requests": concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": k,
        "candidate_k": candidate_k,
        "ef_search": ef_search,
        "entrypoint": "server",
        "local_count": sum(1 for plan in plans if plan.get("mode") == "local"),
        "faas_count": sum(1 for plan in plans if plan.get("mode") == "faas"),
        "cold_start_num": len(cold_start_load_ms),
        "avg_cold_start_load_ms": round(avg_cold_start_load_ms, 3),
        "avg_entry_request_ms": round(avg_entry_request_ms, 3),
        "avg_function_request_ms": round(function_request_ms, 3),
        "avg_function_handler_ms": round(avg_function_timing("handler_total"), 3),
        "avg_function_ann_search_ms": round(function_ann_search_ms, 3),
        "avg_function_rerank_ms": 0.0,
        "avg_server_total_ms": round(server_total_ms, 3),
        "avg_server_candidate_stage_ms": round(server_candidate_stage_ms, 3),
        "avg_server_rerank_ms": round(server_rerank_ms, 3),
    }


def write_timing_diagnostics(
    path: Path,
    responses: list[dict],
    summary: dict,
    batch_start_wall_time: float,
) -> None:
    """Write stage distributions needed to locate client/VM queueing."""

    successful = [item for item in responses if "error" not in item]

    def values(getter) -> list[float]:
        result = []
        for item in successful:
            value = getter(item)
            if value is not None:
                result.append(float(value))
        return result

    client_ms = values(lambda item: float(item.get("client_elapsed_s", 0.0)) * 1000.0)
    client_dispatch_ms = values(
        lambda item: (
            float(item["_client_started_at_s"])
            - batch_start_wall_time
            - float(item.get("_planned_offset_s") or 0.0)
        )
        * 1000.0
    )
    client_to_asgi_ms = values(lambda item: item.get("_client_to_asgi_ms"))
    vm_http_ms = values(lambda item: item.get("_vm_http_process_ms"))
    server_to_client_ms = values(lambda item: item.get("_server_to_client_ms"))
    worker_active = values(lambda item: item.get("_worker_active_at_entry"))
    app_total_ms = values(lambda item: item.get("timings_ms", {}).get("total", 0.0))
    plan_ms = values(lambda item: item.get("timings_ms", {}).get("plan", 0.0))
    candidate_ms = values(lambda item: item.get("timings_ms", {}).get("candidates", 0.0))
    remote_ms = values(lambda item: item.get("timings_ms", {}).get("remote_invoke"))
    rerank_ms = values(lambda item: item.get("timings_ms", {}).get("rerank", 0.0))

    client_minus_vm = _differences(client_ms, vm_http_ms)
    vm_minus_app = _differences(vm_http_ms, app_total_ms)
    app_unaccounted = [
        total - plan - candidate - rerank
        for total, plan, candidate, rerank in zip(app_total_ms, plan_ms, candidate_ms, rerank_ms)
    ]

    lines = [
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"query_count={len(responses)} success_count={len(successful)}",
        f"vm_http_samples={len(vm_http_ms)}/{len(successful)}",
        "stage_ms:",
    ]
    for name, data in (
        ("client_elapsed", client_ms),
        ("client_dispatch_delay", client_dispatch_ms),
        ("client_to_asgi", client_to_asgi_ms),
        ("vm_http_process", vm_http_ms),
        ("server_to_client", server_to_client_ms),
        ("worker_active_at_entry", worker_active),
        ("search_service_total", app_total_ms),
        ("plan", plan_ms),
        ("candidates", candidate_ms),
        ("remote_invoke", remote_ms),
        ("rerank", rerank_ms),
        ("client_minus_vm_http", client_minus_vm),
        ("vm_http_minus_search_service", vm_minus_app),
        ("search_service_unaccounted", app_unaccounted),
    ):
        lines.append(f"{name}={_distribution(data)}")

    workers: dict[int, list[dict]] = {}
    for item in successful:
        worker_pid = item.get("_worker_pid")
        if isinstance(worker_pid, int):
            workers.setdefault(worker_pid, []).append(item)
    lines.append(f"worker_count={len(workers)}")
    for worker_pid, items in sorted(workers.items()):
        worker_client = [float(item.get("client_elapsed_s", 0.0)) * 1000.0 for item in items]
        worker_vm_http = [float(item["_vm_http_process_ms"]) for item in items if "_vm_http_process_ms" in item]
        worker_admission = [float(item["_client_to_asgi_ms"]) for item in items if "_client_to_asgi_ms" in item]
        worker_peak = max((float(item.get("_worker_max_active", 0.0)) for item in items), default=0.0)
        local_count = sum(1 for item in items if item.get("plan", {}).get("mode") == "local")
        lines.append(
            f"worker pid={worker_pid} requests={len(items)} local={local_count} "
            f"faas={len(items) - local_count} peak_active={worker_peak:.0f} "
            f"client_elapsed=({_distribution(worker_client)}) "
            f"client_to_asgi=({_distribution(worker_admission)}) "
            f"vm_http_process=({_distribution(worker_vm_http)})"
        )
    lines.append("clock_note=client_to_asgi and server_to_client require synchronized client/server clocks")
    lines.append(
        "summary="
        f"avg_entry_ms:{summary['avg_entry_request_ms']} "
        f"avg_server_total_ms:{summary['avg_server_total_ms']} "
        f"avg_function_request_ms:{summary['avg_function_request_ms']}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n\n")


def _differences(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)] if len(left) == len(right) else []


def _distribution(data: list[float]) -> str:
    if not data:
        return "samples:0"
    ordered = sorted(data)

    def percentile(value: float) -> float:
        index = min(len(ordered) - 1, int((len(ordered) - 1) * value))
        return ordered[index]

    return (
        f"samples:{len(data)} avg:{sum(data) / len(data):.3f} "
        f"p50:{percentile(0.50):.3f} p95:{percentile(0.95):.3f} "
        f"p99:{percentile(0.99):.3f} max:{ordered[-1]:.3f}"
    )


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
    parser = argparse.ArgumentParser(description="Send dataset queries to FaasANN /search and calculate recall")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", default=os.environ.get("FAASANN_DATASET", DEFAULT_DATASET))
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--groundtruth-file", default=None)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=20,
        help="maximum client worker threads; in plan mode this is only a capacity cap, not fixed concurrency",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument("--use-faas", action="store_true", default=None)
    parser.add_argument("--use-local", dest="use_faas", action="store_false")
    parser.add_argument(
        "--send-vectors",
        dest="send_vectors",
        action="store_true",
        default=True,
        help="send raw query vectors from the selected dataset query file; this is the correct recall mode",
    )
    parser.add_argument(
        "--send-query-id",
        dest="send_vectors",
        action="store_false",
        help="send query_id instead of query vector; useful only for server debugging",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-file", default=None, help="replay query arrivals from workload-generator plan.bin")
    parser.add_argument("--no-progress", action="store_true", help="disable live sent/received progress output")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--timing-log-file", default=None)
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
        args.log_file = f"logs/run_queries_{args.dataset}.csv"
    if args.timing_log_file is None:
        args.timing_log_file = f"logs/run_queries_{args.dataset}_timing.log"
    window_label = f"{args.p99_window_seconds:g}"
    if args.p99_window_seconds <= 0:
        raise SystemExit("--p99-window-seconds must be positive")
    if args.p99_log_file is None:
        args.p99_log_file = f"logs/P99/7_24/faasann_{args.dataset}_p99_{window_label}s.csv"
    if args.latency_trace_file is None:
        args.latency_trace_file = f"logs/P99/7_24/faasann_{args.dataset}_query_trace.csv"
    return args


def main() -> None:
    args = parse_args()
    query_file = ROOT / args.query_file
    groundtruth_file = ROOT / args.groundtruth_file
    log_file = ROOT / args.log_file

    print("Entrypoint: server")
    print(f"Server URL: {args.server_url}")
    print(f"Dataset: {args.dataset}")
    print(f"Query file: {query_file}")
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"Log file: {log_file}")

    plan = load_workload_plan(args.plan_file, root=ROOT) if args.plan_file else None
    try:
        request_count = resolve_request_count(args.query_num, plan)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Reading queries from {query_file}")
    queries = read_fvecs(query_file, max_vectors=request_count)
    print(f"Read {len(queries)} query vectors")
    if len(queries) == 0:
        raise SystemExit("query file has no vectors")

    print(f"Reading groundtruth from {groundtruth_file}")
    groundtruth = read_ivecs(groundtruth_file, max_vectors=len(queries))
    print(f"Read {len(groundtruth)} groundtruth rows")
    if len(groundtruth) < len(queries):
        raise SystemExit("groundtruth row count is smaller than query count")

    print(
        f"Sending {request_count} requests with client_workers={args.concurrent_requests} "
        f"(cycling {len(queries)} query vectors, "
        f"rough per-worker requests={request_count / max(1, args.concurrent_requests):.1f})"
    )
    print(describe_workload_plan(plan, request_count))
    start_t = time.perf_counter()
    try:
        responses, batch_start_wall_time = run_queries(
            server_url=args.server_url,
            query_vectors=queries,
            groundtruth=groundtruth,
            count=request_count,
            k=args.k,
            candidate_k=args.candidate_k,
            ef_search=args.ef_search,
            use_faas=args.use_faas,
            send_vectors=args.send_vectors,
            concurrent_requests=args.concurrent_requests,
            timeout=args.timeout,
            continue_on_error=args.continue_on_error,
            plan=plan,
            show_progress=not args.no_progress,
        )
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc

    elapsed = time.perf_counter() - start_t
    errors = [item for item in responses if "error" in item]
    if errors and not args.continue_on_error:
        raise SystemExit(f"Run stopped after query error: {errors[0]['error']}")
    summary = summarize_run(
        responses,
        elapsed,
        args.k,
        args.candidate_k,
        args.ef_search,
        args.concurrent_requests,
        args.dataset,
        batch_start_wall_time,
    )
    write_rows(log_file, [summary])
    write_timing_diagnostics(ROOT / args.timing_log_file, responses, summary, batch_start_wall_time)
    write_p99_logs(
        ROOT / args.p99_log_file,
        ROOT / args.latency_trace_file,
        responses,
        batch_start_wall_time,
        args.dataset,
        args.p99_window_seconds,
        method="faasann",
    )
    print(
        f"Finished: queries={summary['query_count']}, "
        f"success={summary['success_count']}, errors={summary['error_count']}, "
        f"concurrency={summary['concurrent_requests']}, "
        f"elapsed={summary['client_elapsed_s']:.3f}s, "
        f"qps={summary['qps_client']:.2f}, "
        f"average_recall@{args.k}={summary['recall']:.4f}, "
        f"local={summary['local_count']}, faas={summary['faas_count']}, "
        f"cold_start_num={summary['cold_start_num']}"
    )
    for item in errors[:5]:
        print(f"Error query_id={item['query_id']}: {item['error']}")
    print(f"Log saved to {log_file}")
    print(f"Timing log saved to {ROOT / args.timing_log_file}")
    print(f"P99 time series saved to {ROOT / args.p99_log_file}")
    print(f"Per-query latency trace saved to {ROOT / args.latency_trace_file}")


if __name__ == "__main__":
    main()
