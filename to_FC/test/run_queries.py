# -*- coding: utf-8 -*-
"""Evaluate Client -> FC Candidate Search -> VM Server -> Client callback flow."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workload-generator"))

from vectors.fvecs import read_fvecs
from vectors.ivecs import read_ivecs
from workload_plan import QueryProgress, describe_workload_plan, load_workload_plan, resolve_request_count, submit_with_plan


DEFAULT_DATASET = "sift100w"


class CallbackCollector:
    def __init__(self, expected_count: int, progress: QueryProgress | None = None) -> None:
        self.expected_count = expected_count
        self.progress = progress
        self._items: dict[str, dict] = {}
        self._queue: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()

    def put(self, item: dict) -> None:
        request_id = str(item.get("request_id") or "")
        if not request_id:
            return
        item["client_callback_received_at"] = time.perf_counter()
        with self._lock:
            if request_id in self._items:
                self._items[request_id] = item
                return
            self._items[request_id] = item
            progress = self.progress
        self._queue.put(item)
        if progress is not None:
            progress.mark_received()

    def set_expected_count(self, expected_count: int) -> None:
        with self._lock:
            self.expected_count = expected_count

    def stop_progress_updates(self) -> None:
        with self._lock:
            self.progress = None

    def wait_all(self, timeout: float) -> list[dict]:
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                if len(self._items) >= self.expected_count:
                    return list(self._items.values())
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                with self._lock:
                    return list(self._items.values())
            try:
                self._queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                pass


class CallbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], collector: CallbackCollector) -> None:
        self.collector = collector
        super().__init__(server_address, CallbackHandler)


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/callback":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json_body()
            self.server.collector.put(body)  # type: ignore[attr-defined]
            self._send_json({"status": "ok", "request_id": body.get("request_id")})
        except Exception as exc:
            self._send_json({"error": str(exc), "error_type": type(exc).__name__}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().strip("/")
    if not dataset or "/" in dataset or dataset in {".", ".."}:
        raise SystemExit("dataset must be one directory name, for example: sift100w or gist")
    return dataset


def dataset_file_prefix(dataset: str) -> str:
    return "sift" if dataset == "sift100w" else dataset


def default_query_file(dataset: str) -> str:
    prefix = dataset_file_prefix(dataset)
    return f"data/{dataset}/{prefix}_query.fvecs"


def default_groundtruth_file(dataset: str) -> str:
    prefix = dataset_file_prefix(dataset)
    return f"data/{dataset}/{prefix}_groundtruth.ivecs"


def post_json(url: str, payload: dict, timeout: float, *, no_proxy: bool = False) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        if no_proxy:
            opener = build_opener(ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def calculate_recall(result_ids: list[int], truth_ids: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(result_ids[:k]).intersection(set(truth_ids[:k]))) / k


def run_queries(
    args: argparse.Namespace,
    query_vectors,
    groundtruth,
    request_count: int,
    plan,
) -> tuple[list[dict], list[dict], float, float]:
    if len(query_vectors) == 0:
        raise ValueError("no query vectors loaded")

    progress = QueryProgress(request_count, enabled=not args.no_progress)
    collector = CallbackCollector(expected_count=request_count, progress=progress)
    callback_server = CallbackHTTPServer((args.callback_host, args.callback_port), collector)
    actual_port = callback_server.server_address[1]
    callback_url = args.client_callback_url or f"http://{args.callback_advertise_host}:{actual_port}/callback"
    server_thread = threading.Thread(target=callback_server.serve_forever, name="to-fc-client-callback", daemon=True)
    server_thread.start()

    submitted_at: dict[str, float] = {}
    acks: list[dict] = []
    submit_errors: list[dict] = []

    def submit(i: int) -> dict:
        request_id = f"query-{i}"
        source_query_id = i % len(query_vectors)
        progress.mark_sent()
        payload = {
            "type": "search_callback",
            "request_id": request_id,
            "query": query_vectors[source_query_id].astype("float32").tolist(),
            "k": args.k,
            "candidate_k": args.candidate_k,
            "ef_search": args.ef_search,
            "rerank_server_url": args.rerank_server_url.rstrip("/"),
            "client_callback_url": callback_url,
            "rerank_accept_timeout": args.rerank_accept_timeout,
        }
        start = time.perf_counter()
        submitted_at[request_id] = start
        try:
            ack = post_json(args.function_url.rstrip("/"), payload, args.fc_accept_timeout, no_proxy=args.no_proxy)
            ack["request_id"] = request_id
            ack["fc_accept_elapsed_s"] = time.perf_counter() - start
            return ack
        except Exception as exc:
            return {
                "request_id": request_id,
                "error": str(exc),
                "fc_accept_elapsed_s": time.perf_counter() - start,
            }

    try:
        start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
            futures, batch_start_wall_time = submit_with_plan(executor, request_count, submit, plan)
            for future in as_completed(futures):
                item = future.result()
                if "error" in item:
                    submit_errors.append(item)
                    progress.mark_received()
                else:
                    acks.append(item)
                if "error" in item and not args.continue_on_error:
                    raise RuntimeError(item["error"])

        collector.set_expected_count(len(acks))
        results = collector.wait_all(args.result_timeout)
        collector.stop_progress_updates()
        missing_results = _missing_callback_errors(
            acks=acks,
            results=results,
            submitted_at=submitted_at,
            groundtruth=groundtruth,
            k=args.k,
        )
        for _item in missing_results:
            progress.mark_received()
        results.extend(missing_results)
        elapsed = time.perf_counter() - start_all
    finally:
        progress.close()
        callback_server.shutdown()
        callback_server.server_close()
        server_thread.join(timeout=2.0)

    for item in results:
        request_id = str(item.get("request_id") or "")
        start = submitted_at.get(request_id)
        if start is not None and "client_callback_received_at" in item:
            item["client_final_result_elapsed_s"] = item["client_callback_received_at"] - start
        query_idx = _query_index_from_request_id(request_id)
        if query_idx is not None and len(groundtruth) > 0:
            source_query_id = query_idx % len(groundtruth)
            ids = [int(result["id"]) for result in item.get("results", [])]
            item["query_id"] = query_idx
            item["source_query_id"] = source_query_id
            item["result_ids"] = ids
            item["recall"] = calculate_recall(ids, groundtruth[source_query_id].tolist(), args.k)

    for item in submit_errors:
        query_idx = _query_index_from_request_id(str(item.get("request_id") or ""))
        item["query_id"] = query_idx
        item["source_query_id"] = query_idx % len(groundtruth) if query_idx is not None and len(groundtruth) > 0 else None
        item["result_ids"] = []
        item["recall"] = 0.0
    return results + submit_errors, acks, elapsed, batch_start_wall_time


def _missing_callback_errors(
    *,
    acks: list[dict],
    results: list[dict],
    submitted_at: dict[str, float],
    groundtruth,
    k: int,
) -> list[dict]:
    received_ids = {str(item.get("request_id") or "") for item in results}
    missing: list[dict] = []
    for ack in acks:
        request_id = str(ack.get("request_id") or "")
        if not request_id or request_id in received_ids:
            continue
        query_idx = _query_index_from_request_id(request_id)
        source_query_id = query_idx % len(groundtruth) if query_idx is not None and len(groundtruth) > 0 else None
        start = submitted_at.get(request_id)
        item = {
            "request_id": request_id,
            "query_id": query_idx,
            "source_query_id": source_query_id,
            "result_ids": [],
            "recall": 0.0,
            "error": "callback timeout before final result",
        }
        if start is not None:
            item["client_final_result_elapsed_s"] = time.perf_counter() - start
        missing.append(item)
    return missing


def summarize(args: argparse.Namespace, responses: list[dict], acks: list[dict], elapsed: float, batch_start_wall_time: float) -> dict:
    query_count = len(responses)
    success = [item for item in responses if "error" not in item and item.get("results") is not None]
    success_count = len(success)
    error_count = query_count - success_count
    recall = sum(float(item.get("recall") or 0.0) for item in success) / success_count if success_count else 0.0
    function_timings = [item.get("function_timings_ms", {}) for item in responses]
    server_timings = [item.get("server_timings_ms", {}) for item in responses]
    metrics = [item.get("function_metrics", {}) for item in responses]
    cold_loads = cold_start_load_times(responses, batch_start_wall_time)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "query_count": query_count,
        "success_count": success_count,
        "error_count": error_count,
        "concurrent_requests": args.concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "qps_client_final_result": round(success_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "ef_search": args.ef_search,
        "entrypoint": "function_callback",
        "cold_start_num": len(cold_loads),
        "avg_cold_start_load_ms": round(sum(cold_loads.values()) / len(cold_loads), 3) if cold_loads else 0.0,
        "avg_fc_accept_request_ms": round(_avg(ack.get("fc_accept_elapsed_s", 0.0) * 1000.0 for ack in acks), 3),
        "avg_client_final_result_ms": round(
            _avg(item.get("client_final_result_elapsed_s", 0.0) * 1000.0 for item in success), 3
        ),
        "avg_function_handler_ms": round(_avg_metric(function_timings, "handler_total"), 3),
        "avg_function_ann_search_ms": round(_avg_metric(function_timings, "faiss_search"), 3),
        "avg_function_to_vm_accept_ms": round(_avg_metric(function_timings, "vm_accept_request"), 3),
        "avg_server_rerank_ms": round(_avg_metric(server_timings, "rerank"), 3),
        "avg_server_total_before_callback_ms": round(_avg_metric(server_timings, "total_before_callback"), 3),
        "avg_candidate_count": round(_avg_metric(metrics, "candidate_count"), 3),
    }


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


def _avg(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


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


def _query_index_from_request_id(request_id: str) -> int | None:
    if not request_id.startswith("query-"):
        return None
    try:
        return int(request_id.split("-", 1)[1])
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Client -> FC -> VM -> Client callback queries")
    parser.add_argument("--function-url", default="http://127.0.0.1:9000")
    parser.add_argument("--rerank-server-url", default="http://127.0.0.1:8081")
    parser.add_argument("--client-callback-url", default=None)
    parser.add_argument("--callback-host", default="0.0.0.0")
    parser.add_argument("--callback-advertise-host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=18080)
    parser.add_argument("--dataset", default=os.environ.get("FAASANN_DATASET", DEFAULT_DATASET))
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--groundtruth-file", default=None)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="maximum client worker threads; in plan mode this is only a capacity cap, not fixed concurrency",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=130)
    parser.add_argument("--ef-search", type=int, default=130)
    parser.add_argument("--fc-accept-timeout", type=float, default=60.0)
    parser.add_argument("--rerank-accept-timeout", type=float, default=10.0)
    parser.add_argument("--result-timeout", type=float, default=650.0)
    parser.add_argument("--plan-file", default=None, help="replay query arrivals from workload-generator plan.bin")
    parser.add_argument("--no-progress", action="store_true", help="disable live sent/received progress output")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-proxy", action="store_true", default=True)
    args = parser.parse_args()
    args.dataset = normalize_dataset(args.dataset)
    if args.query_file is None:
        args.query_file = default_query_file(args.dataset)
    if args.groundtruth_file is None:
        args.groundtruth_file = default_groundtruth_file(args.dataset)
    if args.log_file is None:
        args.log_file = f"logs/run_queries_to_FC_{args.dataset}.csv"
    return args


def main() -> None:
    args = parse_args()
    query_file = ROOT / args.query_file
    groundtruth_file = ROOT / args.groundtruth_file
    log_file = ROOT / args.log_file

    print("Flow: Client -> FC Candidate Search -> VM Server -> Client")
    print(f"Function URL: {args.function_url}")
    print(f"Rerank server URL: {args.rerank_server_url}")
    print(f"Client callback URL: {args.client_callback_url or f'http://{args.callback_advertise_host}:{args.callback_port}/callback'}")
    print(f"Dataset: {args.dataset}")
    print(f"Query file: {query_file}")
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"Log file: {log_file}")

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
        f"Sending {request_count} requests through FC callback flow with client_workers={args.concurrent_requests} "
        f"(cycling {len(queries)} query vectors, "
        f"rough per-worker requests={request_count / max(1, args.concurrent_requests):.1f})"
    )
    print(describe_workload_plan(plan, request_count))
    if request_count > len(queries):
        print(f"Replaying {request_count} requests by cycling {len(queries)} query vectors")

    responses, acks, elapsed, batch_start_wall_time = run_queries(args, queries, groundtruth, request_count, plan)
    summary = summarize(args, responses, acks, elapsed, batch_start_wall_time)
    write_rows(log_file, [summary])

    print(
        f"Finished: queries={summary['query_count']}, success={summary['success_count']}, "
        f"errors={summary['error_count']}, qps={summary['qps_client_final_result']:.2f}, "
        f"recall@{args.k}={summary['recall']:.4f}, cold_start_num={summary['cold_start_num']}"
    )
    for item in responses:
        if "error" in item:
            print(f"Error request_id={item.get('request_id')}: {item['error']}")
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
