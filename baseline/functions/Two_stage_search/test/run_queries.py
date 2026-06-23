"""Send dataset queries directly to the two-stage Function Compute baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from vectors.fvecs import read_fvecs
from vectors.ivecs import read_ivecs


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
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def send_one_query(
    *,
    endpoint: str,
    query_id: int,
    vector: list[float],
    k: int,
    candidate_k: int,
    ef_search: int,
    timeout: float,
) -> dict:
    start = time.perf_counter()
    payload = {
        "request_id": f"query-{query_id}",
        "query": vector,
        "k": k,
        "candidate_k": candidate_k,
        "ef_search": ef_search,
    }
    response = post_json(endpoint, payload, timeout)
    result_ids = [int(item["id"]) for item in response.get("candidates", [])[:k]]

    return {
        "query_id": query_id,
        "result_ids": result_ids,
        "client_elapsed_s": time.perf_counter() - start,
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

    queries = read_fvecs(query_file, max_vectors=args.query_num)
    groundtruth = read_ivecs(groundtruth_file, max_vectors=len(queries))
    if len(groundtruth) < len(queries):
        raise SystemExit("groundtruth row count is smaller than query count")

    def submit(i: int) -> dict:
        response = send_one_query(
            endpoint=args.endpoint,
            query_id=i,
            vector=queries[i].astype("float32").tolist(),
            k=args.k,
            candidate_k=args.candidate_k,
            ef_search=args.ef_search,
            timeout=args.timeout,
        )
        response["recall"] = calculate_recall(response["result_ids"], groundtruth[i].tolist(), args.k)
        return response

    args.batch_start_wall_time = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
        futures = [executor.submit(submit, i) for i in range(len(queries))]
        for future in as_completed(futures):
            results.append(future.result())
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
    avg_entry_request_ms = sum(item["client_elapsed_s"] for item in responses) * 1000.0 / query_count if query_count else 0.0
    function_timings = [item.get("timings_ms", {}) for item in responses]
    cold_start_load_ms = cold_start_load_times(responses, batch_start_wall_time)
    avg_cold_start_load_ms = sum(cold_start_load_ms.values()) / len(cold_start_load_ms) if cold_start_load_ms else 0.0
    function_metrics = [item.get("function_metrics", {}) for item in responses]

    def avg_function_timing(name: str) -> float:
        return _avg_metric(function_timings, name)

    recall = sum(item.get("recall", 0.0) for item in responses) / query_count if query_count else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "query_count": query_count,
        "concurrent_requests": args.concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "ef_search": args.ef_search,
        "nomemory": _any_metric_truthy(function_metrics, "nomemory"),
        "cold_start_num": len(cold_start_load_ms),
        "avg_cold_start_load_ms": round(avg_cold_start_load_ms, 3),
        "avg_entry_request_ms": round(avg_entry_request_ms, 3),
        "avg_function_request_ms": round(avg_entry_request_ms, 3),
        "avg_function_handler_ms": round(avg_function_timing("handler_total"), 3),
        "avg_function_ann_search_ms": round(avg_function_timing("faiss_search"), 3),
        "avg_function_rerank_ms": round(avg_function_timing("rerank_total"), 3),
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


def _any_metric_truthy(items: list[dict], name: str) -> bool:
    for item in items:
        if not isinstance(item, dict) or name not in item:
            continue
        value = item[name]
        if isinstance(value, str):
            if value.strip().lower() in {"1", "true", "yes", "on", "nomemory"}:
                return True
            continue
        if bool(value):
            return True
    return False


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
    parser = argparse.ArgumentParser(description="Send dataset queries to the two-stage baseline")
    parser.add_argument("--endpoint", default="https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run")
    parser.add_argument("--dataset", default=os.environ.get("FAASANN_DATASET", DEFAULT_DATASET))
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--groundtruth-file", default=None)
    parser.add_argument("--query-num", type=int, default=10)
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--ef-search", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    args.dataset = normalize_dataset(args.dataset)
    if args.query_file is None:
        args.query_file = default_query_file(args.dataset)
    if args.groundtruth_file is None:
        args.groundtruth_file = default_groundtruth_file(args.dataset)
    if args.log_file is None:
        args.log_file = f"baseline/functions/Two_stage_search/test/result/run_queries_{args.dataset}.csv"
    return args


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        responses = run_queries(args)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc

    elapsed = time.perf_counter() - start
    summary = summarize_run(args, responses, elapsed, args.batch_start_wall_time)
    write_rows(ROOT / args.log_file, [summary])
    print(
        f"Finished: queries={summary['query_count']}, "
        f"concurrency={summary['concurrent_requests']}, elapsed={summary['client_elapsed_s']:.3f}s, "
        f"qps={summary['qps_client']:.2f}, recall@{args.k}={summary['recall']:.4f}, "
        f"cold_start_num={summary['cold_start_num']}"
    )


if __name__ == "__main__":
    main()
