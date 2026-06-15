"""Send SIFT queries directly to the two-stage Function Compute baseline."""

from __future__ import annotations

import argparse
import csv
import json
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
    }


def calculate_recall(result_ids: list[int], truth_ids: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(result_ids[:k]).intersection(map(int, truth_ids[:k]))) / k


def run_queries(args: argparse.Namespace) -> list[dict]:
    query_file = ROOT / args.query_file
    groundtruth_file = ROOT / args.groundtruth_file
    print(f"Endpoint: {args.endpoint}")
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

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
        futures = [executor.submit(submit, i) for i in range(len(queries))]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize_run(args: argparse.Namespace, responses: list[dict], elapsed: float) -> dict:
    query_count = len(responses)
    avg_client_ms = sum(item["client_elapsed_s"] for item in responses) * 1000.0 / query_count if query_count else 0.0
    recall = sum(item.get("recall", 0.0) for item in responses) / query_count if query_count else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_count": query_count,
        "concurrent_requests": args.concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "ef_search": args.ef_search,
        "avg_client_ms": round(avg_client_ms, 3),
    }


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
    parser = argparse.ArgumentParser(description="Send SIFT queries to the two-stage baseline")
    parser.add_argument("--endpoint", default="https://base-lie-search-aobkfnfjxd.cn-hongkong.fcapp.run")
    parser.add_argument("--query-file", default="data/sift100w/sift_query.fvecs")
    parser.add_argument("--groundtruth-file", default="data/sift100w/sift_groundtruth.ivecs")
    parser.add_argument("--query-num", type=int, default=10)
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--ef-search", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-file", default="baseline/functions/Two_stage_search/test/result/run_queries.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        responses = run_queries(args)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc

    elapsed = time.perf_counter() - start
    summary = summarize_run(args, responses, elapsed)
    write_rows(ROOT / args.log_file, [summary])
    print(
        f"Finished: queries={summary['query_count']}, "
        f"concurrency={summary['concurrent_requests']}, elapsed={summary['client_elapsed_s']:.3f}s, "
        f"qps={summary['qps_client']:.2f}, recall@{args.k}={summary['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
