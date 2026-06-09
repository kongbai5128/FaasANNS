"""向 FaasANN 服务器发送 SIFT 查询并计算 Recall。

这个脚本参考旧项目 /home/qian/faasann/test/hnsw/run.py 的实验方式：
读取查询向量和 groundtruth，按并发度向当前服务器的 `/search` 接口发送请求，
统计客户端 QPS、Recall@K、local/FaaS 路径数量和阶段耗时，并把最终汇总结果写入 CSV。
"""

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

ROOT = Path(__file__).resolve().parents[2]
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
    server_url: str,
    query_id: int,
    vector: list[float] | None,
    k: int,
    candidate_k: int | None,
    ef_search: int | None,
    use_faas: bool | None,
    timeout: float,
) -> dict:
    payload: dict = {
        "request_id": f"query-{query_id}",
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

    start = time.perf_counter()
    response = post_json(f"{server_url.rstrip('/')}/search", payload, timeout)
    response["client_elapsed_s"] = time.perf_counter() - start
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
) -> list[dict]:
    def submit(i: int) -> dict:
        query_id = i
        vector = query_vectors[query_id].astype("float32").tolist() if send_vectors else None
        response = send_one_query(
            server_url=server_url,
            query_id=query_id,
            vector=vector,
            k=k,
            candidate_k=candidate_k,
            ef_search=ef_search,
            use_faas=use_faas,
            timeout=timeout,
        )
        result_ids = [int(item["id"]) for item in response.get("results", [])]
        response["query_id"] = query_id
        response["result_ids"] = result_ids
        response["recall"] = calculate_recall(result_ids, groundtruth[query_id].tolist(), k)
        return response

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(submit, i) for i in range(count)]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize_run(responses: list[dict], elapsed: float, k: int, concurrent_requests: int) -> dict:
    query_count = len(responses)
    timings = [item.get("timings_ms", {}) for item in responses]
    plans = [item.get("plan", {}) for item in responses]

    def avg_timing(name: str) -> float:
        values = [float(item[name]) for item in timings if name in item]
        return sum(values) / len(values) if values else 0.0

    recall = sum(item.get("recall", 0.0) for item in responses) / query_count if query_count else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_count": query_count,
        "concurrent_requests": concurrent_requests,
        "client_elapsed_s": round(elapsed, 6),
        "qps_client": round(query_count / elapsed, 2) if elapsed > 0 else 0.0,
        "recall": round(recall, 6),
        "k": k,
        "local_count": sum(1 for plan in plans if plan.get("mode") == "local"),
        "faas_count": sum(1 for plan in plans if plan.get("mode") == "faas"),
        "faas_reasons": json.dumps(_count_plan_reasons(plans), ensure_ascii=False),
        "avg_total_ms": round(avg_timing("total"), 3),
        "avg_candidates_ms": round(avg_timing("candidates"), 3),
        "avg_rerank_ms": round(avg_timing("rerank"), 3),
        "avg_remote_invoke_ms": round(avg_timing("remote_invoke"), 3),
    }


def _count_plan_reasons(plans: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        reason = str(plan.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
    parser = argparse.ArgumentParser(description="Send SIFT queries to FaasANN /search and calculate recall")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--query-file", default="data/sift100w/sift_query.fvecs")
    parser.add_argument("--groundtruth-file", default="data/sift100w/sift_groundtruth.ivecs")
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--concurrent-requests", type=int, default=20)
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
        help="send raw query vectors from sift_query.fvecs; this is the correct recall mode",
    )
    parser.add_argument(
        "--send-query-id",
        dest="send_vectors",
        action="store_false",
        help="send query_id instead of query vector; useful only for server debugging",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log-file", default="logs/run_queries.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_file = ROOT / args.query_file
    groundtruth_file = ROOT / args.groundtruth_file
    log_file = ROOT / args.log_file

    print(f"Server URL: {args.server_url}")
    print(f"Query file: {query_file}")
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"Log file: {log_file}")

    print(f"Reading queries from {query_file}")
    queries = read_fvecs(query_file, max_vectors=args.query_num)
    print(f"Read {len(queries)} query vectors")

    print(f"Reading groundtruth from {groundtruth_file}")
    groundtruth = read_ivecs(groundtruth_file, max_vectors=len(queries))
    print(f"Read {len(groundtruth)} groundtruth rows")
    if len(groundtruth) < len(queries):
        raise SystemExit("groundtruth row count is smaller than query count")

    print(
        f"Sending {len(queries)} queries with concurrency={args.concurrent_requests} "
        f"(rough per-worker queries={len(queries) / max(1, args.concurrent_requests):.1f})"
    )
    start_t = time.perf_counter()
    try:
        responses = run_queries(
            server_url=args.server_url,
            query_vectors=queries,
            groundtruth=groundtruth,
            count=len(queries),
            k=args.k,
            candidate_k=args.candidate_k,
            ef_search=args.ef_search,
            use_faas=args.use_faas,
            send_vectors=args.send_vectors,
            concurrent_requests=args.concurrent_requests,
            timeout=args.timeout,
        )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc

    elapsed = time.perf_counter() - start_t
    summary = summarize_run(responses, elapsed, args.k, args.concurrent_requests)
    write_rows(log_file, [summary])
    print(
        f"Finished: queries={summary['query_count']}, "
        f"concurrency={summary['concurrent_requests']}, "
        f"elapsed={summary['client_elapsed_s']:.3f}s, "
        f"qps={summary['qps_client']:.2f}, "
        f"average_recall@{args.k}={summary['recall']:.4f}, "
        f"local={summary['local_count']}, faas={summary['faas_count']}, "
        f"reasons={summary['faas_reasons']}"
    )
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
