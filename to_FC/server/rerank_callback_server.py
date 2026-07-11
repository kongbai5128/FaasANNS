# -*- coding: utf-8 -*-
"""VM-side rerank server for Client -> FC -> VM -> Client callback querying."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vectors.fvecs import read_fvecs
from vectors.vector_store import VectorStore


DEFAULT_DATASET = "sift100w"


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().strip("/")
    if not dataset or "/" in dataset or dataset in {".", ".."}:
        raise SystemExit("dataset must be one directory name, for example: sift100w or gist")
    return dataset


def dataset_file_prefix(dataset: str) -> str:
    return "sift" if dataset == "sift100w" else dataset


def default_base_file(dataset: str) -> Path:
    prefix = dataset_file_prefix(dataset)
    return ROOT / "data" / dataset / f"{prefix}_base.fvecs"


class SharedState:
    def __init__(self, vector_store: VectorStore, callback_timeout: float, callback_workers: int) -> None:
        self.vector_store = vector_store
        self.callback_timeout = callback_timeout
        self.callback_executor = ThreadPoolExecutor(max_workers=callback_workers, thread_name_prefix="to-fc-callback")
        self.accepted_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.started_at = time.time()

    def close(self) -> None:
        self.callback_executor.shutdown(wait=True, cancel_futures=False)


STATE: SharedState | None = None


class HighConcurrencyHTTPServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        backlog: int,
        max_workers: int,
    ) -> None:
        self.request_queue_size = backlog
        self._request_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="to-fc-vm-http")
        super().__init__(server_address, handler_class)

    def process_request(self, request: Any, client_address: Any) -> None:
        self._request_executor.submit(self._process_request_worker, request, client_address)

    def _process_request_worker(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        self._request_executor.shutdown(wait=True, cancel_futures=False)


class RerankCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        state = _state()
        if self.path in {"/", "/health"}:
            self._send_json(
                {
                    "status": "ok",
                    "vectors": state.vector_store.size,
                    "dimension": state.vector_store.dimension,
                    "accepted_count": state.accepted_count,
                    "completed_count": state.completed_count,
                    "failed_count": state.failed_count,
                    "started_at": state.started_at,
                }
            )
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/rerank_callback":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            payload = self._read_json_body()
            request_id = str(payload.get("request_id") or "")
            if not request_id:
                raise ValueError("request_id is required")
            callback_url = str(payload.get("client_callback_url") or "").strip()
            if not callback_url:
                raise ValueError("client_callback_url is required")

            state = _state()
            state.accepted_count += 1
            state.callback_executor.submit(_rerank_and_callback, payload)
            self._send_json({"status": "accepted", "request_id": request_id})
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


def _rerank_and_callback(payload: dict) -> None:
    state = _state()
    total_start = time.perf_counter()
    rerank_ms = 0.0
    request_id = str(payload.get("request_id") or "")
    callback_url = str(payload.get("client_callback_url") or "")
    try:
        query = np.asarray(payload["query"], dtype="float32")
        if query.shape != (state.vector_store.dimension,):
            raise ValueError(f"query dimension must be {state.vector_store.dimension}, got {query.shape}")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        k = int(payload["k"])

        rerank_start = time.perf_counter()
        results = state.vector_store.rerank(query, candidates, k)
        rerank_ms = _elapsed_ms(rerank_start)
        response = {
            "request_id": request_id,
            "results": [{"id": item.id, "score": item.score} for item in results],
            "plan": {"mode": "fc_vm_client_callback"},
            "timings_ms": {
                "server_rerank": rerank_ms,
                "server_total_before_callback": _elapsed_ms(total_start),
            },
            "server_timings_ms": {
                "rerank": rerank_ms,
                "total_before_callback": _elapsed_ms(total_start),
            },
            "function_timings_ms": payload.get("function_timings_ms", {}),
            "function_metrics": payload.get("function_metrics", {}),
            "cold_start_id": payload.get("cold_start_id"),
            "index_loaded_at": payload.get("index_loaded_at"),
            "rerank_metrics": {
                "candidate_count": len(candidates),
                "result_count": len(results),
            },
        }
        _post_json(callback_url, response, state.callback_timeout)
        state.completed_count += 1
    except Exception as exc:
        state.failed_count += 1
        error_payload = {
            "request_id": request_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "timings_ms": {
                "server_rerank": rerank_ms,
                "server_total_before_callback": _elapsed_ms(total_start),
            },
            "function_timings_ms": payload.get("function_timings_ms", {}),
            "function_metrics": payload.get("function_metrics", {}),
            "cold_start_id": payload.get("cold_start_id"),
            "index_loaded_at": payload.get("index_loaded_at"),
        }
        try:
            if callback_url:
                _post_json(callback_url, error_payload, state.callback_timeout)
        except Exception:
            pass


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"client callback HTTP {exc.code}: {body}") from exc
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(f"client callback request failed: {exc}") from exc


def _state() -> SharedState:
    if STATE is None:
        raise RuntimeError("server state is not initialized")
    return STATE


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VM rerank callback server for to_FC")
    parser.add_argument("dataset", nargs="?", default=os.environ.get("FAASANN_DATASET", DEFAULT_DATASET))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8081))
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--max-vectors", type=int, default=None)
    parser.add_argument("--http-workers", type=int, default=_env_positive_int("FAASANN_HTTP_MAX_WORKERS", 128))
    parser.add_argument("--callback-workers", type=int, default=_env_positive_int("FAASANN_CALLBACK_WORKERS", 128))
    parser.add_argument("--backlog", type=int, default=_env_positive_int("FAASANN_HTTP_BACKLOG", 1024))
    parser.add_argument("--callback-timeout", type=float, default=float(os.environ.get("FAASANN_CALLBACK_TIMEOUT", "60")))
    return parser.parse_args()


def main() -> None:
    global STATE

    args = parse_args()
    dataset = normalize_dataset(args.dataset)
    base_path = Path(args.base_path) if args.base_path else default_base_file(dataset)
    if args.dimension is None:
        vectors = read_fvecs(base_path, max_vectors=args.max_vectors)
    else:
        vectors = read_fvecs(base_path, dimension=args.dimension, max_vectors=args.max_vectors)
    STATE = SharedState(
        vector_store=VectorStore(vectors),
        callback_timeout=args.callback_timeout,
        callback_workers=args.callback_workers,
    )

    print(
        f"to_FC rerank callback server loaded vectors={STATE.vector_store.size}, "
        f"dimension={STATE.vector_store.dimension}, base_path={base_path}",
        flush=True,
    )
    server = HighConcurrencyHTTPServer(
        (args.host, args.port),
        RerankCallbackHandler,
        backlog=args.backlog,
        max_workers=args.http_workers,
    )
    print(
        f"to_FC rerank callback server listening on {args.host}:{args.port}, "
        f"http_workers={args.http_workers}, callback_workers={args.callback_workers}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        STATE.close()


if __name__ == "__main__":
    main()
