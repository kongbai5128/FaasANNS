# -*- coding: utf-8 -*-
"""Custom HTTP runtime for the to_FC candidate callback function."""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "python"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

DEFAULT_DATASET = "sift100w"


def _configure_dataset_from_argv(argv: list[str]) -> str:
    if len(argv) > 2:
        raise SystemExit("usage: python3 app.py [dataset]")
    dataset = argv[1] if len(argv) == 2 else os.environ.get("FAASANN_DATASET", DEFAULT_DATASET)
    dataset = dataset.strip().strip("/")
    if not dataset or "/" in dataset or dataset in {".", ".."}:
        raise SystemExit("dataset must be one directory name, for example: sift100w or gist")
    os.environ["FAASANN_DATASET"] = dataset
    return dataset


DATASET = _configure_dataset_from_argv(sys.argv)

from handler import handler as callback_candidate_handler
from index_loader import index_status, warmup


DEFAULT_PORT = 9000
DEFAULT_HTTP_BACKLOG = 1024
DEFAULT_HTTP_MAX_WORKERS = 128


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be a positive integer, got {value!r}")
    return parsed


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
        self._request_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="to-fc-http")
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


class CandidateCallbackHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._send_json({"status": "ok", "index": index_status()})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            result = callback_candidate_handler(payload)
            self._send_json(result)
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            self._send_json({"error": str(exc), "error_type": type(exc).__name__}, status=500)

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


def main() -> None:
    print(f"to_FC callback candidate function loading index, dataset={DATASET}", flush=True)
    warmup()
    print(f"to_FC callback candidate function index loaded: {index_status()}", flush=True)
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or DEFAULT_PORT)
    backlog = _env_positive_int("FAASANN_HTTP_BACKLOG", DEFAULT_HTTP_BACKLOG)
    max_workers = _env_positive_int("FAASANN_HTTP_MAX_WORKERS", DEFAULT_HTTP_MAX_WORKERS)
    server = HighConcurrencyHTTPServer(
        ("0.0.0.0", port),
        CandidateCallbackHTTPHandler,
        backlog=backlog,
        max_workers=max_workers,
    )
    print(
        f"to_FC callback candidate function listening on 0.0.0.0:{port}, "
        f"dataset={DATASET}, backlog={backlog}, max_workers={max_workers}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
