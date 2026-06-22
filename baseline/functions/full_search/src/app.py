# -*- coding: utf-8 -*-
"""阿里云 FC Web 函数自定义运行时入口。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

from handler import handler as full_search_handler
from index_loader import index_status, warmup


DEFAULT_PORT = 9000


class FullSearchHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._send_json({"status": "ok", "index": index_status()})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            result = full_search_handler(payload)
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
    print(f"full_search baseline loading index before listening, dataset={DATASET}", flush=True)
    warmup()
    print(f"full_search baseline index loaded: {index_status()}", flush=True)
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or DEFAULT_PORT)
    server = ThreadingHTTPServer(("0.0.0.0", port), FullSearchHTTPHandler)
    print(f"full_search baseline listening on 0.0.0.0:{port}, dataset={DATASET}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
