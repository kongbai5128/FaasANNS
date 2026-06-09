# -*- coding: utf-8 -*-
"""阿里云 FC Web 函数自定义运行时入口。"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "python"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from handler import handler as candidate_handler
from index_loader import index_status


DEFAULT_PORT = 9000


class CandidateSearchHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._send_json({"status": "ok", "index": index_status()})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            result = candidate_handler(payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

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
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or DEFAULT_PORT)
    server = ThreadingHTTPServer(("0.0.0.0", port), CandidateSearchHTTPHandler)
    print(f"ann_candidate_search listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
