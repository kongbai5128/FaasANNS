# -*- coding: utf-8 -*-
"""阿里云 FC Web 函数自定义运行时入口。"""

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


def _configure_runtime_from_argv(argv: list[str]) -> tuple[str, bool]:
    """从启动参数/环境变量确定数据集和 memory/nomemory 模式。"""

    if len(argv) > 3:
        raise SystemExit("usage: python3 app.py [dataset] [memory|nomemory]")

    dataset = argv[1] if len(argv) >= 2 else os.environ.get("FAASANN_DATASET", DEFAULT_DATASET)
    dataset = dataset.strip().strip("/")
    if not dataset or "/" in dataset or dataset in {".", ".."}:
        raise SystemExit("dataset must be one directory name, for example: sift100w or gist")
    os.environ["FAASANN_DATASET"] = dataset

    nomemory = _env_truthy("FAASANN_NOMEMORY")
    if len(argv) == 3:
        mode = argv[2].strip().lower()
        if mode not in {"nomemory", "memory"}:
            raise SystemExit("second argument must be 'nomemory' or 'memory'")
        nomemory = mode == "nomemory"
    os.environ["FAASANN_NOMEMORY"] = "1" if nomemory else "0"
    return dataset, nomemory


def _env_truthy(name: str) -> bool:
    """按常见布尔字符串解析环境变量。"""

    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "nomemory"}


DATASET, NOMEMORY = _configure_runtime_from_argv(sys.argv)

from handler import handler as two_stage_handler
from index_loader import index_status, warmup


DEFAULT_PORT = 9000
DEFAULT_HTTP_BACKLOG = 1024
DEFAULT_HTTP_MAX_WORKERS = 128


def _env_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量，用于 HTTP backlog 和 worker 数。"""

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
    """用线程池处理 HTTP 请求，避免单线程 server 串行阻塞。"""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        backlog: int,
        max_workers: int,
    ) -> None:
        """初始化监听 backlog 和请求处理线程池。"""

        self.request_queue_size = backlog
        self._request_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="faasann-http",
        )
        super().__init__(server_address, handler_class)

    def process_request(self, request: Any, client_address: Any) -> None:
        """把每个 socket 请求交给线程池处理。"""

        self._request_executor.submit(self._process_request_worker, request, client_address)

    def _process_request_worker(self, request: Any, client_address: Any) -> None:
        """在线程池 worker 中执行 HTTPServer 原本的请求生命周期。"""

        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        """关闭监听 socket 并回收线程池。"""

        super().server_close()
        self._request_executor.shutdown(wait=True, cancel_futures=False)


class TwoStageHTTPHandler(BaseHTTPRequestHandler):
    """本地/自定义运行时 HTTP 适配器，把 HTTP 请求转给 two-stage handler。"""

    def do_GET(self) -> None:
        """提供 health/status 查询。"""

        if self.path in {"/", "/health"}:
            self._send_json({"status": "ok", "index": index_status()})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        """解析 JSON 请求体并执行 two-stage 查询。"""

        try:
            payload = self._read_json_body()
            result = two_stage_handler(payload)
            self._send_json(result)
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            self._send_json({"error": str(exc), "error_type": type(exc).__name__}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        """禁用默认访问日志，避免压测输出被刷屏。"""

        return

    def _read_json_body(self) -> dict:
        """读取 HTTP body 并解析为 JSON dict。"""

        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        """把 dict/list 响应序列化为 JSON HTTP response。"""

        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """预加载索引并启动自定义 HTTP runtime。"""

    print(f"two_stage baseline loading index before listening, dataset={DATASET}, nomemory={NOMEMORY}", flush=True)
    warmup()
    print(f"two_stage baseline index loaded: {index_status()}", flush=True)
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or DEFAULT_PORT)
    backlog = _env_positive_int("FAASANN_HTTP_BACKLOG", DEFAULT_HTTP_BACKLOG)
    max_workers = _env_positive_int("FAASANN_HTTP_MAX_WORKERS", DEFAULT_HTTP_MAX_WORKERS)
    server = HighConcurrencyHTTPServer(
        ("0.0.0.0", port),
        TwoStageHTTPHandler,
        backlog=backlog,
        max_workers=max_workers,
    )
    print(
        "two_stage baseline listening on "
        f"0.0.0.0:{port}, dataset={DATASET}, nomemory={NOMEMORY}, "
        f"backlog={backlog}, max_workers={max_workers}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
