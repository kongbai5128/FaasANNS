"""FastAPI 中间件。

当前中间件给每个响应添加 `x-process-time-ms`，用于快速观察端到端 HTTP 请求耗时。
后续可以在这里扩展 request id、访问日志、限流和 tracing。
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

_active_requests = 0
_max_active_requests = 0


async def add_process_time_header(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    global _active_requests, _max_active_requests

    start = time.perf_counter()
    asgi_started_ns = time.time_ns()
    _active_requests += 1
    active_at_entry = _active_requests
    _max_active_requests = max(_max_active_requests, active_at_entry)
    try:
        response = await call_next(request)
    finally:
        _active_requests -= 1

    response_ready_ns = time.time_ns()
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - start) * 1000.0:.3f}"
    response.headers["x-faasann-worker-pid"] = str(os.getpid())
    response.headers["x-faasann-worker-active"] = str(active_at_entry)
    response.headers["x-faasann-worker-max-active"] = str(_max_active_requests)
    response.headers["x-faasann-asgi-started-ns"] = str(asgi_started_ns)
    response.headers["x-faasann-response-ready-ns"] = str(response_ready_ns)

    client_started = request.headers.get("x-faasann-client-started-ns")
    if client_started is not None:
        try:
            client_to_asgi_ms = (asgi_started_ns - int(client_started)) / 1_000_000.0
            response.headers["x-faasann-client-to-asgi-ms"] = f"{client_to_asgi_ms:.3f}"
        except ValueError:
            pass
    return response
