"""FastAPI 中间件。

当前中间件给每个响应添加 `x-process-time-ms`，用于快速观察端到端 HTTP 请求耗时。
后续可以在这里扩展 request id、访问日志、限流和 tracing。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response


async def add_process_time_header(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - start) * 1000.0:.3f}"
    return response
