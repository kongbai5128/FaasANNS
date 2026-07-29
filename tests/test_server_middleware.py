"""Request admission diagnostics exposed by the FastAPI middleware."""

from __future__ import annotations

import asyncio
import os
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

from server.middleware import add_process_time_header


def test_process_time_middleware_reports_worker_and_admission_timing() -> None:
    client_started_ns = time.time_ns() - 1_000_000
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/search",
        "headers": [(b"x-faasann-client-started-ns", str(client_started_ns).encode("ascii"))],
    }
    request = Request(scope)

    async def call_next(_request: Request):
        await asyncio.sleep(0)
        return JSONResponse({"ok": True})

    response = asyncio.run(add_process_time_header(request, call_next))

    assert response.headers["x-faasann-worker-pid"] == str(os.getpid())
    assert int(response.headers["x-faasann-worker-active"]) >= 1
    assert int(response.headers["x-faasann-worker-max-active"]) >= 1
    assert float(response.headers["x-faasann-client-to-asgi-ms"]) >= 0.0
    assert int(response.headers["x-faasann-asgi-started-ns"]) >= client_started_ns
    assert int(response.headers["x-faasann-response-ready-ns"]) >= client_started_ns
