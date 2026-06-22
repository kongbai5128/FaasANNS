# -*- coding: utf-8 -*-
"""阿里云函数计算两阶段搜索 baseline 入口。"""

from __future__ import annotations

import json
import time
from typing import Any

from index_loader import index_status, search_with_timings, warmup


def handler(event: bytes | str | dict, context: Any = None) -> dict | list[bytes]:
    if _is_wsgi_request(event, context):
        return _handle_wsgi_request(event, context)

    payload = _decode_payload(event)
    return _handle_payload(payload)


def _is_wsgi_request(event: Any, context: Any) -> bool:
    return isinstance(event, dict) and callable(context) and "wsgi.input" in event


def _handle_wsgi_request(environ: dict, start_response: Any) -> list[bytes]:
    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(content_length) if content_length > 0 else b"{}"
    payload = _decode_payload(body)
    result = _handle_payload(payload)

    start_response("200 OK", [("Content-Type", "application/json")])
    return [json.dumps(result).encode("utf-8")]


def _decode_payload(event: bytes | str | dict) -> dict:
    if isinstance(event, bytes):
        return json.loads(event.decode("utf-8"))
    if isinstance(event, str):
        return json.loads(event)
    return event


def _handle_payload(payload: dict) -> dict:
    handler_start = time.perf_counter()
    if payload.get("type") == "status":
        return {"status": "ok", "index": index_status()}

    if payload.get("type") == "warmup":
        warmup_start = time.perf_counter()
        warmup()
        return {
            "status": "ok",
            "index": index_status(),
            "timings_ms": {"warmup_total": _elapsed_ms(warmup_start), "handler_total": _elapsed_ms(handler_start)},
        }

    candidates, timings = search_with_timings(
        query=payload["query"],
        k=int(payload["k"]),
        candidate_k=int(payload["candidate_k"]),
        ef_search=int(payload.get("ef_search") or 0),
    )
    status = index_status()
    timings["handler_total"] = _elapsed_ms(handler_start)
    return {
        "request_id": payload.get("request_id"),
        "candidates": candidates,
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
        "timings_ms": timings,
        "function_metrics": {
            "candidate_count": len(candidates),
            "index_file_size_bytes": status.get("index_file_size_bytes"),
            "base_file_size_bytes": status.get("base_file_size_bytes"),
            "index_load_ms": status.get("index_load_ms"),
        },
    }


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
