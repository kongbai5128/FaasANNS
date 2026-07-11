# -*- coding: utf-8 -*-
"""阿里云函数计算 Faiss HNSW-PQ 候选召回入口。"""

from __future__ import annotations

import time
from typing import Any
import json

from index_loader import index_status, search_with_timings, warmup


def handler(event: bytes | str | dict, context: Any = None) -> dict | list[bytes]:
    """统一入口：只负责候选召回，不做 VM rerank 转发。"""

    if _is_wsgi_request(event, context):
        return _handle_wsgi_request(event, context)

    payload = _decode_payload(event)
    return _handle_payload(payload)


def _is_wsgi_request(event: Any, context: Any) -> bool:
    """判断当前调用是否来自 FC Web 函数的 WSGI 适配层。"""

    return isinstance(event, dict) and callable(context) and "wsgi.input" in event


def _handle_wsgi_request(environ: dict, start_response: Any) -> list[bytes]:
    """把 WSGI 请求体解析成普通 payload，再交给核心 handler。"""

    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(content_length) if content_length > 0 else b"{}"
    payload = _decode_payload(body)
    result = _handle_payload(payload)

    start_response("200 OK", [("Content-Type", "application/json")])
    return [json.dumps(result).encode("utf-8")]


def _decode_payload(event: bytes | str | dict) -> dict:
    """把 bytes/string/dict 三种 FC 输入统一成 dict。"""

    if isinstance(event, bytes):
        return json.loads(event.decode("utf-8"))
    if isinstance(event, str):
        return json.loads(event)
    return event


def _handle_payload(payload: dict) -> dict:
    """分发 status、warmup 和候选召回请求。"""

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
        candidate_k=int(payload["candidate_k"]),
        ef_search=int(payload["ef_search"]),
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
            "index_load_ms": status.get("index_load_ms"),
        },
    }


def _elapsed_ms(start: float) -> float:
    """返回从 start 到当前时刻的毫秒数。"""

    return round((time.perf_counter() - start) * 1000.0, 3)
