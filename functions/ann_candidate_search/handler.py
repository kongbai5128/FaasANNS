# -*- coding: utf-8 -*-
"""阿里云函数计算 PQ 候选召回入口。"""

from __future__ import annotations

import json
from typing import Any

from index_loader import index_status, search, warmup


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
    if payload.get("type") == "status":
        return {"status": "ok", "index": index_status()}

    if payload.get("type") == "warmup":
        warmup()
        return {"status": "ok", "index": index_status()}

    candidates = search(
        query=payload["query"],
        candidate_k=int(payload["candidate_k"]),
    )
    return {
        "request_id": payload.get("request_id"),
        "candidates": candidates,
    }
