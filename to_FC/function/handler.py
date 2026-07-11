# -*- coding: utf-8 -*-
"""FC entry for Client -> FC -> VM -> Client callback querying.

This function does not return final ANN results to the original HTTP caller.
It searches candidates, sends them to the VM callback-rerank service, and
returns only an acknowledgement. The final top-k result is sent by the VM
directly to the client's callback URL.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from index_loader import index_status, search_with_timings, warmup


def handler(event: bytes | str | dict, context: Any = None) -> dict | list[bytes]:
    if _is_wsgi_request(event, context):
        return _handle_wsgi_request(event, context)
    return _handle_payload(_decode_payload(event))


def _is_wsgi_request(event: Any, context: Any) -> bool:
    return isinstance(event, dict) and callable(context) and "wsgi.input" in event


def _handle_wsgi_request(environ: dict, start_response: Any) -> list[bytes]:
    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(content_length) if content_length > 0 else b"{}"
    result = _handle_payload(_decode_payload(body))
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
    request_type = payload.get("type")
    if request_type == "status":
        return {"status": "ok", "index": index_status()}

    if request_type == "warmup":
        warmup_start = time.perf_counter()
        warmup()
        return {
            "status": "ok",
            "index": index_status(),
            "timings_ms": {"warmup_total": _elapsed_ms(warmup_start), "handler_total": _elapsed_ms(handler_start)},
        }

    if request_type not in {None, "search_callback"}:
        raise ValueError(f"unsupported request type: {request_type!r}")

    rerank_server_url = str(payload.get("rerank_server_url") or "").rstrip("/")
    client_callback_url = str(payload.get("client_callback_url") or "").strip()
    if not rerank_server_url:
        raise ValueError("rerank_server_url is required")
    if not client_callback_url:
        raise ValueError("client_callback_url is required")

    search_start = time.perf_counter()
    candidates, function_timings = search_with_timings(
        query=payload["query"],
        candidate_k=int(payload["candidate_k"]),
        ef_search=int(payload["ef_search"]),
    )
    function_timings["ann_search_and_format"] = _elapsed_ms(search_start)

    status = index_status()
    vm_payload = {
        "request_id": payload.get("request_id"),
        "query": payload["query"],
        "candidates": candidates,
        "k": int(payload["k"]),
        "client_callback_url": client_callback_url,
        "function_timings_ms": function_timings,
        "function_metrics": {
            "candidate_count": len(candidates),
            "index_file_size_bytes": status.get("index_file_size_bytes"),
            "index_load_ms": status.get("index_load_ms"),
        },
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
    }

    vm_start = time.perf_counter()
    vm_ack = _post_json(
        rerank_server_url + "/rerank_callback",
        vm_payload,
        timeout=float(payload.get("rerank_accept_timeout") or 10.0),
    )
    function_timings["vm_accept_request"] = _elapsed_ms(vm_start)
    function_timings["handler_total"] = _elapsed_ms(handler_start)

    return {
        "status": "accepted",
        "request_id": payload.get("request_id"),
        "plan": {"mode": "fc_to_vm_callback"},
        "vm_ack": vm_ack,
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
        "timings_ms": function_timings,
        "function_metrics": vm_payload["function_metrics"],
    }


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"rerank callback server HTTP {exc.code}: {body}") from exc
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(f"rerank callback server request failed: {exc}") from exc


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
