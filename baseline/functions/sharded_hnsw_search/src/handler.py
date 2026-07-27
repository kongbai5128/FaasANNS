# -*- coding: utf-8 -*-
"""Distributed sharded HNSW baseline handler.

Shard 0 is the coordinator. It searches its local partition, sends
``type=local_search`` requests to peer shard servers, merges all candidates, and
returns the final top-k.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from index_loader import SHARD_ID, index_status, local_search, warmup


DEFAULT_PEER_ENDPOINTS = ",".join(
    [
        "http://sharding-fazozlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-faznzlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-fazmzlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-faztzlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-fazszlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-fazrzlktgc.cn-hongkong-vpc.fcapp.run",
        "http://sharding-fazqzlktgc.cn-hongkong-vpc.fcapp.run",
    ]
)
PEER_ENDPOINTS = [
    item.strip().rstrip("/")
    for item in os.environ.get("FAASANN_PEER_ENDPOINTS", DEFAULT_PEER_ENDPOINTS).split(",")
    if item.strip()
]
PEER_TIMEOUT = float(os.environ.get("FAASANN_PEER_TIMEOUT", "120"))
PEER_WORKERS = int(os.environ.get("FAASANN_PEER_WORKERS", str(max(len(PEER_ENDPOINTS), 1))))


def handler(event: bytes | str | dict, context: Any = None) -> dict | list[bytes]:
    """统一入口：兼容 FC 事件调用和本地 HTTP 调用。"""

    if _is_wsgi_request(event, context):
        return _handle_wsgi_request(event, context)
    return _handle_payload(_decode_payload(event))


def _is_wsgi_request(event: Any, context: Any) -> bool:
    """判断当前调用是否来自 FC Web 函数的 WSGI 适配层。"""

    return isinstance(event, dict) and callable(context) and "wsgi.input" in event


def _handle_wsgi_request(environ: dict, start_response: Any) -> list[bytes]:
    """把 WSGI 请求体解析成普通 payload，再交给核心 handler。"""

    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(content_length) if content_length > 0 else b"{}"
    result = _handle_payload(_decode_payload(body))
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
    """Handle status, warmup, local shard search, and coordinator search."""

    handler_start = time.perf_counter()
    request_type = payload.get("type")
    if request_type == "status":
        return {"status": "ok", "index": index_status(), "peers": PEER_ENDPOINTS}

    if request_type == "warmup":
        return _handle_warmup(payload, handler_start)

    if request_type == "local_search":
        return _handle_local_search(payload, handler_start)

    if SHARD_ID != 0:
        raise ValueError("only shard 0 can handle coordinator search requests; use type=local_search for worker shards")
    return _handle_coordinator_search(payload, handler_start)


def _handle_local_search(payload: dict, handler_start: float) -> dict:
    """Search only this server's partition."""

    candidates, timings = local_search(
        query=payload["query"],
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
            "shard_id": status.get("shard_id"),
            "shard_count": status.get("shard_count"),
            "index_file_size_bytes": status.get("index_file_size_bytes"),
            "index_load_ms": status.get("index_load_ms"),
        },
    }


def _handle_coordinator_search(payload: dict, handler_start: float) -> dict:
    """Search shard 0 locally, fan out to peers, and merge all results."""

    candidate_k = int(payload["candidate_k"])
    fanout_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="faasann-coordinator") as executor:
        peer_future = executor.submit(_query_peers, payload)
        local_response = _handle_local_search({**payload, "type": "local_search"}, time.perf_counter())
        peer_responses = peer_future.result()
    fanout_ms = _elapsed_ms(fanout_start)

    merge_start = time.perf_counter()
    candidates = _merge_candidates([local_response, *peer_responses], candidate_k)
    merge_ms = _elapsed_ms(merge_start)

    status = index_status()
    shard_metrics = _collect_shard_metrics(local_response, peer_responses)
    timings = _coordinator_timings(local_response, peer_responses)
    timings["coordinator_fanout"] = fanout_ms
    timings["coordinator_merge"] = merge_ms
    timings["handler_total"] = _elapsed_ms(handler_start)
    return {
        "request_id": payload.get("request_id"),
        "candidates": candidates,
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
        "timings_ms": timings,
        "function_metrics": {
            "candidate_count": len(candidates),
            "shard_id": status.get("shard_id"),
            "shard_count": status.get("shard_count"),
            "peer_count": len(PEER_ENDPOINTS),
            "index_file_size_bytes": status.get("index_file_size_bytes"),
            "index_load_ms": status.get("index_load_ms"),
            "shards": shard_metrics,
            "max_shard_index_load_ms": _max_metric(shard_metrics, "index_load_ms"),
        },
    }


def _handle_warmup(payload: dict, handler_start: float) -> dict:
    """Warm local shard, or let shard 0 warm all shards in parallel."""

    if payload.get("local_only") or SHARD_ID != 0:
        return _handle_local_warmup(payload, handler_start)

    fanout_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="faasann-warmup") as executor:
        peer_future = executor.submit(_query_peer_warmups, payload)
        local_response = _handle_local_warmup({**payload, "local_only": True}, time.perf_counter())
        peer_responses = peer_future.result()
    fanout_ms = _elapsed_ms(fanout_start)

    status = index_status()
    shard_metrics = _collect_shard_metrics(local_response, peer_responses)
    timings = _coordinator_warmup_timings(local_response, peer_responses)
    timings["coordinator_fanout"] = fanout_ms
    timings["handler_total"] = _elapsed_ms(handler_start)
    return {
        "status": "ok",
        "index": status,
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
        "timings_ms": timings,
        "function_metrics": {
            "shard_id": status.get("shard_id"),
            "shard_count": status.get("shard_count"),
            "peer_count": len(PEER_ENDPOINTS),
            "index_load_ms": status.get("index_load_ms"),
            "shards": shard_metrics,
            "max_shard_index_load_ms": _max_metric(shard_metrics, "index_load_ms"),
        },
    }


def _handle_local_warmup(payload: dict, handler_start: float) -> dict:
    """Warm only this server's local partition index."""

    warmup_start = time.perf_counter()
    warmup()
    status = index_status()
    timings = {"warmup_total": _elapsed_ms(warmup_start), "handler_total": _elapsed_ms(handler_start)}
    return {
        "status": "ok",
        "request_id": payload.get("request_id"),
        "index": status,
        "cold_start_id": status.get("cold_start_id"),
        "index_loaded_at": status.get("index_loaded_at"),
        "timings_ms": timings,
        "function_metrics": {
            "candidate_count": 0,
            "shard_id": status.get("shard_id"),
            "shard_count": status.get("shard_count"),
            "index_file_size_bytes": status.get("index_file_size_bytes"),
            "index_load_ms": status.get("index_load_ms"),
        },
    }


def _query_peers(payload: dict) -> list[dict]:
    """Send local_search requests to peer shard servers in parallel."""

    if not PEER_ENDPOINTS:
        return []
    peer_payload = {
        "type": "local_search",
        "request_id": payload.get("request_id"),
        "query": payload["query"],
        "candidate_k": int(payload["candidate_k"]),
        "ef_search": int(payload.get("ef_search") or 0),
    }
    with ThreadPoolExecutor(max_workers=max(PEER_WORKERS, 1), thread_name_prefix="faasann-peer") as executor:
        futures = [executor.submit(_post_json, endpoint, peer_payload) for endpoint in PEER_ENDPOINTS]
        return [future.result() for future in as_completed(futures)]


def _query_peer_warmups(payload: dict) -> list[dict]:
    """Send warmup requests to peer shard servers in parallel."""

    if not PEER_ENDPOINTS:
        return []
    peer_payload = {
        "type": "warmup",
        "local_only": True,
        "request_id": payload.get("request_id"),
    }
    with ThreadPoolExecutor(max_workers=max(PEER_WORKERS, 1), thread_name_prefix="faasann-peer-warmup") as executor:
        futures = [executor.submit(_post_json, endpoint, peer_payload) for endpoint in PEER_ENDPOINTS]
        return [future.result() for future in as_completed(futures)]


def _post_json(endpoint: str, payload: dict) -> dict:
    """POST JSON to one peer endpoint."""

    start = time.perf_counter()
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=PEER_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"peer {endpoint} returned HTTP {exc.code}: {body[:500]}") from exc
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(f"peer {endpoint} request failed: {exc}") from exc
    result["_peer_endpoint"] = endpoint
    result["_peer_request_ms"] = _elapsed_ms(start)
    return result


def _merge_candidates(responses: list[dict], candidate_k: int) -> list[dict]:
    """Merge candidates returned by local and peer shard searches."""

    candidates = []
    for response in responses:
        candidates.extend(response.get("candidates", []))
    candidates.sort(key=lambda item: float(item["approx_score"]))
    return candidates[:candidate_k]


def _coordinator_timings(local_response: dict, peer_responses: list[dict]) -> dict[str, float]:
    """Build explicit coordinator timing fields from local and peer responses."""

    local_timings = local_response.get("timings_ms", {})
    peer_timings = [response.get("timings_ms", {}) for response in peer_responses]
    peer_request_ms = [_as_float(response.get("_peer_request_ms")) for response in peer_responses]
    peer_request_ms = [value for value in peer_request_ms if value is not None]
    peer_faiss_ms = [_as_float(timing.get("faiss_search")) for timing in peer_timings]
    peer_faiss_ms = [value for value in peer_faiss_ms if value is not None]
    all_faiss_ms = [_as_float(local_timings.get("faiss_search")), *peer_faiss_ms]
    all_faiss_ms = [value for value in all_faiss_ms if value is not None]
    return {
        "shard0_faiss_search": _as_float(local_timings.get("faiss_search")) or 0.0,
        "shard0_search_total": _as_float(local_timings.get("search_total")) or 0.0,
        "shard0_handler_total": _as_float(local_timings.get("handler_total")) or 0.0,
        "peer_request_max": max(peer_request_ms) if peer_request_ms else 0.0,
        "peer_request_avg": sum(peer_request_ms) / len(peer_request_ms) if peer_request_ms else 0.0,
        "peer_request_sum": sum(peer_request_ms),
        "peer_faiss_search_max": max(peer_faiss_ms) if peer_faiss_ms else 0.0,
        "peer_faiss_search_avg": sum(peer_faiss_ms) / len(peer_faiss_ms) if peer_faiss_ms else 0.0,
        "all_shard_faiss_search_max": max(all_faiss_ms) if all_faiss_ms else 0.0,
        "all_shard_faiss_search_sum": sum(all_faiss_ms),
    }


def _coordinator_warmup_timings(local_response: dict, peer_responses: list[dict]) -> dict[str, float]:
    """Build explicit warmup timing fields from local and peer responses."""

    local_timings = local_response.get("timings_ms", {})
    peer_timings = [response.get("timings_ms", {}) for response in peer_responses]
    peer_request_ms = [_as_float(response.get("_peer_request_ms")) for response in peer_responses]
    peer_request_ms = [value for value in peer_request_ms if value is not None]
    peer_warmup_ms = [_as_float(timing.get("warmup_total")) for timing in peer_timings]
    peer_warmup_ms = [value for value in peer_warmup_ms if value is not None]
    all_warmup_ms = [_as_float(local_timings.get("warmup_total")), *peer_warmup_ms]
    all_warmup_ms = [value for value in all_warmup_ms if value is not None]
    return {
        "shard0_warmup": _as_float(local_timings.get("warmup_total")) or 0.0,
        "peer_request_max": max(peer_request_ms) if peer_request_ms else 0.0,
        "peer_request_avg": sum(peer_request_ms) / len(peer_request_ms) if peer_request_ms else 0.0,
        "peer_warmup_max": max(peer_warmup_ms) if peer_warmup_ms else 0.0,
        "peer_warmup_avg": sum(peer_warmup_ms) / len(peer_warmup_ms) if peer_warmup_ms else 0.0,
        "all_shard_warmup_max": max(all_warmup_ms) if all_warmup_ms else 0.0,
    }


def _collect_shard_metrics(local_response: dict, peer_responses: list[dict]) -> list[dict]:
    """Collect per-shard cold-start, search, and request metrics."""

    return [_response_shard_metric(local_response), *[_response_shard_metric(response) for response in peer_responses]]


def _response_shard_metric(response: dict) -> dict:
    function_metrics = response.get("function_metrics", {})
    timings = response.get("timings_ms", {})
    return {
        "shard_id": function_metrics.get("shard_id"),
        "candidate_count": function_metrics.get("candidate_count"),
        "cold_start_id": response.get("cold_start_id"),
        "index_loaded_at": response.get("index_loaded_at"),
        "index_load_ms": function_metrics.get("index_load_ms"),
        "faiss_search_ms": timings.get("faiss_search"),
        "search_total_ms": timings.get("search_total"),
        "handler_ms": timings.get("handler_total"),
        "peer_endpoint": response.get("_peer_endpoint"),
        "peer_request_ms": response.get("_peer_request_ms"),
    }


def _max_metric(items: list[dict], name: str) -> float:
    values = [_as_float(item.get(name)) for item in items]
    values = [value for value in values if value is not None]
    return max(values) if values else 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _elapsed_ms(start: float) -> float:
    """返回从 start 到当前时刻的毫秒数。"""

    return round((time.perf_counter() - start) * 1000.0, 3)
