"""阿里云函数计算 HTTP provider。

这个 provider 将候选搜索请求 POST 到阿里云 Function Compute HTTP Trigger。
函数端只返回 PQ candidate ids 和近似分数；raw vectors 仍由 VM 侧 `vectors.VectorStore` 保存并精排。
"""

from __future__ import annotations

import asyncio
import email.utils
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
import urllib.request

from faas.payload import CandidateSearchPayload


class AliyunHTTPProvider:
    def __init__(self, endpoints: dict[str, str], timeout_seconds: float, invoke_workers: int):
        if "default" not in endpoints:
            raise RuntimeError("faas.endpoints.default is required when provider=aliyun_http")
        self.endpoint = endpoints["default"]
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(
            max_workers=invoke_workers,
            thread_name_prefix="faasann-faas-invoke",
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if not self.endpoint:
            raise RuntimeError("faas.endpoints.default is required when provider=aliyun_http")

    async def invoke(self, payload: CandidateSearchPayload) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._post_candidates, payload.to_json())

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._post_json, {"type": "warmup"})

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _post_json(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "content-type": "application/json",
                "date": email.utils.formatdate(usegmt=True),
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FaaS HTTP {exc.code}: {body}") from exc

    def _post_candidates(self, payload: dict) -> list[dict]:
        data = self._post_json(payload)
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            raise ValueError("FaaS response must be a list or an object with 'candidates'")
        if "candidates" in data:
            candidates = _require_candidates(data["candidates"])
            cold_start_id = data.get("cold_start_id")
            if cold_start_id is not None and candidates:
                candidates[0]["_cold_start_id"] = str(cold_start_id)
                if data.get("index_loaded_at") is not None:
                    candidates[0]["_index_loaded_at"] = data["index_loaded_at"]
            if isinstance(data.get("timings_ms"), dict) and candidates:
                candidates[0]["_function_timings_ms"] = data["timings_ms"]
            if isinstance(data.get("function_metrics"), dict) and candidates:
                candidates[0]["_function_metrics"] = data["function_metrics"]
            return candidates
        raise ValueError("FaaS response missing required 'candidates' field")


def _require_candidates(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("FaaS response candidates must be a list")
    return value
