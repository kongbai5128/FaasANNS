"""阿里云函数计算 HTTP provider。

这个 provider 将候选搜索请求 POST 到阿里云 Function Compute HTTP Trigger。
函数端只返回 PQ candidate ids 和近似分数；raw vectors 仍由 VM 侧 `vectors.VectorStore` 保存并精排。
"""

from __future__ import annotations

import asyncio
import email.utils
import json
from urllib.error import HTTPError
import urllib.request

from faas.payload import CandidateSearchPayload


class AliyunHTTPProvider:
    def __init__(self, endpoints: dict[str, str], timeout_seconds: float):
        if "default" not in endpoints:
            raise RuntimeError("faas.endpoints.default is required when provider=aliyun_http")
        self.endpoint = endpoints["default"]
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if not self.endpoint:
            raise RuntimeError("faas.endpoints.default is required when provider=aliyun_http")

    async def invoke(self, payload: CandidateSearchPayload) -> list[dict]:
        return await asyncio.to_thread(self._post_candidates, payload.to_json())

    async def warmup(self) -> None:
        await asyncio.to_thread(self._post_json, {"type": "warmup"})

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
            return _require_candidates(data["candidates"])
        raise ValueError("FaaS response missing required 'candidates' field")


def _require_candidates(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("FaaS response candidates must be a list")
    return value
