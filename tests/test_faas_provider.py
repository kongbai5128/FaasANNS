"""FaaS provider response validation tests."""

from __future__ import annotations

import pytest

from faas.aliyun_fc_provider import AliyunHTTPProvider


def test_aliyun_provider_rejects_response_without_candidates(monkeypatch) -> None:
    provider = AliyunHTTPProvider({"default": "http://example.test"}, timeout_seconds=10.0)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self) -> bytes:
            return b'{"error": "bad response"}'

    monkeypatch.setattr(provider.opener, "open", lambda request, timeout: Response())

    with pytest.raises(ValueError, match="missing required"):
        provider._post_candidates({"request_id": "x"})


def test_aliyun_provider_rejects_non_list_candidates(monkeypatch) -> None:
    provider = AliyunHTTPProvider({"default": "http://example.test"}, timeout_seconds=10.0)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self) -> bytes:
            return b'{"candidates": null}'

    monkeypatch.setattr(provider.opener, "open", lambda request, timeout: Response())

    with pytest.raises(ValueError, match="candidates must be a list"):
        provider._post_candidates({"request_id": "x"})
