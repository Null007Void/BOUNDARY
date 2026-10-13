"""B3.2 AMD HTTP transport and live smoke-test coverage.

Uses httpx.MockTransport so no network calls occur during testing.
"""

import json

import httpx
import pytest
from httpx._types import SyncByteStream

from boundary import smoke_test
from boundary.amd_transport import build_amd_transport
from boundary.inference import (
    InferenceAdapter,
    InferenceConfig,
    InferenceOutcome,
    InferenceUnavailable,
)
from boundary.smoke_test import format_result, main

TEST_BASE_URL = "https://example.com/v1"
TEST_API_KEY = "sk-test-secret-value"


def _valid_content():
    return json.dumps(
        {
            "supplier_id": "SUP-9001",
            "invoice_id": "INV-2048",
            "po_reference": "PO-77",
            "amount_cents": 125000,
            "discrepancy_cents": 500,
            "confidence": 0.9,
            "notes": "",
        }
    )


def _valid_raw():
    return {
        "choices": [{"message": {"content": _valid_content()}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _adapter_with_handler(handler, config=None):
    config = config or InferenceConfig(base_url=TEST_BASE_URL, api_key=TEST_API_KEY)
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)
    transport = build_amd_transport(config, client=client)
    return config, InferenceAdapter(config, transport)


def test_request_format():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_valid_raw())

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("some evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert captured["method"] == "POST"
    assert captured["url"] == TEST_BASE_URL + "/chat/completions"
    assert captured["auth"] == f"Bearer {TEST_API_KEY}"
    assert captured["body"]["model"] == "DeepSeek-V4-Flash"
    assert "messages" in captured["body"]


def test_success_with_telemetry():
    def handler(request):
        return httpx.Response(200, json=_valid_raw())

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert result.telemetry.total_tokens == 30
    assert result.interpretation.amount_cents == 125000


def test_auth_failure_401():
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.AUTH_FAILURE


def test_auth_failure_403():
    def handler(request):
        return httpx.Response(403, json={"error": "forbidden"})

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.AUTH_FAILURE


def test_rate_limited_429():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE


def test_server_error_500():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE


def test_network_failure():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE


def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TIMEOUT


def test_malformed_body():
    def handler(request):
        return httpx.Response(200, content=b"this is not json")

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE


def test_oversized_body_rejected():
    def handler(request):
        return httpx.Response(200, content=b"x" * 20000)

    _, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE


def test_credential_not_leaked_on_error():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})

    config, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.AUTH_FAILURE
    assert config.api_key not in result.telemetry.error_class
    assert config.api_key not in " ".join(result.reasons)
    assert config.api_key not in repr(result)


def test_smoke_test_no_live_no_network(capsys):
    rc = main([])
    assert rc == 0
    assert "no live request" in capsys.readouterr().out.lower()


def test_smoke_test_live_without_key(capsys, monkeypatch):
    monkeypatch.delenv("BOUNDARY_AMD_API_KEY", raising=False)
    rc = main(["--live"])
    assert rc == 1
    assert "BOUNDARY_AMD_API_KEY" in capsys.readouterr().out


def test_format_result_never_includes_credential():
    def handler(request):
        return httpx.Response(200, json=_valid_raw())

    config, adapter = _adapter_with_handler(handler)
    result = adapter.interpret("evidence")
    text = format_result(result)
    assert config.api_key not in text
    assert "amount_cents=125000" in text


class _CountingStream(SyncByteStream):
    """A sync byte stream that records how many bytes are actually consumed."""

    def __init__(self, chunks, counter):
        self._chunks = chunks
        self._counter = counter

    def __iter__(self):
        for chunk in self._chunks:
            self._counter.append(len(chunk))
            yield chunk


def test_streaming_stops_at_limit():
    counter = []

    def handler(request):
        chunks = [b"x"] * 1000  # 1000 bytes total
        return httpx.Response(200, stream=_CountingStream(chunks, counter))

    # max_body_bytes = max_output_chars * 4 = 64
    config = InferenceConfig(base_url=TEST_BASE_URL, api_key=TEST_API_KEY, max_output_chars=16)
    _, adapter = _adapter_with_handler(handler, config=config)
    result = adapter.interpret("evidence")

    assert result.outcome == InferenceOutcome.UNAVAILABLE
    # Transport aborts at the bound; it must not consume the entire oversized body.
    assert len(counter) < 1000
    assert sum(counter) < 100


def test_smoke_test_exit_status_success(monkeypatch):
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", "test-key")

    def fake_transport(config):
        def transport(payload):
            return _valid_raw()

        return transport

    monkeypatch.setattr(smoke_test, "build_amd_transport", fake_transport)
    assert main(["--live"]) == 0


def test_smoke_test_exit_status_reflects_failure(monkeypatch):
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", "test-key")

    def fake_transport(config):
        def transport(payload):
            raise InferenceUnavailable("service down")

        return transport

    monkeypatch.setattr(smoke_test, "build_amd_transport", fake_transport)
    assert main(["--live"]) == 1
