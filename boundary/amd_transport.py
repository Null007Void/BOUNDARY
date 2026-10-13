"""httpx transport for AMD Token Factory (B3.2).

Builds a transport callable compatible with :class:`InferenceAdapter`'s injected
transport interface. It POSTs a single non-streaming OpenAI-compatible
chat-completion request to ``{base_url}/chat/completions`` with Bearer
authentication, enforcing the configured network timeout and a bounded response
body. It performs exactly one request and never retries.

The transport raises :class:`InferenceAuthError` (401/403),
:class:`InferenceTimeout` (timeout), or :class:`InferenceUnavailable` (429,
server errors, network failures, malformed bodies). Error messages never carry
the API credential.
"""

from __future__ import annotations

import json

import httpx

from .inference import InferenceAuthError, InferenceTimeout, InferenceUnavailable


def build_amd_transport(config, client: httpx.Client | None = None):
    """Return a transport callable for the given :class:`InferenceConfig`.

    ``client`` is an optional :class:`httpx.Client`; when omitted, a client with
    the configured timeout is created. Tests inject a client backed by
    ``httpx.MockTransport``.
    """
    timeout = httpx.Timeout(config.timeout_seconds)
    if client is None:
        client = httpx.Client(timeout=timeout)
    # Allow generous JSON overhead on top of the adapter's content bound so a
    # runaway response cannot be read into memory unbounded.
    max_body_bytes = config.max_output_chars * 4

    def transport(payload: dict) -> dict:
        url = config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code in (401, 403):
                    raise InferenceAuthError("authentication failed")
                if resp.status_code == 429:
                    raise InferenceUnavailable("rate limited")
                if resp.status_code >= 500:
                    raise InferenceUnavailable("server error")
                if resp.status_code != 200:
                    raise InferenceUnavailable("unexpected status")

                # Count decoded response bytes incrementally and abort as soon
                # as the bound is exceeded, rather than buffering the whole body.
                chunks = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_body_bytes:
                        raise InferenceUnavailable("response body too large")
                    chunks.append(chunk)
                body = b"".join(chunks)
        except httpx.TimeoutException:
            raise InferenceTimeout("request timed out")
        except httpx.RequestError:
            raise InferenceUnavailable("network or transport failure")

        try:
            return json.loads(body)
        except ValueError:
            raise InferenceUnavailable("malformed response body")

    return transport
