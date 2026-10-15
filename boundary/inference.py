"""Offline-bounded inference adapter for the AMD Token Factory endpoint (B3.1).

A small, reusable adapter that wraps an OpenAI-compatible model endpoint to
interpret *ambiguous* synthetic invoice/PO evidence. It is preparation for live
AMD inference; B3.1 uses local mock transports only and never calls the live API.

Model output is strictly validated and treated as non-authoritative advisory
evidence: it can never issue approval, bypass deterministic controls, create a
human grant, or write to the ledger. Unparseable, incomplete, contradictory, or
authority-claiming responses fail closed.

The transport (HTTP layer) is injected so the adapter is swappable and testable
without network access. Secrets are read from environment configuration and are
never printed.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Default serving identifier for the verified AMD Token Factory model family.
DEFAULT_MODEL_ID = "DeepSeek-V4-Flash"

# The only model for which code-fence normalization may be enabled (catalogue-verified Gemma name).
GEMMA_MODEL_ID = "Gemma-4-31B-IT"
# Default AMD Token Factory base URL (configurable via BOUNDARY_AMD_BASE_URL).
DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"

# Environment variable names for base URL, model id, and API credential.
ENV_BASE_URL = "BOUNDARY_AMD_BASE_URL"
ENV_MODEL_ID = "BOUNDARY_AMD_MODEL_ID"
ENV_API_KEY = "BOUNDARY_AMD_API_KEY"

# Response contract: required keys, allowed keys, and forbidden authority keys.
_REQUIRED_KEYS = {"supplier_id", "invoice_id", "amount_cents", "discrepancy_cents", "confidence"}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"po_reference", "notes"}
_AUTHORITY_KEYS = {
    "approve", "approval", "authorize", "authorization", "authority",
    "grant", "decision", "execute", "execution", "commit", "action",
}


class InferenceError(Exception):
    """Base class for inference/transport failures.

    Carries optional structured diagnostics (a failure category and, when received, the HTTP
    status code) without any raw response body, credential, header, or URL.
    """

    def __init__(self, message: str = "", *, diagnostics: Optional[dict] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class InferenceTimeout(InferenceError):
    """The request exceeded its configured timeout."""


class InferenceAuthError(InferenceError):
    """Authentication/authorization to the model endpoint failed."""


class InferenceUnavailable(InferenceError):
    """The model service is unavailable (a received HTTP error)."""


class InferenceConnectionError(InferenceUnavailable):
    """The connection was lost without any HTTP response (remote uncertainty)."""


class BudgetExceeded(InferenceError):
    """A hard request/token budget was exhausted; the run must stop (not a soft failure)."""


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    api_key: str = Field(default="", repr=False)  # never printed
    timeout_seconds: float = 30.0  # configuration only; enforced by B3.2 HTTP transport
    max_input_chars: int = 4096
    max_output_chars: int = 4096
    code_fence_normalization: bool = False  # opt-in Gemma code-fence recovery; default OFF

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        return cls(
            base_url=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
            model_id=os.environ.get(ENV_MODEL_ID, DEFAULT_MODEL_ID),
            api_key=os.environ.get(ENV_API_KEY, ""),
        )


class EvidenceInterpretation(BaseModel):
    """The constrained advisory interpretation contract returned by the model."""

    model_config = ConfigDict(extra="forbid")

    supplier_id: str
    invoice_id: str
    po_reference: Optional[str] = None
    amount_cents: int
    discrepancy_cents: int
    confidence: float
    notes: str = ""

    @field_validator("amount_cents", "discrepancy_cents", mode="before")
    @classmethod
    def _integer_cents(cls, v):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("monetary cents must be a real integer, not a boolean or numeric string")
        if v < -(2 ** 63) or v > (2 ** 63 - 1):
            raise ValueError("monetary cents outside signed 64-bit range")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _numeric_confidence(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence must be a finite number, not a string or boolean")
        try:
            f = float(v)
        except (OverflowError, ValueError):
            raise ValueError("confidence is not a finite representable number")
        if not math.isfinite(f):
            raise ValueError("confidence must be finite")
        return v


class InferenceOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    CONTRADICTORY_RESPONSE = "CONTRADICTORY_RESPONSE"
    UNEXPECTED_STRUCTURE = "UNEXPECTED_STRUCTURE"
    AUTHORITY_CLAIM = "AUTHORITY_CLAIM"
    TELEMETRY_INVALID = "TELEMETRY_INVALID"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    AUTH_FAILURE = "AUTH_FAILURE"


class InferenceTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    elapsed_ms: int = 0
    outcome: InferenceOutcome
    error_class: Optional[str] = None
    diagnostics: Optional[dict] = None  # structured failure classification (category + http_status)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: InferenceOutcome
    interpretation: Optional[EvidenceInterpretation] = None
    telemetry: InferenceTelemetry
    reasons: list[str] = Field(default_factory=list)
    normalized: bool = False  # True iff a code-fenced response was recovered (not natively valid)
    endpoint_model_id: Optional[str] = None  # the model id the response actually reported (or None)
    receipt: Optional[str] = None  # "received" (got an HTTP response) | "unconfirmed" (no response) | None


# Exactly one ```json code fence wrapping exactly one complete JSON object.
_CODE_FENCE_RE = re.compile(r"^```json[ \t]*\r?\n(.*?)\r?\n```\s*\Z", re.DOTALL)


def normalize_code_fenced_json(content: Optional[str]) -> Optional[str]:
    """Extract the inner JSON from a SINGLE permitted ```json fence, strictly.

    Accepts ONLY one complete JSON object enclosed by one ```json fence. Rejects prose, multiple
    objects, nested fences, partial JSON, arrays, and empty bodies. Returns the inner JSON string,
    or None. Used only for opt-in Gemma recovery; the default adapter path does not normalize.
    """
    if not isinstance(content, str) or not content:
        return None
    m = _CODE_FENCE_RE.match(content)
    if m is None:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    if "```" in inner:
        return None
    if not (inner.startswith("{") and inner.endswith("}")):
        return None
    try:
        data = json.loads(inner)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return inner


# A transport takes the OpenAI-compatible request payload and returns the raw
# response dict, or raises an InferenceError subclass.
Transport = Callable[[dict], dict]


class InferenceAdapter:
    """Bounded, validating adapter over an OpenAI-compatible model endpoint."""

    def __init__(self, config: InferenceConfig, transport: Transport):
        self._config = config
        self._transport = transport

    def interpret(self, evidence: str) -> InferenceResult:
        """Interpret ambiguous evidence; returns a validated advisory result.

        A single bounded attempt with no retries. Every response is validated
        against the constrained contract and treated as non-authoritative.
        """
        start = time.monotonic()
        if len(evidence) > self._config.max_input_chars:
            return self._transport_result(InferenceOutcome.INPUT_TOO_LARGE, "input_too_large", start)
        payload = self._build_payload(evidence)

        try:
            raw = self._transport(payload)
        except InferenceTimeout as exc:
            return self._transport_result(InferenceOutcome.TIMEOUT, "timeout", start,
                                          receipt="unconfirmed", diagnostics=exc.diagnostics)
        except InferenceConnectionError as exc:
            return self._transport_result(InferenceOutcome.UNAVAILABLE, "connection_loss", start,
                                          receipt="unconfirmed", diagnostics=exc.diagnostics)
        except InferenceAuthError as exc:
            return self._transport_result(InferenceOutcome.AUTH_FAILURE, "auth_failure", start,
                                          receipt="received", diagnostics=exc.diagnostics)
        except InferenceUnavailable as exc:
            return self._transport_result(InferenceOutcome.UNAVAILABLE, "unavailable", start,
                                          receipt="received", diagnostics=exc.diagnostics)
        except BudgetExceeded:
            raise  # hard budget stop must propagate to the caller, not become a soft UNAVAILABLE
        except Exception:
            return self._transport_result(InferenceOutcome.UNAVAILABLE, "transport_error", start,
                                          receipt="unconfirmed")

        # Capture the model identifier the response actually reported (or its absence), rather
        # than constructing it from configuration.
        endpoint_model_id = raw.get("model") if isinstance(raw, dict) else None

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Validate usage metadata; malformed usage fails closed, never raises.
        prompt, completion, total, usage_err = self._extract_usage(raw)
        if usage_err is not None:
            telemetry = InferenceTelemetry(
                model_id=self._config.model_id,
                elapsed_ms=elapsed_ms,
                outcome=InferenceOutcome.TELEMETRY_INVALID,
                error_class=usage_err,
            )
            return InferenceResult(outcome=InferenceOutcome.TELEMETRY_INVALID,
                                   telemetry=telemetry, reasons=[usage_err],
                                   endpoint_model_id=endpoint_model_id)

        telemetry = InferenceTelemetry(
            model_id=self._config.model_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            elapsed_ms=elapsed_ms,
            outcome=InferenceOutcome.SUCCESS,
        )

        content = self._extract_content(raw)
        if content is None:
            telemetry.outcome = InferenceOutcome.UNEXPECTED_STRUCTURE
            return InferenceResult(outcome=telemetry.outcome, telemetry=telemetry,
                                   reasons=["response structure is unexpected"],
                                   endpoint_model_id=endpoint_model_id)
        if len(content) > self._config.max_output_chars:
            telemetry.outcome = InferenceOutcome.UNEXPECTED_STRUCTURE
            return InferenceResult(outcome=telemetry.outcome, telemetry=telemetry,
                                   reasons=["response exceeds the output size bound"],
                                   endpoint_model_id=endpoint_model_id)

        # Optional code-fence normalization (opt-in, default OFF, and Gemma-only). Even when the
        # flag is set, normalization is refused for any model other than GEMMA_MODEL_ID, so
        # DeepSeek and other adapters can never be normalized merely by flipping a config flag.
        normalized = False
        if self._config.code_fence_normalization and self._config.model_id == GEMMA_MODEL_ID:
            inner = normalize_code_fenced_json(content)
            if inner is not None:
                content = inner
                normalized = True

        outcome, interpretation, reasons = self._parse_and_validate(content)
        telemetry.outcome = outcome
        return InferenceResult(outcome=outcome, interpretation=interpretation,
                               telemetry=telemetry, reasons=reasons, normalized=normalized,
                               endpoint_model_id=endpoint_model_id, receipt="received")

    def _build_payload(self, evidence: str) -> dict:
        system = (
            "You interpret ambiguous invoice/PO evidence. Return ONLY a JSON object with "
            "keys: supplier_id (string), invoice_id (string), po_reference (string or null), "
            "amount_cents (integer), discrepancy_cents (integer), confidence (number 0 to 1), "
            "notes (string). Your output is advisory only and carries no authority."
        )
        return {
            "model": self._config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": evidence},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

    def _extract_content(self, raw) -> Optional[str]:
        if not isinstance(raw, dict):
            return None
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str):
            return None
        return content

    def _parse_and_validate(self, content: str):
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return InferenceOutcome.MALFORMED_RESPONSE, None, ["response is not valid JSON"]
        if not isinstance(data, dict):
            return InferenceOutcome.MALFORMED_RESPONSE, None, ["response JSON is not an object"]

        lower_keys = {k.lower() for k in data}
        authority = lower_keys & _AUTHORITY_KEYS
        if authority:
            return (InferenceOutcome.AUTHORITY_CLAIM, None,
                    [f"model attempted to assert authority: {sorted(authority)}"])

        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            return (InferenceOutcome.INCOMPLETE_RESPONSE, None,
                    [f"missing required field(s): {sorted(missing)}"])

        extra = [k for k in data if k not in _ALLOWED_KEYS]
        if extra:
            return (InferenceOutcome.UNEXPECTED_STRUCTURE, None,
                    [f"unexpected field(s): {sorted(extra)}"])

        try:
            interpretation = EvidenceInterpretation.model_validate(data)
        except ValidationError:
            return InferenceOutcome.INCOMPLETE_RESPONSE, None, ["field type(s) invalid"]

        if interpretation.amount_cents < 0:
            return InferenceOutcome.CONTRADICTORY_RESPONSE, None, ["amount is negative"]
        if not (0.0 <= interpretation.confidence <= 1.0):
            return InferenceOutcome.CONTRADICTORY_RESPONSE, None, ["confidence out of range"]
        if not interpretation.supplier_id.strip() or not interpretation.invoice_id.strip():
            return InferenceOutcome.INCOMPLETE_RESPONSE, None, ["blank identifier"]

        return InferenceOutcome.SUCCESS, interpretation, []

    def _extract_usage(self, raw):
        """Return (prompt, completion, total, error) token counts.

        Counts are None when not reported, nonnegative int when valid. Returns a
        non-None error message when usage metadata is malformed (wrong type,
        boolean, negative, or non-integer). Never includes raw response content.
        """
        if not isinstance(raw, dict):
            return None, None, None, None
        usage = raw.get("usage")
        if usage is None:
            return None, None, None, None
        if not isinstance(usage, dict):
            return None, None, None, "usage metadata is not an object"
        counts = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is None:
                counts[key] = None
            elif isinstance(value, bool) or not isinstance(value, int):
                return None, None, None, f"usage {key} is not a nonnegative integer"
            elif value < 0:
                return None, None, None, f"usage {key} is negative"
            else:
                counts[key] = value
        return counts["prompt_tokens"], counts["completion_tokens"], counts["total_tokens"], None

    def _transport_result(self, outcome: InferenceOutcome, error_class: str, start: float,
                          receipt: Optional[str] = None, diagnostics: Optional[dict] = None) -> InferenceResult:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        telemetry = InferenceTelemetry(
            model_id=self._config.model_id,
            elapsed_ms=elapsed_ms,
            outcome=outcome,
            error_class=error_class,
            diagnostics=diagnostics,
        )
        return InferenceResult(outcome=outcome, telemetry=telemetry, reasons=[error_class],
                               receipt=receipt)
