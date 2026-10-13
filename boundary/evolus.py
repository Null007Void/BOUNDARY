"""Evolus integration foundation (B4.1) — offline-testable interface contract.

Evolus is an event workspace with a platform agent and two building blocks:
*document extraction* and *workflow orchestration*. This module defines the
strictly typed, *non-authoritative* input/output contracts for those two blocks,
plus a transport-injected adapter for offline testing.

**This module does not implement real Evolus endpoints.** The wire format, base
URL, auth scheme, and resource IDs are undocumented here; real integration awaits
the operator's platform API specification. What follows is a
truthful, tested interface contract — not a claim of completed integration.

Five evidence categories are kept deliberately distinct:
  - *extracted claims* (``ExtractionClaim``) — advisory platform output;
  - *trusted business facts* (``contracts.POLookupResult`` / ``SupplierLookupResult``);
  - *model interpretations* (``inference.EvidenceInterpretation``);
  - *human grants* (``grant.Grant``);
  - *executed actions* (``execution`` / ``authorized_actions``).

Extracted claims and workflow results never confer execution authority: the
``execution_authority`` field is locked to ``"NONE"``, and any platform output
carrying an authority key is rejected.
"""

from __future__ import annotations

import math
import os
import time
from enum import Enum
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Authority keys rejected from any platform output (top-level exact match).
_AUTHORITY_KEYS = {
    "approve", "approval", "authorize", "authorization", "authority",
    "grant", "decision", "execute", "execution", "commit", "action",
}

# Environment configuration (read at runtime; never hardcoded or printed).
ENV_BASE_URL = "BOUNDARY_EVOLUS_BASE_URL"
ENV_API_KEY = "BOUNDARY_EVOLUS_API_KEY"


class EvolusOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    CONTRADICTORY_RESPONSE = "CONTRADICTORY_RESPONSE"
    UNEXPECTED_STRUCTURE = "UNEXPECTED_STRUCTURE"
    AUTHORITY_CLAIM = "AUTHORITY_CLAIM"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"
    WORKFLOW_INACTIVE = "WORKFLOW_INACTIVE"
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    POLL_EXHAUSTED = "POLL_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    AUTH_FAILURE = "AUTH_FAILURE"


class EvolusError(Exception):
    """Base class for Evolus transport failures."""


class EvolusTimeout(EvolusError):
    """The request exceeded its configured timeout."""


class EvolusAuthError(EvolusError):
    """Authentication/authorization to the platform failed."""


class EvolusUnavailable(EvolusError):
    """The platform service is unavailable."""


class EvolusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Deliberately empty default: the real base URL is undocumented and must be
    # supplied by the operator. No endpoint is invented here.
    base_url: str = ""
    api_key: str = Field(default="", repr=False)  # never printed
    timeout_seconds: float = 30.0
    max_document_chars: int = 4096
    # Configurable workflow-status assumptions (explicit defaults, not invented
    # platform statuses; to be confirmed against the API spec).
    workflow_success_status: str = "SUCCEEDED"
    workflow_failure_status: str = "FAILED"

    @classmethod
    def from_env(cls) -> "EvolusConfig":
        return cls(
            base_url=os.environ.get(ENV_BASE_URL, ""),
            api_key=os.environ.get(ENV_API_KEY, ""),
        )


# --- Contracts: document extraction ------------------------------------------------


class ExtractionClaim(BaseModel):
    """A single claim extracted from a document by the platform.

    A claim is advisory evidence only. It is not a trusted business fact, a model
    interpretation, a human grant, or an executed action.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    value: str
    confidence: float

    @field_validator("field", mode="before")
    @classmethod
    def _non_blank_field(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("claim field must be a non-blank string")
        return v

    @field_validator("value", mode="before")
    @classmethod
    def _non_blank_value(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("claim value must be a non-blank string")
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
        if not math.isfinite(f) or not (0.0 <= f <= 1.0):
            raise ValueError("confidence must be a finite number in [0, 1]")
        return v


class DocumentExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_content: str
    fields: list[str] = Field(default_factory=list)


class DocumentExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    extractor_id: str  # provenance: which extractor produced these claims
    claims: list[ExtractionClaim] = Field(default_factory=list)
    execution_authority: Literal["NONE"] = "NONE"


# --- Contracts: workflow orchestration ----------------------------------------------


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: str
    notes: str = ""


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    case_id: str
    step_ids: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    run_id: str  # provenance: which run produced this result
    status: str
    steps: list[WorkflowStep] = Field(default_factory=list)
    execution_authority: Literal["NONE"] = "NONE"


# --- Telemetry and call results ------------------------------------------------------


class EvolusTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_count: int = 0
    elapsed_ms: Optional[int] = None
    outcome: EvolusOutcome
    failure_state: Optional[str] = None
    usage: Optional[dict] = None  # platform-reported usage, when available


class DocumentExtractionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: EvolusOutcome
    result: Optional[DocumentExtractionResult] = None
    telemetry: EvolusTelemetry
    reasons: list[str] = Field(default_factory=list)


class WorkflowCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: EvolusOutcome
    result: Optional[WorkflowResult] = None
    telemetry: EvolusTelemetry
    reasons: list[str] = Field(default_factory=list)


# A transport takes a request payload dict and returns the raw response dict, or
# raises an EvolusError subclass.
Transport = Callable[[dict], dict]


class EvolusAdapter:
    """Offline-testable adapter over the two Evolus building blocks.

    Transports are injected (``extract_transport`` and ``workflow_transport``) so
    tests can substitute mocks. Each call is a single bounded attempt, validated
    strictly, and recorded as non-authoritative.
    """

    def __init__(
        self,
        config: EvolusConfig,
        extract_transport: Optional[Transport] = None,
        workflow_transport: Optional[Transport] = None,
    ):
        self._config = config
        self._extract_transport = extract_transport
        self._workflow_transport = workflow_transport

    # -- document extraction -------------------------------------------------

    def extract_document(self, request: DocumentExtractionRequest) -> DocumentExtractionCall:
        start = time.monotonic()

        if not request.document_content.strip():
            return self._failure(
                DocumentExtractionCall, EvolusOutcome.MISSING_EVIDENCE, "missing evidence", start, "no document content"
            )
        if len(request.document_content) > self._config.max_document_chars:
            return self._failure(
                DocumentExtractionCall, EvolusOutcome.INPUT_TOO_LARGE, "input_too_large", start, "document exceeds size bound"
            )
        if self._extract_transport is None:
            return self._failure(
                DocumentExtractionCall, EvolusOutcome.UNAVAILABLE, "unavailable", start, "extraction transport not configured"
            )

        payload = {
            "document_id": request.document_id,
            "document_content": request.document_content,
            "fields": request.fields,
        }
        raw, transport_outcome, transport_reason = self._invoke(self._extract_transport, payload)
        if transport_outcome is not None:
            return self._failure(
                DocumentExtractionCall, transport_outcome, transport_reason, start, transport_reason, call_count=1
            )
        if raw is None:
            return self._failure(
                DocumentExtractionCall, EvolusOutcome.MALFORMED_RESPONSE, "malformed_response", start,
                "transport returned no response", call_count=1,
            )

        outcome, result, reasons, usage = self._parse_extraction(raw, request)
        return self._finish(DocumentExtractionCall, outcome, result, reasons, start, usage)

    # -- workflow orchestration ------------------------------------------------

    def run_workflow(self, request: WorkflowRequest) -> WorkflowCall:
        start = time.monotonic()

        if self._workflow_transport is None:
            return self._failure(
                WorkflowCall, EvolusOutcome.UNAVAILABLE, "unavailable", start, "workflow transport not configured"
            )

        payload = {
            "workflow_id": request.workflow_id,
            "case_id": request.case_id,
            "step_ids": request.step_ids,
        }
        raw, transport_outcome, transport_reason = self._invoke(self._workflow_transport, payload)
        if transport_outcome is not None:
            return self._failure(
                WorkflowCall, transport_outcome, transport_reason, start, transport_reason, call_count=1
            )
        if raw is None:
            return self._failure(
                WorkflowCall, EvolusOutcome.MALFORMED_RESPONSE, "malformed_response", start,
                "transport returned no response", call_count=1,
            )

        outcome, result, reasons, usage = self._parse_workflow(raw, request)
        return self._finish(WorkflowCall, outcome, result, reasons, start, usage)

    # -- shared helpers ---------------------------------------------------------

    def _invoke(self, transport, payload):
        try:
            return transport(payload), None, None
        except EvolusTimeout:
            return None, EvolusOutcome.TIMEOUT, "timeout"
        except EvolusAuthError:
            return None, EvolusOutcome.AUTH_FAILURE, "auth_failure"
        except EvolusUnavailable:
            return None, EvolusOutcome.UNAVAILABLE, "unavailable"
        except Exception:
            return None, EvolusOutcome.UNAVAILABLE, "transport_error"

    @staticmethod
    def _reject_authority(raw) -> Optional[list[str]]:
        if not isinstance(raw, dict):
            return None
        lower_keys = {k.lower() for k in raw}
        authority = lower_keys & _AUTHORITY_KEYS
        if authority:
            return [f"platform output attempted to assert authority: {sorted(authority)}"]
        # An explicit execution_authority field may only ever be "NONE".
        if "execution_authority" in raw and raw.get("execution_authority") != "NONE":
            return ["platform output attempted to confer execution authority"]
        return None

    def _parse_extraction(self, raw, request):
        authority = self._reject_authority(raw)
        if authority is not None:
            return EvolusOutcome.AUTHORITY_CLAIM, None, authority, None
        if not isinstance(raw, dict):
            return EvolusOutcome.MALFORMED_RESPONSE, None, ["response is not a JSON object"], None
        missing = [k for k in ("document_id", "extractor_id", "claims") if k not in raw]
        if missing:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, [f"missing field(s): {sorted(missing)}"], None
        if raw.get("document_id") != request.document_id:
            return (
                EvolusOutcome.CONTRADICTORY_RESPONSE,
                None,
                ["response document_id does not match request"],
                None,
            )
        usage, usage_err = self._extract_usage(raw)
        if usage_err is not None:
            return EvolusOutcome.MALFORMED_RESPONSE, None, [usage_err], None
        result_data = {k: v for k, v in raw.items() if k != "usage"}
        try:
            result = DocumentExtractionResult.model_validate(result_data)
        except ValidationError:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["field type(s) invalid"], None

        # Provenance: extractor identity must be meaningful.
        if not result.extractor_id.strip():
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["blank extractor identity"], usage
        # Completeness: claims must be present and cover the requested fields.
        if not result.claims:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["no claims extracted"], usage
        fields = [claim.field for claim in result.claims]
        if len(fields) != len(set(fields)):
            return EvolusOutcome.CONTRADICTORY_RESPONSE, None, ["duplicate claim fields"], usage
        missing_fields = [f for f in request.fields if f not in fields]
        if missing_fields:
            return (
                EvolusOutcome.INCOMPLETE_RESPONSE,
                None,
                [f"requested field(s) not extracted: {sorted(missing_fields)}"],
                usage,
            )
        return EvolusOutcome.SUCCESS, result, [], usage

    def _parse_workflow(self, raw, request):
        authority = self._reject_authority(raw)
        if authority is not None:
            return EvolusOutcome.AUTHORITY_CLAIM, None, authority, None
        if not isinstance(raw, dict):
            return EvolusOutcome.MALFORMED_RESPONSE, None, ["response is not a JSON object"], None
        missing = [k for k in ("workflow_id", "run_id", "status", "steps") if k not in raw]
        if missing:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, [f"missing field(s): {sorted(missing)}"], None
        if raw.get("workflow_id") != request.workflow_id:
            return (
                EvolusOutcome.CONTRADICTORY_RESPONSE,
                None,
                ["response workflow_id does not match request"],
                None,
            )
        usage, usage_err = self._extract_usage(raw)
        if usage_err is not None:
            return EvolusOutcome.MALFORMED_RESPONSE, None, [usage_err], None
        result_data = {k: v for k, v in raw.items() if k != "usage"}
        try:
            result = WorkflowResult.model_validate(result_data)
        except ValidationError:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["field type(s) invalid"], None

        # Provenance: run identity must be meaningful.
        if not result.run_id.strip():
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["blank run identity"], usage

        # Truthful workflow outcome: a parsed response is not proof of successful
        # completion. Distinguish failure, pending/incomplete, and completed work.
        status = result.status
        if status == self._config.workflow_failure_status:
            return EvolusOutcome.WORKFLOW_FAILED, result, ["workflow status indicates failure"], usage
        if status != self._config.workflow_success_status:
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["workflow status is pending or unrecognized"], usage

        # Step identifiers must be meaningful.
        if any(not step.step_id.strip() for step in result.steps):
            return EvolusOutcome.INCOMPLETE_RESPONSE, None, ["blank step identifier"], usage
        step_ids = [step.step_id for step in result.steps]
        if len(step_ids) != len(set(step_ids)):
            return EvolusOutcome.CONTRADICTORY_RESPONSE, None, ["duplicate workflow step identifiers"], usage
        missing_steps = [s for s in request.step_ids if s not in step_ids]
        if missing_steps:
            return (
                EvolusOutcome.INCOMPLETE_RESPONSE,
                None,
                [f"requested step(s) missing: {sorted(missing_steps)}"],
                usage,
            )

        # Every requested step must report a recognized successful status.
        step_status_by_id = {step.step_id: step.status for step in result.steps}
        failed = [s for s in request.step_ids if step_status_by_id.get(s) == self._config.workflow_failure_status]
        if failed:
            return (
                EvolusOutcome.WORKFLOW_FAILED,
                result,
                [f"required step(s) failed: {sorted(failed)}"],
                usage,
            )
        not_complete = [
            s for s in request.step_ids if step_status_by_id.get(s) != self._config.workflow_success_status
        ]
        if not_complete:
            return (
                EvolusOutcome.INCOMPLETE_RESPONSE,
                None,
                [f"required step(s) not successfully completed: {sorted(not_complete)}"],
                usage,
            )
        return EvolusOutcome.SUCCESS, result, [], usage

    def _telemetry(self, outcome, start, *, failure_state=None, usage=None, call_count=0):
        return EvolusTelemetry(
            call_count=call_count,
            elapsed_ms=int((time.monotonic() - start) * 1000),
            outcome=outcome,
            failure_state=failure_state,
            usage=usage,
        )

    def _failure(self, call_cls, outcome, failure_state, start, reason, call_count=0):
        telemetry = self._telemetry(outcome, start, failure_state=failure_state, call_count=call_count)
        return call_cls(outcome=outcome, telemetry=telemetry, reasons=[reason])

    def _finish(self, call_cls, outcome, result, reasons, start, usage):
        telemetry = self._telemetry(outcome, start, usage=usage, call_count=1)
        return call_cls(outcome=outcome, result=result, telemetry=telemetry, reasons=reasons)

    @staticmethod
    def _extract_usage(raw):
        """Return (usage, error). usage is None when unreported; error is set when
        usage is present but not an object."""
        if not isinstance(raw, dict) or "usage" not in raw:
            return None, None
        usage = raw.get("usage")
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            return usage, None
        return None, "usage metadata is not an object"


class EvolusIntegrationRequirements(BaseModel):
    """The exact platform API information the operator must supply for real integration.

    Every field is deliberately empty until the operator provides it; no undocumented
    endpoint, auth scheme, or resource ID is invented here.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = ""
    auth_scheme: str = ""
    document_extraction_endpoint: str = ""
    workflow_orchestration_endpoint: str = ""
    document_extraction_request_wire_format: str = ""
    workflow_request_wire_format: str = ""
    usage_field_location: str = ""
    resource_ids: list[str] = Field(default_factory=list)
    api_key_env_var: str = ENV_API_KEY
