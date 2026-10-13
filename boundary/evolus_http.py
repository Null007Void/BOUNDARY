"""Real Evolus Integration API HTTP client (B4.3).

Connects the accepted B4.1 adapter architecture to the documented Evolus
Integration API. Offline-testable via an injected ``httpx.Client`` (or its
transport).

Documented endpoints
---------------------
- ``POST /api/v1/workflows/{workflowId}/execute``
  Accepts optional text (field ``input``) and/or multipart file attachments.
  Returns HTTP 202 with ``executionId``, ``jobId``, ``enqueuedAt``, ``traceId``.
- ``GET /api/v1/workflows/{workflowId}/executions/{executionId}``
  Returns execution ``status``, ``output``, ``error``/``errors``, and steps.

Authentication is via the ``X-ApiKey`` and ``X-Application-Id`` headers.
Credentials are read from the environment only and never printed or committed.

Final-output contract (strict)
------------------------------
The terminal execution ``output`` is a JSON-encoded string that must decode to a
top-level object containing, at the top level:

- ``authority`` exactly ``"NONE"``;
- ``evidenceStatus`` exactly ``"unverified"``;
- ``extraction`` — a nonempty structured object (whose JSON-encoded
  ``lineItems`` / ``supportingReferences`` / ``missingFields`` /
  ``extractionWarnings`` arrays are parsed and validated);
- ``advisoryAnalysis`` — a nonempty advisory response (possibly JSON-encoded).

Authority fields found only in unrelated step output, execution metadata, or
nested advisory text do not satisfy this contract. ``SUCCESS`` is never returned
with an empty/``None`` output.

Unresolved contract uncertainties (handled defensively, never invented)
-----------------------------------------------------------------------
- The exact terminal-status vocabulary (compared case-insensitively against
  configurable assumptions; ``Completed`` is accepted).
- The multipart attachment field is verified as ``file`` for this workflow and
  remains configurable for other workflows.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .evolus import ENV_API_KEY, EvolusOutcome

# Environment variable names (credentials/identifiers; never hardcoded secrets).
ENV_BASE_URL = "BOUNDARY_EVOLUS_BASE_URL"
ENV_APPLICATION_ID = "BOUNDARY_EVOLUS_APPLICATION_ID"
ENV_WORKFLOW_ID = "BOUNDARY_EVOLUS_WORKFLOW_ID"

# Documented, non-secret platform resources (configurable, not credentials).
DEFAULT_BASE_URL = "https://api.evolus.ai"
# Application/workflow identifiers are account-specific. They are NOT committed with
# real values: they must be supplied via environment variables (BOUNDARY_EVOLUS_*).
DEFAULT_APPLICATION_ID = ""
DEFAULT_WORKFLOW_ID = ""

# Required top-level advisory-field constraints (exact match).
REQUIRED_AUTHORITY = "NONE"
REQUIRED_EVIDENCE_STATUS = "unverified"

# Authoritative extraction schema: exactly these 12 fields.
EXTRACTION_SCALAR_FIELDS = (
    "documentType", "supplierId", "invoiceId", "purchaseOrderId",
    "purchaseOrderReference", "issueDate", "currency", "statedTotal",
)
EXTRACTION_ARRAY_FIELDS = ("lineItems", "supportingReferences", "missingFields", "extractionWarnings")
EXTRACTION_FIELDS = EXTRACTION_SCALAR_FIELDS + EXTRACTION_ARRAY_FIELDS

# Authoritative document-type choices (uppercase).
DOCUMENT_TYPE_CHOICES = ("INVOICE", "PURCHASE_ORDER", "UNKNOWN")

# Bound recursion when decoding JSON-encoded strings/arrays in platform output.
_MAX_JSON_DEPTH = 8


class EvolusHttpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_BASE_URL
    application_id: str = DEFAULT_APPLICATION_ID
    workflow_id: str = DEFAULT_WORKFLOW_ID
    api_key: str = Field(default="", repr=False)  # never printed
    timeout_seconds: float = 30.0

    # Bounded polling for the asynchronous execution result.
    poll_interval_seconds: float = 2.0
    max_polls: int = 30

    # Configurable status-vocabulary assumptions (compared case-insensitively).
    terminal_success_statuses: list[str] = Field(
        default_factory=lambda: ["completed", "succeeded", "success"]
    )
    terminal_failure_statuses: list[str] = Field(
        default_factory=lambda: ["failed", "cancelled", "expired", "error"]
    )

    # Documented text field name for submission.
    submit_text_field: str = "input"
    # Verified document attachment field name for this workflow (configurable for
    # other workflows).
    submit_attachment_field: str = "file"

    # Document-type choices (authoritative, uppercase; configurable).
    document_type_choices: list[str] = Field(default_factory=lambda: list(DOCUMENT_TYPE_CHOICES))

    @classmethod
    def from_env(cls) -> "EvolusHttpConfig":
        return cls(
            base_url=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
            application_id=os.environ.get(ENV_APPLICATION_ID, DEFAULT_APPLICATION_ID),
            workflow_id=os.environ.get(ENV_WORKFLOW_ID, DEFAULT_WORKFLOW_ID),
            api_key=os.environ.get(ENV_API_KEY, ""),
        )


# --- Contracts ----------------------------------------------------------------------


class WorkflowSubmission(BaseModel):
    """The HTTP 202 acknowledgment of an enqueued workflow execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    execution_id: str = Field(alias="executionId")
    job_id: str = Field(alias="jobId")
    enqueued_at: Optional[str] = Field(default=None, alias="enqueuedAt")
    trace_id: Optional[str] = Field(default=None, alias="traceId")


class SubmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: EvolusOutcome
    submission: Optional[WorkflowSubmission] = None
    reasons: list[str] = Field(default_factory=list)
    call_count: int = 0
    elapsed_ms: int = 0


class EvolusExecutionResult(BaseModel):
    """The reconciled execution result, with original evidence preserved separately."""

    model_config = ConfigDict(extra="forbid")

    outcome: EvolusOutcome
    execution_id: Optional[str] = None
    status: Optional[str] = None
    output: Optional[dict] = None  # deep-decoded advisory output (never None on SUCCESS)
    extraction: Optional[dict] = None  # extracted evidence (12-field contract)
    advisory_analysis: Optional[object] = None  # advisory content (text or structured)
    errors: list[str] = Field(default_factory=list)
    authority: Optional[str] = None  # "NONE" when validated
    evidence_status: Optional[str] = None  # "unverified" when validated
    original_evidence: str = ""  # verbatim submitted text; empty for file-only
    original_file_name: Optional[str] = None  # attachment filename (byte-exact provenance)
    original_file_sha256: Optional[str] = None  # SHA-256 of attachment bytes
    reasons: list[str] = Field(default_factory=list)
    call_count: int = 0  # POST + GET polls
    elapsed_ms: int = 0


# --- JSON-encoded output decoding ----------------------------------------------------


def _deep_decode(value, depth: int = 0):
    """Recursively decode JSON-encoded strings and arrays within a value.

    Platform output may embed JSON-encoded arrays and JSON-encoded advisory text
    as strings; this unwraps them so fields can be inspected without mutating the
    original values.
    """
    if depth > _MAX_JSON_DEPTH:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "[") and stripped[-1:] in ("}", "]"):
            try:
                return _deep_decode(json.loads(stripped), depth + 1)
            except (ValueError, TypeError):
                return value
        return value
    if isinstance(value, dict):
        return {k: _deep_decode(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_decode(v, depth + 1) for v in value]
    return value


def _validate_extraction(extraction, config) -> tuple[bool, Optional[str]]:
    """Validate the decoded extraction object against the extractor contract."""
    if not isinstance(extraction, dict) or not extraction:
        return False, "extraction is missing or not a nonempty object"

    # All 12 fields must be present; missing keys are not silently accepted.
    missing = [f for f in EXTRACTION_FIELDS if f not in extraction]
    if missing:
        return False, f"extraction missing required field(s): {sorted(missing)}"

    # Scalar fields: string or null (reject booleans, numbers, objects, lists).
    for field in EXTRACTION_SCALAR_FIELDS:
        value = extraction[field]
        if value is not None and not isinstance(value, str):
            return False, f"extraction field {field} is not a string or null"

    # Document type: against configured (uppercase) choices (null permitted).
    document_type = extraction["documentType"]
    if document_type is not None and document_type not in config.document_type_choices:
        return False, f"document type {document_type!r} is not a configured choice"

    # The four array fields must decode to arrays.
    for field in EXTRACTION_ARRAY_FIELDS:
        decoded = _deep_decode(extraction[field])
        if not isinstance(decoded, list):
            return False, f"extraction field {field} is not a valid array"
    return True, None


def _validate_final_output(output, config) -> tuple[EvolusOutcome, Optional[str]]:
    """Validate the decoded final output against the strict contract.

    Returns ``(outcome, reason)`` where ``outcome`` is ``SUCCESS`` on a valid
    output, ``AUTHORITY_CLAIM`` when the top-level authority is an explicit
    non-``NONE`` value, and ``INCOMPLETE_RESPONSE`` for any other defect.
    """
    if not isinstance(output, dict):
        return EvolusOutcome.INCOMPLETE_RESPONSE, "final output is not a JSON object"

    if "authority" not in output:
        return EvolusOutcome.INCOMPLETE_RESPONSE, "authority missing"
    if output["authority"] != REQUIRED_AUTHORITY:
        return EvolusOutcome.AUTHORITY_CLAIM, "authority is not exactly NONE"
    if output.get("evidenceStatus") != REQUIRED_EVIDENCE_STATUS:
        return EvolusOutcome.INCOMPLETE_RESPONSE, "evidenceStatus is missing or not exactly unverified"

    extraction = _deep_decode(output.get("extraction"))
    ok, reason = _validate_extraction(extraction, config)
    if not ok:
        return EvolusOutcome.INCOMPLETE_RESPONSE, reason

    advisory = _deep_decode(output.get("advisoryAnalysis"))
    if advisory is None or isinstance(advisory, bool) or not isinstance(advisory, (str, dict, list)):
        return EvolusOutcome.INCOMPLETE_RESPONSE, "advisoryAnalysis is not meaningful text or structured content"
    if isinstance(advisory, str):
        if not advisory.strip():
            return EvolusOutcome.INCOMPLETE_RESPONSE, "advisoryAnalysis is whitespace-only"
        if advisory.strip()[:1] in ("{", "["):
            return EvolusOutcome.INCOMPLETE_RESPONSE, "advisoryAnalysis is malformed JSON-like text"
    if advisory == "" or advisory == {} or advisory == []:
        return EvolusOutcome.INCOMPLETE_RESPONSE, "advisoryAnalysis is empty"
    return EvolusOutcome.SUCCESS, None


# --- HTTP client --------------------------------------------------------------------


class EvolusHttpClient:
    """Submits a document and reconciles the asynchronous workflow execution."""

    def __init__(self, config: EvolusHttpConfig, client: Optional[httpx.Client] = None):
        self._config = config
        self._client = client  # inject for tests (httpx.MockTransport)
        self._success_statuses = {s.lower() for s in config.terminal_success_statuses}
        self._failure_statuses = {s.lower() for s in config.terminal_failure_statuses}

    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=httpx.Timeout(self._config.timeout_seconds))

    def _headers(self) -> dict:
        return {
            "X-ApiKey": self._config.api_key,
            "X-Application-Id": self._config.application_id,
        }

    def _submit_url(self) -> str:
        return f"{self._config.base_url.rstrip('/')}/api/v1/workflows/{self._config.workflow_id}/execute"

    def _execution_url(self, execution_id: str) -> str:
        return (
            f"{self._config.base_url.rstrip('/')}/api/v1/workflows/"
            f"{self._config.workflow_id}/executions/{execution_id}"
        )

    # -- submission ------------------------------------------------------------

    def submit(
        self,
        *,
        text: Optional[str] = None,
        filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        content_type: Optional[str] = None,
        raw_text: bool = False,
    ) -> SubmissionResult:
        start = time.monotonic()

        # Explicitly supplied text must be a non-blank string; never silently
        # drop invalid text, even when a valid attachment is present.
        if text is not None and (not isinstance(text, str) or not text.strip()):
            return SubmissionResult(
                outcome=EvolusOutcome.MISSING_EVIDENCE,
                reasons=["text must be a non-blank string when supplied"],
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        # A document must be supplied as text and/or a file attachment. Text and
        # file are independent and are never auto-duplicated into each other. The
        # submitted text is never stripped or otherwise altered.
        has_text = isinstance(text, str) and bool(text.strip())
        has_file = filename is not None or file_bytes is not None

        if has_file:
            if filename is None or file_bytes is None:
                return SubmissionResult(
                    outcome=EvolusOutcome.MISSING_EVIDENCE,
                    reasons=["attachment requires both filename and file bytes"],
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )
            if not isinstance(filename, str) or not filename.strip():
                return SubmissionResult(
                    outcome=EvolusOutcome.MISSING_EVIDENCE,
                    reasons=["attachment filename is blank or not a string"],
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )
            if not isinstance(file_bytes, (bytes, bytearray)):
                return SubmissionResult(
                    outcome=EvolusOutcome.MISSING_EVIDENCE,
                    reasons=["attachment bytes are not bytes"],
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )
            if len(file_bytes) == 0:
                return SubmissionResult(
                    outcome=EvolusOutcome.MISSING_EVIDENCE,
                    reasons=["attachment bytes are empty"],
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )

        if raw_text and has_file:
            return SubmissionResult(
                outcome=EvolusOutcome.MISSING_EVIDENCE,
                reasons=["raw text submission cannot be combined with an attachment"],
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        if not has_text and not has_file:
            return SubmissionResult(
                outcome=EvolusOutcome.MISSING_EVIDENCE,
                reasons=["no document content"],
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        headers = self._headers()
        if raw_text:
            raw_headers = {**headers, "Content-Type": content_type or "text/plain"}
            try:
                resp = self._http().post(self._submit_url(), content=text.encode("utf-8"), headers=raw_headers)
            except httpx.TimeoutException:
                return SubmissionResult(
                    outcome=EvolusOutcome.TIMEOUT, reasons=["submission timed out; not retried"],
                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000),
                )
            except httpx.RequestError:
                return SubmissionResult(
                    outcome=EvolusOutcome.UNAVAILABLE, reasons=["submission transport failure"],
                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000),
                )
        else:
            # Multipart form-data: optional `input` text field and/or the `file`
            # attachment. The two are separate; neither is auto-populated from the other.
            files = {}
            if has_text:
                files[self._config.submit_text_field] = (None, text)
            if has_file:
                if not self._config.submit_attachment_field:
                    return SubmissionResult(
                        outcome=EvolusOutcome.UNAVAILABLE,
                        reasons=["attachment field name is not configured"],
                        elapsed_ms=int((time.monotonic() - start) * 1000),
                    )
                files[self._config.submit_attachment_field] = (filename, file_bytes, content_type or "text/plain")
            try:
                resp = self._http().post(self._submit_url(), files=files, headers=headers)
            except httpx.TimeoutException:
                return SubmissionResult(
                    outcome=EvolusOutcome.TIMEOUT, reasons=["submission timed out; not retried"],
                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000),
                )
            except httpx.RequestError:
                return SubmissionResult(
                    outcome=EvolusOutcome.UNAVAILABLE, reasons=["submission transport failure"],
                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000),
                )

        if resp.status_code in (401, 403):
            return SubmissionResult(outcome=EvolusOutcome.AUTH_FAILURE, reasons=["rejected credentials"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))
        if resp.status_code == 409:
            return SubmissionResult(outcome=EvolusOutcome.WORKFLOW_INACTIVE, reasons=["workflow inactive"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))
        if resp.status_code == 404:
            return SubmissionResult(outcome=EvolusOutcome.WORKFLOW_NOT_FOUND, reasons=["workflow or resource not found"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))
        if resp.status_code >= 500:
            return SubmissionResult(outcome=EvolusOutcome.UNAVAILABLE, reasons=["submission server error"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))
        if resp.status_code != 202:
            return SubmissionResult(outcome=EvolusOutcome.UNEXPECTED_STRUCTURE,
                                    reasons=[f"unexpected submission status {resp.status_code}"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))

        try:
            body = resp.json()
        except ValueError:
            return SubmissionResult(outcome=EvolusOutcome.MALFORMED_RESPONSE,
                                    reasons=["submission response is not valid JSON"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))
        try:
            submission = WorkflowSubmission.model_validate(body)
        except Exception:
            return SubmissionResult(outcome=EvolusOutcome.INCOMPLETE_RESPONSE,
                                    reasons=["submission acknowledgment missing required fields"],
                                    call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))

        return SubmissionResult(outcome=EvolusOutcome.SUCCESS, submission=submission,
                                call_count=1, elapsed_ms=int((time.monotonic() - start) * 1000))

    # -- polling (single GET) ----------------------------------------------------

    def _poll(self, execution_id: str) -> tuple[EvolusOutcome, Optional[dict], list[str]]:
        try:
            resp = self._http().get(self._execution_url(execution_id), headers=self._headers())
        except httpx.TimeoutException:
            return EvolusOutcome.TIMEOUT, None, ["poll timed out"]
        except httpx.RequestError:
            return EvolusOutcome.UNAVAILABLE, None, ["poll transport failure"]

        if resp.status_code in (401, 403):
            return EvolusOutcome.AUTH_FAILURE, None, ["rejected credentials"]
        if resp.status_code == 409:
            return EvolusOutcome.WORKFLOW_INACTIVE, None, ["workflow inactive"]
        if resp.status_code == 404:
            return EvolusOutcome.WORKFLOW_NOT_FOUND, None, ["execution or workflow not found"]
        if resp.status_code >= 500:
            return EvolusOutcome.UNAVAILABLE, None, ["poll server error"]
        if resp.status_code != 200:
            return EvolusOutcome.UNEXPECTED_STRUCTURE, None, [f"unexpected poll status {resp.status_code}"]

        try:
            body = resp.json()
        except ValueError:
            return EvolusOutcome.MALFORMED_RESPONSE, None, ["poll response is not valid JSON"]
        if not isinstance(body, dict):
            return EvolusOutcome.MALFORMED_RESPONSE, None, ["poll response is not an object"]
        return EvolusOutcome.SUCCESS, body, []

    # -- reconcile (bounded polling + validation) ---------------------------------

    @staticmethod
    def _extract_errors(body) -> list[str]:
        """Collect the documented singular ``error`` field and any error collection."""
        errors: list[str] = []
        err = body.get("error")
        if err:
            errors.append(str(err))
        errs = body.get("errors")
        if isinstance(errs, list):
            errors.extend(str(e) for e in errs if e)
        elif errs:
            errors.append(str(errs))
        return errors

    def reconcile(self, execution_id: str, original_evidence: str) -> EvolusExecutionResult:
        start = time.monotonic()
        polls = 0
        for _ in range(self._config.max_polls):
            outcome, body, reasons = self._poll(execution_id)
            polls += 1
            if outcome != EvolusOutcome.SUCCESS:
                # Polling is idempotent; a transient GET failure is not fatal here.
                if outcome in (EvolusOutcome.TIMEOUT, EvolusOutcome.UNAVAILABLE):
                    time.sleep(self._config.poll_interval_seconds)
                    continue
                return EvolusExecutionResult(
                    outcome=outcome, execution_id=execution_id, original_evidence=original_evidence,
                    reasons=reasons, call_count=polls, elapsed_ms=int((time.monotonic() - start) * 1000),
                )

            status = body.get("status")
            status_norm = status.lower() if isinstance(status, str) else None
            errors = self._extract_errors(body)

            if errors or (status_norm in self._failure_statuses):
                return EvolusExecutionResult(
                    outcome=EvolusOutcome.WORKFLOW_FAILED, execution_id=execution_id, status=str(status),
                    errors=errors, original_evidence=original_evidence,
                    reasons=["workflow execution failed"], call_count=polls,
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )

            if status_norm in self._success_statuses:
                return self._finalize(execution_id, status, body, original_evidence, polls, start)

            # Pending/running: poll again after the configured interval.
            time.sleep(self._config.poll_interval_seconds)

        return EvolusExecutionResult(
            outcome=EvolusOutcome.POLL_EXHAUSTED, execution_id=execution_id, original_evidence=original_evidence,
            reasons=[f"execution did not reach a terminal state within {self._config.max_polls} polls"],
            call_count=polls, elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    def _finalize(self, execution_id, status, body, original_evidence, polls, start):
        raw_output = body.get("output")
        output = _deep_decode(raw_output)
        outcome, reason = _validate_final_output(output, self._config)
        if outcome != EvolusOutcome.SUCCESS:
            return EvolusExecutionResult(
                outcome=outcome, execution_id=execution_id, status=str(status),
                output=output if isinstance(output, dict) else None,
                original_evidence=original_evidence, reasons=[reason or "invalid final output"],
                call_count=polls, elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        return EvolusExecutionResult(
            outcome=EvolusOutcome.SUCCESS, execution_id=execution_id, status=str(status),
            output=output,
            extraction=output.get("extraction"),
            advisory_analysis=output.get("advisoryAnalysis"),
            authority=output.get("authority"), evidence_status=output.get("evidenceStatus"),
            original_evidence=original_evidence, call_count=polls,
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    # -- full flow ---------------------------------------------------------------

    def run(
        self,
        *,
        text: Optional[str] = None,
        filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> EvolusExecutionResult:
        start = time.monotonic()
        # Verbatim text (never stripped), plus byte-exact file provenance. The
        # decoded file text is not used as proof of byte-exact preservation.
        original_evidence = text if isinstance(text, str) else ""
        original_file_name = filename if file_bytes is not None else None
        original_file_sha256 = (
            hashlib.sha256(file_bytes).hexdigest()
            if isinstance(file_bytes, (bytes, bytearray))
            else None
        )
        submission = self.submit(text=text, filename=filename, file_bytes=file_bytes, content_type=content_type)
        if submission.outcome != EvolusOutcome.SUCCESS:
            return EvolusExecutionResult(
                outcome=submission.outcome, original_evidence=original_evidence,
                original_file_name=original_file_name, original_file_sha256=original_file_sha256,
                reasons=submission.reasons, call_count=submission.call_count,
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
        result = self.reconcile(submission.submission.execution_id, original_evidence)
        result.original_file_name = original_file_name
        result.original_file_sha256 = original_file_sha256
        result.call_count += 1  # count the POST
        result.elapsed_ms = int((time.monotonic() - start) * 1000)
        return result
