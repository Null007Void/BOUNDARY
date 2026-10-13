"""B4.2 real Evolus Integration API HTTP client tests (offline, mocked HTTP)."""

import hashlib
import json

import httpx
import pytest

from boundary.evolus import EvolusOutcome
from boundary.evolus_smoke_test import main, verify_identity
from boundary.evolus_http import (
    DEFAULT_APPLICATION_ID,
    EvolusHttpClient,
    EvolusHttpConfig,
    _deep_decode,
)

BASE = "https://api.evolus.ai"


def _config(**overrides):
    data = dict(base_url=BASE, api_key="test-key", poll_interval_seconds=0, max_polls=5)
    data.update(overrides)
    return EvolusHttpConfig(**data)


def _client(handler, **config_overrides):
    config = _config(**config_overrides)
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)
    return EvolusHttpClient(config, client=client)


def _submission_response(execution_id="exec-1"):
    return httpx.Response(
        202,
        json={
            "executionId": execution_id,
            "jobId": "job-1",
            "enqueuedAt": "2026-10-12T00:00:00Z",
            "traceId": "trace-1",
        },
    )


def _extraction():
    return {
        "documentType": "INVOICE",
        "supplierId": "SUP-9001",
        "invoiceId": "INV-2048",
        "purchaseOrderId": None,
        "purchaseOrderReference": "PO-77",
        "issueDate": "2026-10-01",
        "currency": "USD",
        "statedTotal": "USD 1,250.00",
        "lineItems": json.dumps([{"line": 1, "description": "item", "qty": "1", "price": "1250.00"}]),
        "supportingReferences": json.dumps(["PO-77"]),
        "missingFields": json.dumps([]),
        "extractionWarnings": json.dumps([]),
    }


def _realistic_output_obj():
    return {
        "authority": "NONE",
        "evidenceStatus": "unverified",
        "extraction": _extraction(),
        "advisoryAnalysis": json.dumps({"summary": "ambiguous discrepancy", "confidence": 0.6}),
    }


def _execution_response(status="Completed", output=None, errors=None, error=None):
    body = {"status": status, "steps": []}
    if output is not None:
        # Real output is a JSON-encoded string.
        body["output"] = output if isinstance(output, str) else json.dumps(output)
    if errors is not None:
        body["errors"] = errors
    if error is not None:
        body["error"] = error
    return httpx.Response(200, json=body)


# --- Submission contract (finding 1) ------------------------------------------------


def test_submit_uses_multipart_input_field():
    captured = {}

    def handler(request):
        captured["content_type"] = request.headers.get("Content-Type")
        captured["content"] = request.content
        return _submission_response()

    client = _client(handler)
    client.submit(text="invoice-123")
    assert "multipart/form-data" in captured["content_type"]
    assert b'name="input"' in captured["content"]
    assert b"invoice-123" in captured["content"]


def test_submit_raw_text_body():
    captured = {}

    def handler(request):
        captured["content_type"] = request.headers.get("Content-Type")
        captured["content"] = request.content
        return _submission_response()

    client = _client(handler)
    client.submit(text="invoice-123", raw_text=True)
    assert captured["content"] == b"invoice-123"
    assert "text/plain" in captured["content_type"]


def test_submit_captures_execution_id_and_headers():
    captured = {}

    def handler(request):
        captured["api_key"] = request.headers.get("X-ApiKey")
        captured["app_id"] = request.headers.get("X-Application-Id")
        return _submission_response()

    client = _client(handler)
    result = client.submit(text="synthetic document")
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.submission.execution_id == "exec-1"
    assert captured["api_key"] == "test-key"
    assert captured["app_id"] == DEFAULT_APPLICATION_ID


def test_multipart_attachment_refused_without_field_name():
    def handler(request):
        raise AssertionError("must not POST without an attachment field name configured")

    client = _client(handler, submit_attachment_field="")  # explicitly unconfigured
    result = client.submit(filename="invoice.txt", file_bytes=b"invoice content")
    assert result.outcome == EvolusOutcome.UNAVAILABLE
    assert result.call_count == 0


def test_file_attachment_structure():
    captured = {}

    def handler(request):
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.content
        return _submission_response()

    client = _client(handler)
    result = client.submit(
        filename="BOUNDARY_synthetic_invoice_INV-2048.txt",
        file_bytes=b"synthetic invoice body",
        content_type="text/plain",
    )
    assert result.outcome == EvolusOutcome.SUCCESS
    assert "multipart/form-data" in captured["content_type"]
    body = captured["body"]
    assert b'name="file"' in body
    assert b'filename="BOUNDARY_synthetic_invoice_INV-2048.txt"' in body
    assert b"text/plain" in body
    assert b"synthetic invoice body" in body
    assert b'name="input"' not in body  # text is not auto-duplicated into the file


def test_text_submission_does_not_add_file():
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return _submission_response()

    client = _client(handler)
    client.submit(text="synthetic document")
    body = captured["body"]
    assert b'name="input"' in body
    assert b'name="file"' not in body  # no file part when only text is submitted


def test_attachment_requires_filename_and_bytes():
    def handler(request):
        raise AssertionError("must not POST with an incomplete attachment")

    client = _client(handler)
    result = client.submit(filename="invoice.txt")  # no file_bytes
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_attachment_field_configurable():
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return _submission_response()

    client = _client(handler, submit_attachment_field="document")
    client.submit(filename="invoice.txt", file_bytes=b"content", content_type="text/plain")
    assert b'name="document"' in captured["body"]


# --- HTTP status handling (finding 2) ------------------------------------------------


def test_409_workflow_inactive():
    def handler(request):
        return httpx.Response(409, json={"error": "inactive"})

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.WORKFLOW_INACTIVE


def test_404_workflow_not_found():
    def handler(request):
        return httpx.Response(404, json={"error": "not found"})

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.WORKFLOW_NOT_FOUND


def test_auth_failure():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.AUTH_FAILURE


# --- Execution status handling (finding 3) -------------------------------------------


def test_capitalized_completed_status_succeeds():
    # Representative real response: capitalized status, string-encoded output, no error.
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=_realistic_output_obj())

    client = _client(handler)
    result = client.run(text="synthetic document")
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.execution_id == "exec-1"
    assert result.authority == "NONE"
    assert result.evidence_status == "unverified"
    assert result.output is not None


def test_singular_error_field_fails():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", error="extraction failed")

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.WORKFLOW_FAILED
    assert "extraction failed" in result.errors


def test_error_collection_fails():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="failed", errors=["step failed"])

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.WORKFLOW_FAILED
    assert "step failed" in result.errors


# --- Strict final-output validation (finding 4) --------------------------------------


def test_full_run_success_preserves_original_evidence():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=_realistic_output_obj())

    client = _client(handler)
    result = client.run(text="synthetic document")
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.original_evidence == "synthetic document"
    assert result.call_count == 2  # 1 POST + 1 GET


def test_output_not_object_rejected():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output="[1, 2, 3]")

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE
    assert result.output is None


def test_missing_extraction_rejected():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "advisoryAnalysis": "text"}

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_empty_extraction_rejected():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": {}, "advisoryAnalysis": "text"}

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_missing_advisory_rejected():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": _extraction()}

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_malformed_line_items_rejected():
    extraction = _extraction()
    extraction["lineItems"] = "not a json array"
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_authority_nested_only_rejected():
    # authority appears only inside nested advisory text, not at the top level.
    output = {
        "evidenceStatus": "unverified",
        "extraction": _extraction(),
        "advisoryAnalysis": json.dumps({"authority": "NONE", "note": "nested"}),
    }

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_authority_not_none_rejected():
    output = {
        "authority": "APPROVE",
        "evidenceStatus": "unverified",
        "extraction": _extraction(),
        "advisoryAnalysis": "text",
    }

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.AUTHORITY_CLAIM


def test_evidence_status_verified_rejected():
    output = {
        "authority": "NONE",
        "evidenceStatus": "verified",
        "extraction": _extraction(),
        "advisoryAnalysis": "text",
    }

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


# --- General robustness -------------------------------------------------------------


def test_polling_until_terminal():
    state = {"polls": 0}

    def handler(request):
        if request.method == "POST":
            return _submission_response()
        state["polls"] += 1
        if state["polls"] < 3:
            return _execution_response(status="running")
        return _execution_response(status="Completed", output=_realistic_output_obj())

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.SUCCESS
    assert state["polls"] == 3


def test_malformed_poll_response():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return httpx.Response(200, content=b"not json")

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.MALFORMED_RESPONSE


def test_missing_evidence_no_post():
    def handler(request):
        raise AssertionError("transport must not be invoked for blank evidence")

    client = _client(handler)
    result = client.run(text="   ")
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_submit_timeout_not_retried():
    post_count = {"n": 0}

    def handler(request):
        if request.method == "POST":
            post_count["n"] += 1
            raise httpx.ReadTimeout("timed out", request=request)
        return _execution_response(status="Completed", output=_realistic_output_obj())

    client = _client(handler)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.TIMEOUT
    assert post_count["n"] == 1


def test_poll_exhausted():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="running")

    client = _client(handler, max_polls=3)
    result = client.run(text="doc")
    assert result.outcome == EvolusOutcome.POLL_EXHAUSTED


def test_reconcile_by_execution_id_preserves_identity():
    def handler(request):
        if request.method == "GET":
            return _execution_response(status="Completed", output=_realistic_output_obj())
        return _submission_response()

    client = _client(handler)
    result = client.reconcile("exec-known", "original evidence text")
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.execution_id == "exec-known"
    assert result.original_evidence == "original evidence text"


# --- Decoding helper -----------------------------------------------------------------


def test_deep_decode_unwraps_nested_json():
    nested = '{"a": "[1, 2, 3]", "b": "{\\"x\\": \\"y\\"}"}'
    decoded = _deep_decode(nested)
    assert decoded == {"a": [1, 2, 3], "b": {"x": "y"}}


def test_file_run_returns_typed_result():
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=_realistic_output_obj())

    client = _client(handler)
    result = client.run(
        filename="BOUNDARY_synthetic_invoice_INV-2048.txt",
        file_bytes=b"BOUNDARY synthetic invoice INV-2048",
        content_type="text/plain",
    )
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.execution_id == "exec-1"
    assert result.extraction is not None
    assert result.extraction["invoiceId"] == "INV-2048"
    assert result.extraction["statedTotal"] == "USD 1,250.00"
    assert result.advisory_analysis is not None
    # File-only submission: original evidence is the file, tracked byte-exactly.
    assert result.original_evidence == ""
    assert result.original_file_name == "BOUNDARY_synthetic_invoice_INV-2048.txt"
    assert result.original_file_sha256 == hashlib.sha256(b"BOUNDARY synthetic invoice INV-2048").hexdigest()


# --- Extractor contract (finding 3/4) ------------------------------------------------


def _run_output(output):
    def handler(request):
        if request.method == "POST":
            return _submission_response()
        return _execution_response(status="Completed", output=output)

    client = _client(handler)
    return client.run(text="doc")


def test_missing_array_field_rejected():
    extraction = _extraction()
    del extraction["supportingReferences"]
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_identity_field_wrong_type_rejected():
    extraction = _extraction()
    extraction["invoiceId"] = 12345  # integer, not a string
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_null_identity_field_permitted():
    extraction = _extraction()
    extraction["invoiceId"] = None  # source document lacks the value
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.SUCCESS


def test_document_type_wrong_choice_rejected():
    extraction = _extraction()
    extraction["documentType"] = "receipt"  # not a configured choice
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_document_type_null_permitted():
    extraction = _extraction()
    extraction["documentType"] = None
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.SUCCESS


def test_advisory_boolean_rejected():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": _extraction(), "advisoryAnalysis": True}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_advisory_number_rejected():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": _extraction(), "advisoryAnalysis": 42}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


# --- Authoritative 12-field extractor contract ---------------------------------------


def test_genuine_invoice_extraction_passes():
    extraction = {
        "documentType": "INVOICE",
        "supplierId": "SUP-9001",
        "invoiceId": "INV-2048",
        "purchaseOrderId": None,
        "purchaseOrderReference": "PO-77",
        "issueDate": "2026-10-01",
        "currency": "USD",
        "statedTotal": "USD 1,250.00",
        "lineItems": json.dumps([]),
        "supportingReferences": json.dumps(["PO-77"]),
        "missingFields": json.dumps([]),
        "extractionWarnings": json.dumps([]),
    }
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "ambiguous"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.SUCCESS
    # monetary string preserved verbatim, not normalized
    assert result.output["extraction"]["statedTotal"] == "USD 1,250.00"


def test_only_empty_arrays_fails():
    extraction = {
        "lineItems": json.dumps([]),
        "supportingReferences": json.dumps([]),
        "missingFields": json.dumps([]),
        "extractionWarnings": json.dumps([]),
    }
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_numeric_purchase_order_reference_fails():
    extraction = _extraction()
    extraction["purchaseOrderReference"] = 12345
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_missing_scalar_key_fails():
    extraction = _extraction()
    del extraction["currency"]
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_unknown_document_type_accepted():
    extraction = _extraction()
    extraction["documentType"] = "UNKNOWN"
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": extraction, "advisoryAnalysis": "text"}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.SUCCESS


def test_malformed_json_advisory_fails():
    output = {
        "authority": "NONE",
        "evidenceStatus": "unverified",
        "extraction": _extraction(),
        "advisoryAnalysis": '{"summary": "broken',
    }
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_whitespace_advisory_fails():
    output = {"authority": "NONE", "evidenceStatus": "unverified", "extraction": _extraction(), "advisoryAnalysis": "   "}
    result = _run_output(output)
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


# --- B4.3 review corrections: attachment validation + evidence preservation ----------


def test_empty_attachment_bytes_rejected():
    def handler(request):
        raise AssertionError("must not POST with empty attachment bytes")

    client = _client(handler)
    result = client.submit(filename="invoice.txt", file_bytes=b"")
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_blank_filename_rejected():
    def handler(request):
        raise AssertionError("must not POST with a blank filename")

    client = _client(handler)
    result = client.submit(filename="   ", file_bytes=b"content")
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_invalid_attachment_type_rejected():
    def handler(request):
        raise AssertionError("must not POST with non-bytes attachment")

    client = _client(handler)
    result = client.submit(filename="invoice.txt", file_bytes="not bytes")  # str, not bytes
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_raw_text_with_attachment_rejected():
    def handler(request):
        raise AssertionError("must not POST raw text combined with an attachment")

    client = _client(handler)
    result = client.submit(text="hello", filename="invoice.txt", file_bytes=b"content", raw_text=True)
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_text_not_stripped():
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return _submission_response()

    client = _client(handler)
    client.submit(text="  padded  ")
    assert b"  padded  " in captured["body"]


def test_simultaneous_text_and_file_preserved():
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return _submission_response()

    client = _client(handler)
    client.submit(text="a note", filename="invoice.txt", file_bytes=b"file content")
    body = captured["body"]
    assert b'name="input"' in body
    assert b"a note" in body
    assert b'name="file"' in body
    assert b"file content" in body


def test_verify_identity_matches_fixture():
    from boundary.evolus_http import EvolusExecutionResult

    ok = EvolusExecutionResult(outcome=EvolusOutcome.SUCCESS, extraction={"invoiceId": "INV-2048", "statedTotal": "USD 1,250.00"})
    assert verify_identity(ok) is True

    bad = EvolusExecutionResult(outcome=EvolusOutcome.SUCCESS, extraction={"invoiceId": "OTHER", "statedTotal": "USD 1,250.00"})
    assert verify_identity(bad) is False


def test_smoke_dry_run_no_network():
    assert main([]) == 0


def test_smoke_live_requires_file():
    assert main(["--live"]) == 1


def test_smoke_unreadable_file():
    assert main(["--live", "--file", "/nonexistent/nope.txt"]) == 1


def test_non_string_text_rejected_even_with_attachment():
    def handler(request):
        raise AssertionError("must not POST when invalid text is supplied")

    client = _client(handler)
    result = client.submit(text=123, filename="invoice.txt", file_bytes=b"invoice")
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0


def test_whitespace_text_rejected_even_with_attachment():
    def handler(request):
        raise AssertionError("must not POST when whitespace text is supplied")

    client = _client(handler)
    result = client.submit(text="   ", filename="invoice.txt", file_bytes=b"invoice")
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.call_count == 0
