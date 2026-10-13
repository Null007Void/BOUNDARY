"""B4.1 Evolus integration foundation tests (offline, mocked transports)."""

from boundary.evolus import (
    DocumentExtractionCall,
    DocumentExtractionRequest,
    EvolusAdapter,
    EvolusAuthError,
    EvolusConfig,
    EvolusOutcome,
    EvolusTimeout,
    EvolusUnavailable,
    WorkflowCall,
    WorkflowRequest,
)


def _config():
    return EvolusConfig(base_url="", api_key="test-key")


def _extract_request(**overrides):
    data = dict(document_id="doc-1", document_content="invoice document text", fields=["supplier_id", "invoice_id"])
    data.update(overrides)
    return DocumentExtractionRequest(**data)


def _workflow_request(**overrides):
    data = dict(workflow_id="wf-1", case_id="case-1", step_ids=["extract", "classify"])
    data.update(overrides)
    return WorkflowRequest(**data)


def _extract_response(**overrides):
    data = dict(
        document_id="doc-1",
        extractor_id="extractor-v1",
        claims=[
            {"field": "supplier_id", "value": "SUP-9001", "confidence": 0.9},
            {"field": "invoice_id", "value": "INV-2048", "confidence": 0.8},
        ],
    )
    data.update(overrides)
    return data


def _workflow_response(**overrides):
    data = dict(
        workflow_id="wf-1",
        run_id="run-1",
        status="SUCCEEDED",
        steps=[
            {"step_id": "extract", "status": "SUCCEEDED", "notes": ""},
            {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
        ],
    )
    data.update(overrides)
    return data


# --- Success paths ----------------------------------------------------------------


def test_document_extraction_success():
    captured = {}

    def transport(payload):
        captured["payload"] = payload
        return _extract_response()

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())

    assert result.outcome == EvolusOutcome.SUCCESS
    assert isinstance(result, DocumentExtractionCall)
    assert result.result.claims[0].field == "supplier_id"
    assert result.result.execution_authority == "NONE"
    assert result.telemetry.call_count == 1
    assert captured["payload"]["document_id"] == "doc-1"
    assert captured["payload"]["fields"] == ["supplier_id", "invoice_id"]


def test_workflow_success():
    def transport(payload):
        return _workflow_response()

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())

    assert result.outcome == EvolusOutcome.SUCCESS
    assert isinstance(result, WorkflowCall)
    assert result.result.status == "SUCCEEDED"
    assert result.result.execution_authority == "NONE"
    assert result.telemetry.call_count == 1


# --- Rejection paths ---------------------------------------------------------------


def test_malformed_response_rejected():
    def transport(payload):
        return ["not", "a", "dict"]

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.MALFORMED_RESPONSE
    assert result.result is None


def test_authority_key_rejected():
    def transport(payload):
        return _extract_response(approve=True)

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.AUTHORITY_CLAIM
    assert result.result is None


def test_execution_authority_override_rejected():
    def transport(payload):
        return _extract_response(execution_authority="APPROVE")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.AUTHORITY_CLAIM
    assert result.result is None


def test_contradictory_document_id_rejected():
    def transport(payload):
        return _extract_response(document_id="OTHER-DOC")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.CONTRADICTORY_RESPONSE


def test_contradictory_workflow_id_rejected():
    def transport(payload):
        return _workflow_response(workflow_id="OTHER-WF")

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.CONTRADICTORY_RESPONSE


def test_missing_evidence_rejected_without_call():
    adapter = EvolusAdapter(_config(), extract_transport=lambda p: _extract_response())
    result = adapter.extract_document(_extract_request(document_content="   "))
    assert result.outcome == EvolusOutcome.MISSING_EVIDENCE
    assert result.telemetry.call_count == 0


def test_duplicate_claim_fields_rejected():
    claims = [
        {"field": "supplier_id", "value": "SUP-1", "confidence": 0.9},
        {"field": "supplier_id", "value": "SUP-2", "confidence": 0.7},
    ]

    def transport(payload):
        return _extract_response(claims=claims)

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.CONTRADICTORY_RESPONSE


# --- Transport failure mapping ------------------------------------------------------


def test_timeout_mapped():
    def transport(payload):
        raise EvolusTimeout("timed out")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.TIMEOUT
    assert result.telemetry.failure_state == "timeout"


def test_auth_failure_mapped():
    def transport(payload):
        raise EvolusAuthError("bad key")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.AUTH_FAILURE


def test_unavailable_mapped():
    def transport(payload):
        raise EvolusUnavailable("down")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.UNAVAILABLE


# --- Telemetry ----------------------------------------------------------------------


def test_usage_captured_when_reported():
    def transport(payload):
        return _extract_response(usage={"documents": 1, "tokens": 42})

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.telemetry.usage == {"documents": 1, "tokens": 42}


def test_no_transport_configured_fails_closed():
    adapter = EvolusAdapter(_config())  # no transports injected
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.UNAVAILABLE
    assert result.telemetry.call_count == 0


# --- Finding 1: malformed responses fail closed -------------------------------------


def test_transport_returns_none_fails_closed():
    def transport(payload):
        return None

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.MALFORMED_RESPONSE
    assert result.result is None
    assert result.telemetry.call_count == 1


def test_malformed_usage_list_fails_closed():
    def transport(payload):
        return _extract_response(usage=["not", "an", "object"])

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.MALFORMED_RESPONSE
    assert result.result is None


def test_malformed_usage_string_fails_closed():
    def transport(payload):
        return _extract_response(usage="not an object")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.MALFORMED_RESPONSE


# --- Finding 2: count actual transport attempts -------------------------------------


def test_timeout_counts_attempt():
    def transport(payload):
        raise EvolusTimeout("timed out")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.TIMEOUT
    assert result.telemetry.call_count == 1


def test_auth_failure_counts_attempt():
    def transport(payload):
        raise EvolusAuthError("bad key")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.telemetry.call_count == 1


def test_unavailable_counts_attempt():
    def transport(payload):
        raise EvolusUnavailable("down")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.telemetry.call_count == 1


# --- Finding 3: extraction completeness and provenance -------------------------------


def test_missing_requested_fields_incomplete():
    def transport(payload):
        return _extract_response(
            claims=[{"field": "supplier_id", "value": "SUP-1", "confidence": 0.9}]
        )

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())  # fields include invoice_id
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_empty_claims_incomplete():
    def transport(payload):
        return _extract_response(claims=[])

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_blank_extractor_identity_incomplete():
    def transport(payload):
        return _extract_response(extractor_id="   ")

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_whitespace_claim_value_incomplete():
    claims = [
        {"field": "supplier_id", "value": "   ", "confidence": 0.9},
        {"field": "invoice_id", "value": "INV-1", "confidence": 0.8},
    ]

    def transport(payload):
        return _extract_response(claims=claims)

    adapter = EvolusAdapter(_config(), extract_transport=transport)
    result = adapter.extract_document(_extract_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


# --- Finding 4: truthful workflow outcomes ------------------------------------------


def test_workflow_failed_status_distinct():
    def transport(payload):
        return _workflow_response(status="FAILED")

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.WORKFLOW_FAILED
    assert result.result is not None
    assert result.result.status == "FAILED"


def test_workflow_missing_steps_incomplete():
    def transport(payload):
        return _workflow_response(steps=[{"step_id": "extract", "status": "SUCCEEDED", "notes": ""}])

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())  # requests extract + classify
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_workflow_duplicate_steps_contradictory():
    steps = [
        {"step_id": "extract", "status": "SUCCEEDED", "notes": ""},
        {"step_id": "extract", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request(step_ids=["extract"]))
    assert result.outcome == EvolusOutcome.CONTRADICTORY_RESPONSE


# --- Workflow step outcomes and run provenance ---------------------------------------


def test_required_step_failed():
    steps = [
        {"step_id": "extract", "status": "FAILED", "notes": ""},
        {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.WORKFLOW_FAILED


def test_required_step_pending():
    steps = [
        {"step_id": "extract", "status": "PENDING", "notes": ""},
        {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_required_step_blank_status():
    steps = [
        {"step_id": "extract", "status": "", "notes": ""},
        {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_blank_run_id_incomplete():
    def transport(payload):
        return _workflow_response(run_id="   ")

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_blank_step_identifier_incomplete():
    steps = [
        {"step_id": "  ", "status": "SUCCEEDED", "notes": ""},
        {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.INCOMPLETE_RESPONSE


def test_valid_completed_workflow():
    steps = [
        {"step_id": "extract", "status": "SUCCEEDED", "notes": ""},
        {"step_id": "classify", "status": "SUCCEEDED", "notes": ""},
    ]

    def transport(payload):
        return _workflow_response(steps=steps)

    adapter = EvolusAdapter(_config(), workflow_transport=transport)
    result = adapter.run_workflow(_workflow_request())
    assert result.outcome == EvolusOutcome.SUCCESS
    assert result.result.run_id == "run-1"
    assert result.result.execution_authority == "NONE"
