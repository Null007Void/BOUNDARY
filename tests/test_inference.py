import json

from boundary.inference import (
    EvidenceInterpretation,
    InferenceAdapter,
    InferenceAuthError,
    InferenceConfig,
    InferenceOutcome,
    InferenceTimeout,
    InferenceUnavailable,
)


def _content(**overrides):
    data = {
        "supplier_id": "SUP-1",
        "invoice_id": "INV-1",
        "po_reference": "PO-1",
        "amount_cents": 10000,
        "discrepancy_cents": 0,
        "confidence": 0.9,
        "notes": "clear",
    }
    data.update(overrides)
    return json.dumps(data)


def _raw(content=None, usage=None):
    return {
        "choices": [{"message": {"content": content if content is not None else _content()}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _adapter(transport):
    config = InferenceConfig(base_url="http://mock.local", model_id="DeepSeek-V4-Flash")
    return InferenceAdapter(config, transport)


def test_successful_inference():
    adapter = _adapter(lambda payload: _raw())
    result = adapter.interpret("ambiguous evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert isinstance(result.interpretation, EvidenceInterpretation)
    assert result.interpretation.supplier_id == "SUP-1"
    assert result.interpretation.amount_cents == 10000


def test_token_usage_capture():
    adapter = _adapter(lambda payload: _raw())
    result = adapter.interpret("evidence")
    assert result.telemetry.prompt_tokens == 10
    assert result.telemetry.completion_tokens == 20
    assert result.telemetry.total_tokens == 30
    assert result.telemetry.model_id == "DeepSeek-V4-Flash"
    assert result.telemetry.elapsed_ms >= 0


def test_malformed_json():
    adapter = _adapter(lambda payload: _raw(content="not json at all"))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.MALFORMED_RESPONSE
    assert result.interpretation is None


def test_missing_required_fields():
    data = {"supplier_id": "SUP-1"}  # missing invoice_id, amount_cents, etc.
    adapter = _adapter(lambda payload: _raw(content=json.dumps(data)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_unexpected_response_structure():
    # No "choices" key at all.
    adapter = _adapter(lambda payload: {"foo": "bar"})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNEXPECTED_STRUCTURE


def test_unavailable_service():
    adapter = _adapter(lambda payload: (_ for _ in ()).throw(InferenceUnavailable("down")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.UNAVAILABLE
    assert result.telemetry.error_class == "unavailable"


def test_timeout():
    adapter = _adapter(lambda payload: (_ for _ in ()).throw(InferenceTimeout("too slow")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TIMEOUT
    assert result.telemetry.error_class == "timeout"


def test_auth_failure():
    adapter = _adapter(lambda payload: (_ for _ in ()).throw(InferenceAuthError("401")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.AUTH_FAILURE
    assert result.telemetry.error_class == "auth_failure"


def test_authority_claim():
    # The model tries to assert approval authority.
    data = {
        "supplier_id": "SUP-1",
        "invoice_id": "INV-1",
        "amount_cents": 10000,
        "discrepancy_cents": 0,
        "confidence": 0.9,
        "approve": True,
    }
    adapter = _adapter(lambda payload: _raw(content=json.dumps(data)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.AUTHORITY_CLAIM
    assert result.interpretation is None


def test_contradictory_response():
    data = {
        "supplier_id": "SUP-1",
        "invoice_id": "INV-1",
        "amount_cents": -5,
        "discrepancy_cents": 0,
        "confidence": 0.9,
    }
    adapter = _adapter(lambda payload: _raw(content=json.dumps(data)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.CONTRADICTORY_RESPONSE


def test_config_does_not_repr_secret():
    config = InferenceConfig(base_url="http://x", api_key="super-secret-token")
    assert "super-secret-token" not in repr(config)
    assert "super-secret-token" not in str(config)


def test_config_default_model():
    assert InferenceConfig(base_url="http://x").model_id == "DeepSeek-V4-Flash"


def test_adapter_records_model_id():
    adapter = _adapter(lambda payload: _raw())
    result = adapter.interpret("evidence")
    assert result.telemetry.model_id == "DeepSeek-V4-Flash"


def test_usage_missing_is_none():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}]})  # no usage key
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert result.telemetry.prompt_tokens is None
    assert result.telemetry.total_tokens is None


def test_usage_null_does_not_raise():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}], "usage": None})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert result.telemetry.total_tokens is None


def test_usage_list_is_telemetry_invalid():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}], "usage": [1, 2, 3]})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TELEMETRY_INVALID


def test_usage_string_is_telemetry_invalid():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}], "usage": "abc"})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TELEMETRY_INVALID


def test_usage_negative_is_telemetry_invalid():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}],
                                        "usage": {"prompt_tokens": -1, "completion_tokens": 0, "total_tokens": 0}})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TELEMETRY_INVALID


def test_usage_boolean_is_telemetry_invalid():
    adapter = _adapter(lambda payload: {"choices": [{"message": {"content": _content()}}],
                                        "usage": {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 0}})
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.TELEMETRY_INVALID


def test_reported_zero_distinct_from_missing():
    adapter = _adapter(lambda payload: _raw(usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert result.telemetry.prompt_tokens == 0  # reported zero, not None


def test_amount_boolean_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(amount_cents=True)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_amount_numeric_string_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(amount_cents="10000")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_discrepancy_numeric_string_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(discrepancy_cents="0")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_confidence_string_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(confidence="0.9")))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_confidence_overflow_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(confidence=10 ** 400)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_amount_64bit_overflow_rejected():
    adapter = _adapter(lambda payload: _raw(content=_content(amount_cents=2 ** 63)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.INCOMPLETE_RESPONSE


def test_negative_discrepancy_allowed():
    adapter = _adapter(lambda payload: _raw(content=_content(discrepancy_cents=-500)))
    result = adapter.interpret("evidence")
    assert result.outcome == InferenceOutcome.SUCCESS
    assert result.interpretation.discrepancy_cents == -500


def test_input_too_large():
    adapter = _adapter(lambda payload: _raw())
    result = adapter.interpret("x" * 5000)  # exceeds max_input_chars
    assert result.outcome == InferenceOutcome.INPUT_TOO_LARGE
