"""B5.2 live benchmark readiness tests (offline, mocked HTTP)."""

import json
import types
from dataclasses import asdict

import httpx
import pytest

from boundary import benchmark_live as bl
from boundary.benchmark_live import (
    LIVE_ZERO_REQUEST,
    NOT_CAPTURED,
    OBSERVED_LIVE,
    OBSERVED_OFFLINE,
    SCENARIOS,
    LiveScenario,
    SingleRequestGuard,
    run_scenario,
    token_label,
)
from boundary.amd_transport import build_amd_transport
from boundary.inference import InferenceAdapter, InferenceConfig, InferenceUnavailable
from boundary.router import Route


AMD_URL = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"


def _success_content() -> str:
    return json.dumps({
        "supplier_id": "SUP-9001", "invoice_id": "INV-2048", "po_reference": "PO-77",
        "amount_cents": 125000, "discrepancy_cents": 500, "confidence": 0.6, "notes": "",
    })


def _contradictory_content() -> str:
    # Valid JSON and valid types, but the invoice_id contradicts the source case.
    return json.dumps({
        "supplier_id": "SUP-9001", "invoice_id": "WRONG", "po_reference": "PO-77",
        "amount_cents": 125000, "discrepancy_cents": 500, "confidence": 0.6, "notes": "",
    })


def make_mock_adapter(handler, model_id="DeepSeek-V4-Flash"):
    """Wire the real AMD transport + single-request guard over a mock HTTP layer."""
    config = InferenceConfig(model_id=model_id, api_key="test-key")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    guard = SingleRequestGuard(build_amd_transport(config, client=client), client, AMD_URL)
    adapter = InferenceAdapter(config, guard)
    return config, adapter, guard


def _success_handler(request):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": _success_content()}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


def _disc() -> LiveScenario:
    return SCENARIOS["financial_discrepancy"]


def run_mock(scenario, adapter, guard, model_id="m"):
    return run_scenario(scenario, adapter=adapter, model_id=model_id, outbound=guard)


# --- routing / invocation ---------------------------------------------------------


def test_dry_run_makes_no_outbound_request():
    result = run_scenario(_disc(), adapter=None, model_id=None, outbound=None, dry_run=True)
    assert result.provenance == OBSERVED_OFFLINE
    assert result.dry_run is True
    assert result.transport_kind == bl.TRANSPORT_NONE
    assert result.outbound_requests == 0
    assert result.model_calls == 0
    assert result.route == "HELD"  # no adapter in dry run -> held


def test_exact_match_routes_deterministic_zero_calls():
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(SCENARIOS["exact_match"], adapter, guard)
    assert result.route == "DETERMINISTIC"
    assert result.model_calls == 0
    assert result.outbound_requests == 0
    assert result.harness_safe is True
    assert result.measurement_success is True


def test_missing_po_routes_held_zero_calls():
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(SCENARIOS["missing_po"], adapter, guard)
    assert result.route == "HELD"
    assert result.model_calls == 0
    assert result.outbound_requests == 0
    assert result.harness_safe is True
    assert result.measurement_success is True


def test_financial_discrepancy_makes_exactly_one_call():
    config, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(_disc(), adapter, guard, model_id=config.model_id)
    assert result.route == "INFERENCE"
    assert result.model_calls == 1
    assert result.outbound_requests == 1
    assert result.advisory_consistent is True
    assert result.execution_authority == "NONE"
    assert result.harness_safe is True
    assert result.measurement_success is True


# --- single-request enforcement ---------------------------------------------------


def test_single_request_guard_rejects_second_request():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    guard = SingleRequestGuard(lambda payload: {}, client, AMD_URL)
    guard({})
    with pytest.raises(InferenceUnavailable):
        guard({})
    assert guard.calls == 2


# --- telemetry provenance ---------------------------------------------------------


def test_missing_token_telemetry_is_not_captured():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _success_content()}}],
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.prompt_tokens is None
    assert token_label(result.prompt_tokens) == NOT_CAPTURED


def test_zero_token_telemetry_is_distinct_from_missing():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _success_content()}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.prompt_tokens == 0
    assert token_label(result.prompt_tokens) == "0 (zero consumption)"


def test_positive_token_telemetry_observed():
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.elapsed_ms is not None


# --- failure paths: harness-safe vs measurement-success ----------------------------


def test_auth_failure_is_harness_safe_but_not_measurement_success():
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.route == "INFERENCE"
    assert result.outbound_requests == 1
    assert result.advisory_outcome == "AUTH_FAILURE"
    assert result.advisory_consistent is False
    assert result.execution_authority == "NONE"
    assert result.harness_safe is True
    assert result.measurement_success is False


def test_malformed_is_harness_safe_but_not_measurement_success():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "this is not valid json {"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.advisory_outcome == "MALFORMED_RESPONSE"
    assert result.harness_safe is True
    assert result.measurement_success is False


def test_contradictory_is_harness_safe_but_not_measurement_success():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _contradictory_content()}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.advisory_outcome == "SUCCESS"
    assert result.advisory_consistent is False
    assert result.harness_safe is True
    assert result.measurement_success is False


def test_authority_claim_fails_closed_preserves_authority():
    def handler(request):
        content = json.dumps({"approve": True, "supplier_id": "SUP-9001",
                              "invoice_id": "INV-2048", "po_reference": "PO-77",
                              "amount_cents": 125000, "discrepancy_cents": 500,
                              "confidence": 0.6, "notes": ""})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.advisory_outcome == "AUTHORITY_CLAIM"
    assert result.advisory_consistent is False
    assert result.execution_authority == "NONE"
    assert result.measurement_success is False


# --- provenance: mock must never be labeled live ----------------------------------


def test_mock_transport_is_not_labeled_live():
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.transport_kind == bl.TRANSPORT_OFFLINE
    assert result.provenance == OBSERVED_OFFLINE
    assert result.provenance != OBSERVED_LIVE
    assert result.confirmed_response is False


def test_mock_as_live_counterexample_is_rejected():
    # The exact counterexample from review: a mock that yields a successful,
    # source-consistent response must NOT be representable as observed live inference.
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.outbound_requests == 1
    assert result.measurement_success is True
    # ...but provenance originates at the transport boundary, so it is offline:
    assert result.provenance == OBSERVED_OFFLINE
    assert result.confirmed_response is False
    data = asdict(result)
    assert data["provenance"] == OBSERVED_OFFLINE
    assert data["confirmed_response"] is False


def test_guard_derives_transport_kind_from_client():
    real_client = httpx.Client()
    real_guard = SingleRequestGuard(lambda p: {}, real_client, AMD_URL)
    assert real_guard.transport_kind == bl.TRANSPORT_LIVE

    mock_client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    mock_guard = SingleRequestGuard(lambda p: {}, mock_client, AMD_URL)
    assert mock_guard.transport_kind == bl.TRANSPORT_OFFLINE


def test_wsgi_transport_is_never_live():
    # A WSGI app returns a valid AMD-style HTTP 200 entirely in-process, with no real
    # network request. It must be labeled offline, never OBSERVED_LIVE.
    def wsgi_app(environ, start_response):
        body = json.dumps({
            "choices": [{"message": {"content": _success_content()}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json")])
        return [body]

    config = InferenceConfig(model_id="DeepSeek-V4-Flash", api_key="test-key")
    client = httpx.Client(transport=httpx.WSGITransport(app=wsgi_app))
    guard = SingleRequestGuard(build_amd_transport(config, client=client), client, AMD_URL)
    adapter = InferenceAdapter(config, guard)
    result = run_mock(_disc(), adapter, guard)

    assert result.outbound_requests == 1
    assert result.measurement_success is True   # the in-process response is valid
    # ...but it is NOT live inference:
    assert result.transport_kind == bl.TRANSPORT_OFFLINE
    assert result.provenance == OBSERVED_OFFLINE
    assert result.provenance != OBSERVED_LIVE
    assert result.confirmed_response is False
    data = asdict(result)
    assert data["provenance"] == OBSERVED_OFFLINE
    assert data["confirmed_response"] is False


def test_amd_hostname_mount_is_never_live():
    # A client with the default HTTPTransport but a MockTransport mounted on the AMD
    # hostname serves the request entirely offline. It must not be labeled live.
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _success_content()}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    config = InferenceConfig(model_id="DeepSeek-V4-Flash", api_key="test-key")
    client = httpx.Client(mounts={"https://developer.amd.com.cn": httpx.MockTransport(handler)})
    guard = SingleRequestGuard(build_amd_transport(config, client=client), client, AMD_URL)
    adapter = InferenceAdapter(config, guard)
    result = run_mock(_disc(), adapter, guard)

    assert result.outbound_requests == 1
    assert result.measurement_success is True   # the mounted mock response is valid
    # ...but the AMD request is served offline by the mount, so it is NOT live:
    assert result.transport_kind == bl.TRANSPORT_OFFLINE
    assert result.provenance == OBSERVED_OFFLINE
    assert result.provenance != OBSERVED_LIVE
    assert result.confirmed_response is False
    data = asdict(result)
    assert data["provenance"] == OBSERVED_OFFLINE
    assert data["confirmed_response"] is False


def test_build_live_adapter_produces_live_guard():
    config = InferenceConfig(model_id="m", api_key="k")
    adapter, guard = bl.build_live_adapter(config)
    assert guard.transport_kind == bl.TRANSPORT_LIVE


def test_zero_request_scenario_reports_live_zero_request():
    # A real live transport on a deterministic scenario makes no requests and is
    # labeled LIVE_ZERO_REQUEST, not a real inference observation.
    config = InferenceConfig(model_id="m", api_key="k")
    real_client = httpx.Client()
    adapter, guard = bl._wire_adapter(config, real_client)
    result = run_scenario(SCENARIOS["exact_match"], adapter=adapter, model_id="m", outbound=guard)
    assert result.transport_kind == bl.TRANSPORT_LIVE
    assert result.outbound_requests == 0
    assert result.confirmed_response is False
    assert result.provenance == LIVE_ZERO_REQUEST


# --- telemetry preserved on failed attempts ----------------------------------------


def test_failed_attempt_preserves_token_telemetry():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not valid json {"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    assert result.measurement_success is False
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert result.total_tokens == 10


# --- model identity and serialization ---------------------------------------------


def test_model_identity_captured():
    _, adapter, guard = make_mock_adapter(_success_handler, model_id="DeepSeek-V4-Flash")
    result = run_mock(_disc(), adapter, guard, model_id="DeepSeek-V4-Flash")
    assert result.model_id == "DeepSeek-V4-Flash"


def test_serialized_result_reflects_measurement_success():
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _, adapter, guard = make_mock_adapter(handler)
    result = run_mock(_disc(), adapter, guard)
    data = asdict(result)
    assert data["measurement_success"] is False
    assert data["harness_safe"] is True
    assert data["provenance"] == OBSERVED_OFFLINE
    json.dumps(data)


# --- live runner refusal without credentials ---------------------------------------


def test_live_run_refuses_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("BOUNDARY_AMD_API_KEY", raising=False)
    rc = bl._live_run(_disc(), types.SimpleNamespace(json=None))
    assert rc == 1
    assert "refusing" in capsys.readouterr().out


# --- CLI exit codes (mocked HTTP through the real CLI path) ------------------------


def _install_mock_live_adapter(monkeypatch, handler):
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", "test-key")

    def fake_build_live_adapter(config):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        guard = SingleRequestGuard(build_amd_transport(config, client=client), client, AMD_URL)
        return InferenceAdapter(config, guard), guard

    monkeypatch.setattr(bl, "build_live_adapter", fake_build_live_adapter)


def test_cli_exit_zero_on_successful_inference(monkeypatch):
    _install_mock_live_adapter(monkeypatch, _success_handler)
    assert bl.main(["--live", "--scenario", "financial_discrepancy"]) == 0


def test_cli_exit_nonzero_on_auth_failure(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _install_mock_live_adapter(monkeypatch, handler)
    assert bl.main(["--live", "--scenario", "financial_discrepancy"]) == 1


def test_cli_exit_nonzero_on_malformed(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not valid json {"}}],
        })

    _install_mock_live_adapter(monkeypatch, handler)
    assert bl.main(["--live", "--scenario", "financial_discrepancy"]) == 1


def test_cli_exit_nonzero_on_contradictory(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _contradictory_content()}}],
        })

    _install_mock_live_adapter(monkeypatch, handler)
    assert bl.main(["--live", "--scenario", "financial_discrepancy"]) == 1


def test_cli_dry_run_is_successful_preflight(monkeypatch):
    monkeypatch.delenv("BOUNDARY_AMD_API_KEY", raising=False)
    assert bl.main(["--scenario", "financial_discrepancy"]) == 0


# --- scenario expectations ---------------------------------------------------------


def test_all_scenarios_present_and_expected():
    assert set(SCENARIOS) == {"exact_match", "financial_discrepancy", "missing_po"}
    assert SCENARIOS["exact_match"].expected_route == Route.DETERMINISTIC
    assert SCENARIOS["financial_discrepancy"].expected_route == Route.INFERENCE
    assert SCENARIOS["missing_po"].expected_route == Route.HELD
    assert SCENARIOS["financial_discrepancy"].requires_successful_inference is True
    assert SCENARIOS["exact_match"].requires_successful_inference is False


def test_scenario_that_skips_intended_route_does_not_pass():
    from boundary.contracts import POLookupFound

    scenario = LiveScenario(
        name="financial_discrepancy",
        case=bl._case(
            "CASE-MISMATCH",
            bl._invoice("INV-2048", "SUP-9001", "PO-77", [bl._line(1, "1", "1250.00")], "1250.00"),
            POLookupFound(purchase_order=bl._po("PO-77", "SUP-OTHER", [bl._line(1, "1", "1245.00")], "1245.00")),
        ),
        evidence_text=bl.SYNTHETIC_EVIDENCE,
        expected_route=Route.INFERENCE,
        expected_outbound=1,
        requires_successful_inference=True,
    )
    _, adapter, guard = make_mock_adapter(_success_handler)
    result = run_mock(scenario, adapter, guard)
    assert result.route == "HELD"
    assert result.outbound_requests == 0
    assert result.route_correct is False
    assert result.outbound_correct is False
    assert result.harness_safe is False
    assert result.measurement_success is False


def test_transport_makes_exactly_one_http_request():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _success_content()}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    config = InferenceConfig(model_id="m", api_key="k")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = build_amd_transport(config, client=client)
    transport({"model": "m", "messages": []})
    assert len(paths) == 1
