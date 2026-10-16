"""Phase B — protected live AMD demo: negative security tests.

Proves disabled-by-default, operator authentication, single-flight, one-request, no
fabrication on provider error, authority NONE, and no secret disclosure — all offline
(via a mocked AMD transport; no real provider call is ever made).
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from boundary import live_demo as ld
from boundary.app import create_app
from boundary.inference import InferenceUnavailable

SECRET = "test-operator-secret"


def _enable(monkeypatch, secret=SECRET, api_key="test-amd-key", enabled="1"):
    monkeypatch.setenv(ld.ENV_ENABLED, enabled)
    monkeypatch.setenv(ld.ENV_OPERATOR_SECRET, secret)
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", api_key)


def _interpretation_json():
    case = ld.build_inv500_case()
    inv = case.invoice
    po = case.po_lookup.purchase_order
    return {
        "supplier_id": inv.supplier_id,
        "invoice_id": inv.invoice_id,
        "po_reference": po.po_id,
        "amount_cents": inv.computed_total_cents,
        "discrepancy_cents": inv.computed_total_cents - po.computed_total_cents,
        "confidence": 0.9,
        "notes": "",
    }


def _mock_amd_transport(monkeypatch, content=None, exc=None):
    calls = {"n": 0}

    def transport(payload):
        calls["n"] += 1
        if exc is not None:
            raise exc
        body = content if content is not None else json.dumps(_interpretation_json())
        return {"model": "self-dploy/DeepSeek-V4-Flash",
                "choices": [{"message": {"content": body}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}

    monkeypatch.setattr(ld, "build_amd_transport", lambda config: transport)
    return calls


# --- disabled / fail-closed ----------------------------------------------- #

def test_disabled_by_default(monkeypatch):
    # No env vars set: disabled, and the endpoint is non-operational.
    assert ld.is_enabled() is False
    assert ld.authenticate(SECRET) is False


def test_enabled_requires_all_three(monkeypatch):
    monkeypatch.setenv(ld.ENV_ENABLED, "1")
    monkeypatch.setenv(ld.ENV_OPERATOR_SECRET, SECRET)
    monkeypatch.delenv("BOUNDARY_AMD_API_KEY", raising=False)
    assert ld.is_enabled() is False  # missing AMD credential -> fail closed


def test_endpoint_disabled_returns_404(monkeypatch, tmp_path):
    client = TestClient(create_app(str(tmp_path / "b.db")))
    assert client.post("/live-demo", headers={"x-boundary-operator-token": SECRET}).status_code == 404


# --- authentication ------------------------------------------------------- #

def test_incorrect_authentication_rejected(monkeypatch, tmp_path):
    _enable(monkeypatch)
    client = TestClient(create_app(str(tmp_path / "b.db")))
    resp = client.post("/live-demo", headers={"x-boundary-operator-token": "wrong"})
    assert resp.status_code == 401


def test_token_length_capped(monkeypatch):
    _enable(monkeypatch)
    assert ld.authenticate("x" * (ld.MAX_OPERATOR_TOKEN_LEN + 1)) is False
    assert ld.authenticate(SECRET) is True


# --- one request, real result, authority NONE ----------------------------- #

def test_one_authorized_request_returns_real_metadata(monkeypatch):
    _enable(monkeypatch)
    calls = _mock_amd_transport(monkeypatch)
    result = ld.run_live_demo()
    assert calls["n"] == 1  # exactly one AMD request
    assert result["execution_authority"] == "NONE"
    assert result["case_id"]
    assert result["model_calls"] == 1
    # Real telemetry present only because a real response was received.
    assert result["receipt"] == "received"
    assert result["endpoint_model_id"] is not None
    assert result["total_tokens"] is not None
    assert result["elapsed_ms"] is not None


def test_no_fabrication_on_provider_error(monkeypatch):
    _enable(monkeypatch)
    _mock_amd_transport(monkeypatch, exc=InferenceUnavailable(
        "server error", diagnostics={"category": "server_error", "http_status": 500}))
    result = ld.run_live_demo()
    # A provider failure must never be reported as a successful interpretation.
    assert result["advisory_outcome"] != "SUCCESS"
    assert result["advisory_consistent"] is False
    assert result["execution_authority"] == "NONE"


def test_single_flight_rejects_duplicate(monkeypatch):
    _enable(monkeypatch)
    # Acquire the execution lock to simulate an in-flight run.
    assert ld._execution_lock.acquire(blocking=False)
    try:
        with pytest.raises(ld.LiveDemoBusy):
            ld.run_live_demo()
    finally:
        ld._execution_lock.release()


# --- no secret disclosure / no journal commit ----------------------------- #

def test_result_does_not_disclose_secrets(monkeypatch):
    _enable(monkeypatch, secret="super-secret-value", api_key="sk-amd-real-key")
    _mock_amd_transport(monkeypatch)
    result = ld.run_live_demo()
    blob = json.dumps(result)
    assert "super-secret-value" not in blob
    assert "sk-amd-real-key" not in blob


def test_run_does_not_write_journal(tmp_path, monkeypatch):
    _enable(monkeypatch)
    _mock_amd_transport(monkeypatch)
    db = str(tmp_path / "boundary.db")
    ld.run_live_demo()
    assert not os.path.exists(db)
    assert not os.path.exists(db + ".batch.db")


def test_outbound_payload_contains_invoice_and_po_facts(monkeypatch):
    # Regression: the model-facing evidence must include BOTH the invoice amount and the
    # trusted purchase-order amount, so the model can calculate/validate the discrepancy.
    _enable(monkeypatch)
    captured = {}

    def transport(payload):
        captured["payload"] = payload
        return {"model": "self-dploy/DeepSeek-V4-Flash",
                "choices": [{"message": {"content": json.dumps(_interpretation_json())}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}

    monkeypatch.setattr(ld, "build_amd_transport", lambda config: transport)
    ld.run_live_demo()

    evidence = captured["payload"]["messages"][1]["content"]
    assert "INV-500" in evidence
    assert "$2729.00" in evidence      # invoice stated total
    assert "PO-500" in evidence
    assert "$2447.23" in evidence     # trusted purchase-order total


def test_successful_http_post_through_live_demo(monkeypatch, tmp_path):
    # Regression: a mocked successful run through the actual POST /live-demo endpoint,
    # not just a direct run_live_demo() invocation.
    _enable(monkeypatch)
    _mock_amd_transport(monkeypatch)
    client = TestClient(create_app(str(tmp_path / "b.db")))
    resp = client.post("/live-demo", headers={"x-boundary-operator-token": SECRET})
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_authority"] == "NONE"
    assert data["model_calls"] == 1
    assert data["advisory_outcome"] == "SUCCESS"
    assert data["endpoint_model_id"] is not None
    assert data["total_tokens"] is not None
