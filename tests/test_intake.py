"""S4E.25 — interactive synthetic invoice intake tests.

Verifies the real vertical slice: the judge's fields are evaluated through the accepted
deterministic pipeline (classify -> guard -> routing), with honest inference status,
authority NONE, no persistence, no provider calls, and no weakening of read-only routes.
"""

import os

import pytest
from fastapi.testclient import TestClient

from boundary.app import create_app
from boundary.intake import (
    IntakeError,
    build_case,
    evaluate_intake,
)

EXACT = ("INV-700", "SUP-SYN-100", "PO-SYN-100", "1000.00")
DISCREPANCY = ("INV-701", "SUP-SYN-100", "PO-SYN-100", "1100.00")
MISSING_PO = ("INV-702", "SUP-SYN-100", "PO-MISSING", "1000.00")


def _eval(*fields):
    return evaluate_intake(build_case(*fields))


# --- unit: the three scenarios ------------------------------------------- #

def test_exact_match_approves_deterministically():
    result = _eval(*EXACT)
    assert result["guard"].action.value == "APPROVE"
    assert result["route"] == "DETERMINISTIC"
    assert result["inference"]["required"] is False
    assert result["execution_authority"] == "NONE"


def test_discrepancy_holds_and_requires_inference():
    result = _eval(*DISCREPANCY)
    assert result["guard"].action.value == "HOLD"
    assert result["route"] == "INFERENCE"
    assert result["inference"]["required"] is True
    assert "UNAVAILABLE" in result["inference"]["note"].upper()
    # No model response is fabricated — the note must not claim a fresh inference ran.
    assert "historical" not in result["inference"]["note"].lower()


def test_missing_po_holds_without_commit():
    result = _eval(*MISSING_PO)
    assert result["guard"].action.value == "HOLD"
    assert result["route"] == "HELD"
    assert result["inference"]["required"] is False
    # Missing PO evidence never produces a financial commit.
    assert result["proposal"].exact_match_eligible is False


# --- unit: editing changes the computed result --------------------------- #

def test_editing_amount_changes_result():
    exact = _eval("INV-700", "SUP-SYN-100", "PO-SYN-100", "1000.00")
    edited = _eval("INV-700", "SUP-SYN-100", "PO-SYN-100", "1100.00")
    assert exact["guard"].action.value == "APPROVE"
    assert edited["guard"].action.value == "HOLD"
    assert edited["route"] == "INFERENCE"


def test_editing_po_reference_changes_result():
    found = _eval("INV-700", "SUP-SYN-100", "PO-SYN-100", "1000.00")
    missing = _eval("INV-700", "SUP-SYN-100", "PO-MISSING", "1000.00")
    assert found["guard"].action.value == "APPROVE"
    assert missing["guard"].action.value == "HOLD"
    assert missing["route"] == "HELD"


# --- unit: invalid / oversized / unsupported inputs fail safely ---------- #

def test_invalid_amount_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", "not-a-number")


def test_negative_amount_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", "-5.00")


def test_too_many_decimals_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", "1000.001")


def test_oversized_amount_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", "1000000001.00")


def test_oversized_field_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "X" * 101, "PO-SYN-100", "1000.00")


def test_blank_field_rejected():
    with pytest.raises(IntakeError):
        build_case("", "SUP-SYN-100", "PO-SYN-100", "1000.00")


# --- integration: the web endpoint --------------------------------------- #

@pytest.fixture
def client(tmp_path):
    db = str(tmp_path / "boundary.db")
    return TestClient(create_app(db))


def test_try_route_renders(client):
    resp = client.get("/try")
    assert resp.status_code == 200
    assert "Evaluate Invoice" in resp.text


def test_scenario_prefill(client):
    resp = client.get("/try", params={"scenario": "discrepancy"})
    assert resp.status_code == 200
    assert "1100.00" in resp.text


def test_post_evaluate_returns_result(client):
    resp = client.post("/evaluate", data={
        "invoice_id": "INV-700", "supplier_id": "SUP-SYN-100",
        "po_reference": "PO-SYN-100", "amount": "1000.00",
    })
    assert resp.status_code == 200
    assert "APPROVE" in resp.text
    assert "execution_authority" in resp.text
    assert "NONE" in resp.text


def test_post_evaluate_validation_error(client):
    resp = client.post("/evaluate", data={
        "invoice_id": "INV-700", "supplier_id": "SUP-SYN-100",
        "po_reference": "PO-SYN-100", "amount": "abc",
    })
    assert resp.status_code == 400
    assert "Validation error" in resp.text


def test_post_evaluate_does_not_write_journal(tmp_path):
    db = str(tmp_path / "boundary.db")
    client = TestClient(create_app(db))
    client.post("/evaluate", data={
        "invoice_id": "INV-700", "supplier_id": "SUP-SYN-100",
        "po_reference": "PO-SYN-100", "amount": "1000.00",
    })
    # No journal (boundary.db) or batch journal is created by the intake endpoint.
    assert not os.path.exists(db)
    assert not os.path.exists(db + ".batch.db")


def test_post_evaluate_requires_no_credentials(client, monkeypatch):
    for var in ("BOUNDARY_AMD_API_KEY", "BOUNDARY_EVOLUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    resp = client.post("/evaluate", data={
        "invoice_id": "INV-700", "supplier_id": "SUP-SYN-100",
        "po_reference": "PO-SYN-100", "amount": "1000.00",
    })
    assert resp.status_code == 200


def test_other_routes_stay_read_only(client):
    # The permitted POST /evaluate must not weaken unrelated read-only routes.
    for path in ("/batch", "/comparison", "/evidence", "/cases/CASE-1", "/"):
        assert client.post(path).status_code in (405, 404)


# --- finding 2: monetary text precision ----------------------------------- #

def test_scientific_notation_rejected():
    for raw in ("1E+3", "1e3", "1.0e2"):
        with pytest.raises(IntakeError):
            build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", raw)


def test_three_decimal_places_rejected():
    with pytest.raises(IntakeError):
        build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", "1000.000")


def test_signed_amount_rejected():
    for raw in ("-5.00", "+5.00"):
        with pytest.raises(IntakeError):
            build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", raw)


def test_leading_or_trailing_dot_rejected():
    for raw in (".50", "1000."):
        with pytest.raises(IntakeError):
            build_case("INV-1", "SUP-SYN-100", "PO-SYN-100", raw)


# --- finding 1: request body enforcement ---------------------------------- #

def test_oversized_body_rejected(client):
    body = "amount=" + ("9" * 9000)
    resp = client.post("/evaluate", content=body,
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status_code == 413


def test_unsupported_content_type_rejected(client):
    resp = client.post("/evaluate", json={"amount": "1000.00"})
    assert resp.status_code == 415


def test_malformed_form_rejected(client):
    resp = client.post("/evaluate", content="no_equals_sign",
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status_code == 400


def test_excessive_fields_rejected(client):
    body = "&".join(f"f{i}=v{i}" for i in range(9))  # 9 fields > MAX_FORM_FIELDS
    resp = client.post("/evaluate", content=body,
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status_code == 400


def _asgi_post(app, headers, body_bytes):
    """Invoke the ASGI app directly with a hand-built scope, returning (status, body)."""
    import anyio
    result = {}

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
        elif message["type"] == "http.response.body":
            result.setdefault("body", bytearray()).extend(message.get("body", b""))

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/evaluate",
        "raw_path": b"/evaluate", "query_string": b"", "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": ("test", 123), "server": ("test", 80),
    }
    anyio.run(app, scope, receive, send)
    return result.get("status"), bytes(result.get("body", b""))


def test_misleading_content_length_rejected(tmp_path):
    # A body larger than 8KB with a misleading small declared Content-Length must still be
    # rejected, because the ACTUAL streamed body is bounded, not just the header.
    app = create_app(str(tmp_path / "boundary.db"))
    big_body = ("invoice_id=INV&supplier_id=S&po_reference=P&amount=" + "9" * 9000).encode()
    status, _ = _asgi_post(app, [("content-type", "application/x-www-form-urlencoded"),
                                 ("content-length", "30")], big_body)
    assert status == 413


def test_missing_content_length_rejected(tmp_path):
    # No Content-Length at all: the streamed body is still capped at 8KB.
    app = create_app(str(tmp_path / "boundary.db"))
    big_body = ("invoice_id=INV&supplier_id=S&po_reference=P&amount=" + "9" * 9000).encode()
    status, _ = _asgi_post(app, [("content-type", "application/x-www-form-urlencoded")],
                            big_body)
    assert status == 413
