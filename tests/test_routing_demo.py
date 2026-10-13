"""B6 S2 routing-demonstration tests (offline).

Verifies the B3 routing demonstration uses the accepted router with an offline
injected adapter bound to the ACTUAL rendered case (not a cached copy), that
routing outcomes match the real pipeline, that the authority-claim rejection is
exercised only when the validator actually produced it, and that timestamps are
formatted truthfully.
"""

import inspect

from fastapi.testclient import TestClient

from boundary import routing_demo
from boundary.app import create_app, pretty_timestamp
from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.demo import demo_cases, run_demo
from boundary.routing_demo import authority_claim_rejection, routing_for_case


def _demo_cases():
    return {case.case_id: case for _, case in demo_cases()}


def _exact_match_case2():
    """A CASE-2 identifier whose facts are an exact $95 match (zero discrepancy)."""
    inv = Invoice(invoice_id="INV-200", supplier_id="SUP-200", po_reference="PO-200",
                  issued_date="2026-01-01",
                  line_items=[LineItem(line_number=1, description="x", quantity="1", unit_price="95.00")],
                  stated_total="95.00")
    po = PurchaseOrder(po_id="PO-200", supplier_id="SUP-200", issued_date="2026-01-01",
                       line_items=[LineItem(line_number=1, description="x", quantity="1", unit_price="95.00")],
                       stated_total="95.00")
    return CaseInput(case_id="CASE-2", invoice=inv,
                     po_lookup=POLookupFound(purchase_order=po),
                     supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                     policy_version="v1", input_identity="test")


# --- unit: routing outcomes (bound to the actual case) ----------------------------


def test_case1_deterministic_zero_invocations():
    r = routing_for_case(_demo_cases()["CASE-1"])
    assert r["route"] == "DETERMINISTIC"
    assert r["model_calls"] == 0
    assert r["advisory_outcome"] is None
    assert r["execution_authority"] == "NONE"


def test_case2_inference_consistent_advisory():
    r = routing_for_case(_demo_cases()["CASE-2"])
    assert r["route"] == "INFERENCE"
    assert r["model_calls"] == 1
    assert r["advisory_outcome"] == "SUCCESS"
    assert r["advisory_consistent"] is True
    assert r["execution_authority"] == "NONE"
    assert r["interpretation"]["amount_cents"] == 10000
    assert r["interpretation"]["discrepancy_cents"] == 500


def test_case3_held_without_inference():
    r = routing_for_case(_demo_cases()["CASE-3"])
    assert r["route"] == "HELD"
    assert r["model_calls"] == 0
    assert r["advisory_outcome"] is None
    assert r["execution_authority"] == "NONE"


def test_authority_claim_rejected():
    r = authority_claim_rejection(_demo_cases()["CASE-2"])
    assert r is not None
    assert r["advisory_outcome"] == "AUTHORITY_CLAIM"
    assert r["advisory_consistent"] is False
    assert r["execution_authority"] == "NONE"
    assert r["interpretation"] is None


def test_routing_is_advisory_only():
    cases = _demo_cases()
    for cid in ("CASE-1", "CASE-2", "CASE-3"):
        assert routing_for_case(cases[cid])["execution_authority"] == "NONE"
    assert authority_claim_rejection(cases["CASE-2"])["execution_authority"] == "NONE"


def test_no_live_transport_in_routing_demo():
    src = inspect.getsource(routing_demo)
    assert "build_amd_transport" not in src
    assert "httpx" not in src


# --- adversarial: modified same-ID case facts ------------------------------------


def test_modified_case2_exact_match_routes_deterministic():
    # If CASE-2's facts change to an exact match, B3 must reflect the new facts,
    # not a cached discrepancy/INFERENCE result.
    case = _exact_match_case2()
    r = routing_for_case(case)
    assert r["route"] == "DETERMINISTIC"
    assert r["model_calls"] == 0
    assert r["advisory_outcome"] is None
    assert authority_claim_rejection(case) is None


def test_authority_claim_only_when_validator_rejects():
    cases = _demo_cases()
    assert authority_claim_rejection(cases["CASE-2"]) is not None
    assert authority_claim_rejection(cases["CASE-1"]) is None
    assert authority_claim_rejection(cases["CASE-3"]) is None


# --- timestamp formatting ---------------------------------------------------------


def test_pretty_timestamp_converts_to_utc():
    assert pretty_timestamp("2026-10-13T10:00:00+05:00") == "2026-10-13 05:00:00 UTC"
    assert pretty_timestamp("2026-10-13T10:00:00+00:00") == "2026-10-13 10:00:00 UTC"


def test_pretty_timestamp_naive_not_labeled_utc():
    assert pretty_timestamp("2026-10-13T10:00:00") == "2026-10-13 10:00:00"
    assert "UTC" not in pretty_timestamp("2026-10-13T10:00:00")


def test_pretty_timestamp_unparseable_preserved():
    assert pretty_timestamp("not-a-date") == "not-a-date"
    assert pretty_timestamp(None) == "\u2014"


# --- UI ----------------------------------------------------------------------------


def _client(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    return TestClient(create_app(path))


def test_b3_section_present_on_all_cases(tmp_path):
    client = _client(tmp_path)
    for cid in ("CASE-1", "CASE-2", "CASE-3"):
        t = client.get(f"/cases/{cid}").text
        assert "Selective Intelligence" in t


def test_deterministic_case_labels_no_model_response(tmp_path):
    client = _client(tmp_path)
    t = client.get("/cases/CASE-1").text
    assert "No simulated model response generated" in t
    assert "OFFLINE SIMULATION" not in t  # no model was invoked for this path


def test_case2_separates_b1_hold_from_advisory(tmp_path):
    client = _client(tmp_path)
    t = client.get("/cases/CASE-2").text
    assert "$5.00" in t
    assert "HOLD" in t
    assert "INFERENCE" in t
    assert "SUCCESS" in t
    assert "Advisory only" in t
    assert "OFFLINE SIMULATION" in t  # a model WAS invoked here


def test_authority_claim_rejection_visible(tmp_path):
    client = _client(tmp_path)
    t = client.get("/cases/CASE-2").text
    assert "AUTHORITY_CLAIM" in t
    assert "Rejected" in t


def test_readable_supplier_state_not_enum_repr(tmp_path):
    client = _client(tmp_path)
    t = client.get("/cases/CASE-1").text
    assert "MATCH" in t
    assert "SupplierMatchState" not in t


def test_reload_never_mutates_journal(tmp_path):
    from boundary.ledger import ActionJournal
    path = str(tmp_path / "demo.db")
    run_demo(path)
    j = ActionJournal(path)
    before = j.count()
    j.close()

    client = TestClient(create_app(path))
    for cid in ("CASE-1", "CASE-2", "CASE-3"):
        client.get(f"/cases/{cid}")

    j = ActionJournal(path)
    after = j.count()
    j.close()
    assert after == before
