"""Signal Prism UI tests for the BOUNDARY demonstration interface (B6 S1).

Verifies page rendering, status presentation, the four-stage decision/authority
trace, absence of fabricated pipeline stages, stylesheet delivery, and safe
escaping — all offline via FastAPI TestClient.
"""

from fastapi.testclient import TestClient

from boundary import app as appmod
from boundary.app import create_app
from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.demo import run_demo


def _client(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    return TestClient(create_app(path))


def test_stylesheet_delivered(tmp_path):
    client = _client(tmp_path)
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]
    # Signal Prism palette tokens are present.
    for token in ("--obsidian", "--cobalt", "--cyan", "--emerald", "--amber", "--coral"):
        assert token in r.text


def test_list_shows_status_pills(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "simulated approval committed" in r.text   # emerald (exact match)
    assert "held" in r.text                            # amber (holds)


def test_detail_distinguishes_four_stages(tmp_path):
    client = _client(tmp_path)
    r = client.get("/cases/CASE-1")
    assert r.status_code == 200
    for stage in ("Classifier proposal", "Guard decision",
                  "Simulated journal action", "Independent verification"):
        assert stage in r.text


def test_authority_none_and_simulated_labeled(tmp_path):
    client = _client(tmp_path)
    r = client.get("/cases/CASE-1")
    assert "Authority: NONE at every stage" in r.text
    assert "simulation" in r.text.lower()
    assert "not a payment" in r.text.lower()
    assert "execution_authority = NONE" in r.text


def test_no_fabricated_pipeline_stages(tmp_path):
    # Evolus extraction is still not run by this interface, so its output must not
    # appear. The B3 route (added in S2) is presented only as an OFFLINE SIMULATION,
    # never as a live AMD connection.
    client = _client(tmp_path)
    r = client.get("/cases/CASE-2")
    assert "documentType" not in r.text       # no Evolus extraction output fabricated
    assert "OFFLINE SIMULATION" in r.text     # B3 is labeled offline, not live
    assert "NO LIVE AMD REQUEST" in r.text


def test_money_formatted(tmp_path):
    # CASE-2 has a $5.00 discrepancy (invoice $100.00 vs PO $95.00); render as dollars.
    client = _client(tmp_path)
    r = client.get("/cases/CASE-2")
    assert "$5.00" in r.text


def test_emerald_reserved_for_verification(tmp_path):
    # Emerald marks only the independent verifier's VERIFIED state, never a
    # committed (simulated) record or a guard decision.
    client = _client(tmp_path)
    r = client.get("/")
    assert "pill--verified" not in r.text   # list "committed" is info, not verified
    assert "pill--info" in r.text

    d = client.get("/cases/CASE-1")
    assert d.text.count("trace-step--verified") == 1  # only the verification step
    assert "trace-step--info" in d.text               # guard + committed are info

    d2 = client.get("/cases/CASE-2")
    assert "trace-step--verified" not in d2.text       # held case is never "verified"


def test_identifiers_escaped(tmp_path, monkeypatch):
    evil = "<script>alert(1)</script>"
    inv = Invoice(invoice_id=evil, supplier_id="SUP-1", po_reference="PO-1",
                  issued_date="2026-01-01",
                  line_items=[LineItem(line_number=1, description="x", quantity="1", unit_price="1.00")],
                  stated_total="1.00")
    po = PurchaseOrder(po_id="PO-1", supplier_id="SUP-1", issued_date="2026-01-01",
                       line_items=[LineItem(line_number=1, description="x", quantity="1", unit_price="1.00")],
                       stated_total="1.00")
    case = CaseInput(case_id="EVIL", invoice=inv,
                     po_lookup=POLookupFound(purchase_order=po),
                     supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                     policy_version="v1", input_identity="test")

    monkeypatch.setattr(appmod, "demo_cases", lambda: [("evil", case)])
    client = TestClient(appmod.create_app(str(tmp_path / "x.db")))
    r = client.get("/cases/EVIL")
    assert r.status_code == 200
    assert "&lt;script&gt;" in r.text
    assert "<script>alert" not in r.text
