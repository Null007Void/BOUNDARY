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


def test_homepage_hero_metrics_present(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "66.7%" in r.text
    assert "65.5%" in r.text
    assert "AI-granted execution authority" in r.text
    assert "NONE" in r.text


def test_homepage_hero_principle_and_caveat(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    assert "Use intelligence only when intelligence is required" in r.text
    assert "Use autonomy only when authority exists" in r.text
    # Homepage leads with the S4E.15 two-model comparison, labeled a small sample.
    assert "S4E.15 two-model AMD comparison" in r.text
    assert "sixteen genuine requests" in r.text
    assert "not a universal claim" in r.text
    assert "S4E.5" in r.text  # earlier DeepSeek-only pilot still referenced
    assert 'href="/evidence"' in r.text


def test_homepage_process_strip_present(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    for stage in ("Classify", "Guard", "Journal", "Verify"):
        assert stage in r.text
    # Static CSS-only — no client-side JavaScript is introduced.
    assert "<script" not in r.text


def test_homepage_how_it_works(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    assert "Deterministic routing" in r.text
    assert "Selective advisory inference" in r.text
    assert "Independent verification" in r.text


def test_homepage_hero_accessible(tmp_path):
    client = _client(tmp_path)
    r = client.get("/")
    assert 'aria-label="Pipeline stages"' in r.text
    assert 'class="hero-metrics"' in r.text
    assert 'class="process-strip"' in r.text


def test_homepage_metrics_derive_from_evidence():
    # Homepage metrics now derive from the hash-verified S4E.15 loader, not a hardcoded source.
    from boundary.public_experiment import amd_demonstration
    a = amd_demonstration()
    assert a["request_reduction_percent"] == 66.7
    assert a["token_reduction_percent"] == 65.5


def test_zero_call_indicator_on_deterministic_and_held(tmp_path):
    client = _client(tmp_path)
    # CASE-1 (deterministic) and CASE-3 (missing PO) made zero model calls.
    d1 = client.get("/cases/CASE-1")
    assert "0 MODEL CALLS" in d1.text
    assert "Resolved deterministically" in d1.text
    d3 = client.get("/cases/CASE-3")
    assert "0 MODEL CALLS" in d3.text
    assert "Held on trusted lookup" in d3.text
    # CASE-2 (discrepancy) invokes a model, so no zero-call indicator.
    d2 = client.get("/cases/CASE-2")
    assert "0 MODEL CALLS" not in d2.text


def test_homepage_source_link(tmp_path):
    client = _client(tmp_path)
    t = client.get("/").text
    assert "github.com/Null007Void/BOUNDARY" in t
    assert "MIT licensed" in t


def test_experiment_page_renders(tmp_path):
    client = _client(tmp_path)
    assert client.get("/experiment").status_code == 200


def test_experiment_two_model_comparison(tmp_path):
    import re

    client = _client(tmp_path)
    t = client.get("/experiment").text

    assert "Measured two-model comparison" in t

    rows = set(re.findall(
        r"<tr>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*"
        r"<td>(\d+)</td>\s*<td>(\d+)</td>\s*"
        r"<td>(\d+/\d+)</td>\s*</tr>",
        t,
    ))

    expected = {
        ("DeepSeek-V4-Flash", "LLM_FIRST", "6", "1316", "4/6"),
        ("DeepSeek-V4-Flash", "BOUNDARY", "2", "438", "2/2"),
        ("Gemma-4-31B-IT", "LLM_FIRST", "6", "1614", "5/6"),
        ("Gemma-4-31B-IT", "BOUNDARY", "2", "572", "2/2"),
    }

    assert expected.issubset(rows)
    assert "66.7%" in t
    assert "65.5%" in t
    assert "Eight additional BOUNDARY zero-call cells" in t


def test_experiment_routing_journey(tmp_path):
    client = _client(tmp_path)
    t = client.get("/experiment").text
    assert "DETERMINISTIC" in t
    assert "INFERENCE" in t
    assert "HELD" in t
    assert "0 model calls" in t
    assert "ILLUSTRATIVE" in t  # labeled illustrative, not fabricated event replay


def test_experiment_links_and_readonly(tmp_path):
    client = _client(tmp_path)
    t = client.get("/experiment").text
    assert 'href="/cases/CASE-1"' in t
    assert 'href="/cases/CASE-2"' in t
    assert 'href="/cases/CASE-3"' in t
    assert 'href="/evidence"' in t
    assert client.post("/experiment").status_code == 405


def test_experiment_nav_link_present(tmp_path):
    client = _client(tmp_path)
    for path in ("/", "/evidence"):
        assert 'href="/experiment"' in client.get(path).text


def test_experiment_provenance_limitations(tmp_path):
    client = _client(tmp_path)
    t = client.get("/experiment").text

    assert "S4E.10" in t
    assert "S4E.15" in t
    assert "S4E.17" in t
    assert "720" in t

    assert "simulated" in t.lower()
    assert "schedule-derived" in t
    assert "unverified" in t.lower()
    assert "Gemma later unavailable" in t

    assert "sixteen genuine AMD requests" in t
    assert "six synthetic development cases per model" in t
    assert "Financial approval is simulated" in t
    assert "Execution authority" in t

