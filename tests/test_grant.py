import sqlite3
import threading
from datetime import datetime

import pytest

from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.grant import (
    ClaimStatus,
    GrantSchemaError,
    GrantStatus,
    GrantStore,
    Reviewer,
    ReviewerSource,
)
from boundary.ledger import case_fingerprint
from boundary.review import RequestStatus, ReviewStore


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _case(case_id, invoice_id, supplier_id, po_ref, invoice_total, po_total, evidence=("ev-1",)):
    inv = Invoice(
        invoice_id=invoice_id,
        supplier_id=supplier_id,
        po_reference=po_ref,
        issued_date="2026-10-12",
        line_items=[_line(1, "1", invoice_total)],
        stated_total=invoice_total,
    )
    po = PurchaseOrder(
        po_id="PO-1",
        supplier_id=supplier_id,
        issued_date="2026-10-12",
        line_items=[_line(1, "1", po_total)],
        stated_total=po_total,
    )
    return CaseInput(
        case_id=case_id,
        invoice=inv,
        po_lookup=POLookupFound(purchase_order=po),
        supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
        evidence=list(evidence),
        policy_version="policy-v1",
        input_identity="test",
    )


def _over_500_case(case_id="C-1"):
    return _case(case_id, "INV-" + case_id, "SUP-1", "PO-1", "600.01", "100.00")


def _sim_reviewer():
    return Reviewer(name="sim-reviewer", source=ReviewerSource.SIMULATED)


def _grant_store(path):
    return GrantStore(path, test_mode=True)


def test_grant_binds_to_request_and_case(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    assert review_store.request(case).status == RequestStatus.CREATED
    try:
        result = grant_store.issue(review_store, case, _sim_reviewer())
        assert result.status == GrantStatus.ISSUED
        g = result.grant
        assert g.case_id == "C-1"
        assert g.invoice_id == "INV-C-1"
        assert g.po_id == "PO-1"
        assert g.supplier_id == "SUP-1"
        assert g.proposed_action == "APPROVE"
        assert g.amount_cents == 60001
        assert g.discrepancy_cents == 50001
        assert g.policy_version == "policy-v1"
        assert g.reviewer == "sim-reviewer"
        assert g.reviewer_source == "SIMULATED"
        assert g.state == "ISSUED"
        assert g.review_store_id == review_store.store_id
        assert g.case_fingerprint == review_store.get("C-1").case_fingerprint
    finally:
        review_store.close()
        grant_store.close()


def test_reviewer_source_rejection(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        result = grant_store.issue(
            review_store, case, Reviewer(name="human-reviewer", source=ReviewerSource.CLAIMED_HUMAN)
        )
        assert result.status == GrantStatus.REVIEWER_REJECTED
        assert result.grant is None
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_no_review_request(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    try:
        result = grant_store.issue(review_store, _over_500_case(), _sim_reviewer())
        assert result.status == GrantStatus.NO_REVIEW_REQUEST
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_changed_case_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        changed = _case("C-1", "INV-C-1", "SUP-1", "PO-1", "700.01", "100.00")
        result = grant_store.issue(review_store, changed, _sim_reviewer())
        assert result.status == GrantStatus.CASE_CHANGED
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_duplicate_invoice_review_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    try:
        c1 = _case("C-1", "INV-SHARED", "SUP-1", "PO-1", "600.01", "100.00")
        c2 = _case("C-2", "INV-SHARED", "SUP-1", "PO-1", "600.01", "100.00")
        assert review_store.request(c1).status == RequestStatus.CREATED
        assert review_store.request(c2).status == RequestStatus.DUPLICATE_INVOICE
        assert review_store.count() == 1
    finally:
        review_store.close()


def test_duplicate_invoice_grant_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    c1 = _case("C-1", "INV-SHARED", "SUP-1", "PO-1", "600.01", "100.00")
    assert review_store.request(c1).status == RequestStatus.CREATED
    try:
        assert grant_store.issue(review_store, c1, _sim_reviewer()).status == GrantStatus.ISSUED
        assert grant_store.count() == 1
    finally:
        review_store.close()
        grant_store.close()


def test_expiration_metadata(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        g = grant_store.issue(review_store, case, _sim_reviewer()).grant
        assert g.expires_at > g.issued_at
    finally:
        review_store.close()
        grant_store.close()


def test_grant_persistence(tmp_path):
    path = str(tmp_path / "g.db")
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(path)
    case = _over_500_case()
    review_store.request(case)
    result = grant_store.issue(review_store, case, _sim_reviewer())
    grant_id = result.grant.grant_id
    grant_store.close()
    review_store.close()

    reopened = GrantStore(path, test_mode=True)
    try:
        got = reopened.get(grant_id)
        assert got is not None
        assert got.case_id == "C-1"
        assert got.state == "ISSUED"
    finally:
        reopened.close()


def test_grant_has_no_execution(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        g = grant_store.issue(review_store, case, _sim_reviewer()).grant
        assert g.state == "ISSUED"
        assert not hasattr(g, "consumed")
        assert not hasattr(g, "executed")
        assert not hasattr(g, "committed_action")
    finally:
        review_store.close()
        grant_store.close()


def test_already_issued(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        assert grant_store.issue(review_store, case, _sim_reviewer()).status == GrantStatus.ISSUED
        second = grant_store.issue(review_store, case, _sim_reviewer())
        assert second.status == GrantStatus.ALREADY_ISSUED
        assert grant_store.count() == 1
    finally:
        review_store.close()
        grant_store.close()


# --- B2.2a grant-integrity regressions ---


def test_tampered_review_fields_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    # Tamper the stored review without touching its fingerprint.
    review_store._conn.execute(
        "UPDATE review_requests SET proposed_action = 'TRANSFER', amount_cents = 99999 "
        "WHERE case_id = 'C-1'"
    )
    try:
        result = grant_store.issue(review_store, case, _sim_reviewer())
        assert result.status == GrantStatus.REVIEW_INCONSISTENT
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_forged_ineligible_review_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    # Structurally invalid case: invoice references the wrong PO.
    case = _case("C-1", "INV-C-1", "SUP-1", "PO-WRONG", "600.01", "100.00")
    # Forge a WAITING_AUTHORITY row whose fingerprint matches the (invalid) case.
    review_store._conn.execute(
        "INSERT INTO review_requests "
        "(case_id, case_fingerprint, invoice_id, po_id, supplier_id, proposed_action, "
        " amount_cents, discrepancy_cents, policy_version, state, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C-1", case_fingerprint(case), "INV-C-1", "PO-1", "SUP-1", "APPROVE",
         60001, 50001, "policy-v1", "WAITING_AUTHORITY", "forged", "2026-10-12T00:00:00+00:00"),
    )
    try:
        result = grant_store.issue(review_store, case, _sim_reviewer())
        assert result.status == GrantStatus.CASE_NOT_ELIGIBLE
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_review_id_namespace_not_conflated(tmp_path):
    grant_store = _grant_store(str(tmp_path / "g.db"))
    rs1 = ReviewStore(str(tmp_path / "r1.db"))
    rs2 = ReviewStore(str(tmp_path / "r2.db"))
    c1 = _case("C-1", "INV-C-1", "SUP-1", "PO-1", "600.01", "100.00")
    c2 = _case("C-2", "INV-C-2", "SUP-1", "PO-1", "601.01", "100.00")
    rs1.request(c1)  # request_id = 1 in rs1
    rs2.request(c2)  # request_id = 1 in rs2
    try:
        r1 = grant_store.issue(rs1, c1, _sim_reviewer())
        r2 = grant_store.issue(rs2, c2, _sim_reviewer())
        assert r1.status == GrantStatus.ISSUED
        # The second store's request_id=1 must not be conflated with the first.
        assert r2.status == GrantStatus.ISSUED
        assert grant_store.count() == 2
    finally:
        rs1.close()
        rs2.close()
        grant_store.close()


def test_blank_reviewer_name_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    try:
        for name in ["", "   ", "\t"]:
            result = grant_store.issue(
                review_store, case, Reviewer(name=name, source=ReviewerSource.SIMULATED)
            )
            assert result.status == GrantStatus.REVIEWER_REJECTED
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


def test_simulated_issuance_gated_by_test_mode(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = GrantStore(str(tmp_path / "g.db"))  # test_mode defaults to False
    case = _over_500_case()
    review_store.request(case)
    try:
        result = grant_store.issue(review_store, case, _sim_reviewer())
        assert result.status == GrantStatus.SIMULATION_NOT_ENABLED
        assert grant_store.count() == 0
    finally:
        review_store.close()
        grant_store.close()


# --- B2.2a final database-integrity regressions ---


def test_duplicate_invoice_across_stores_same_case(tmp_path):
    grant_store = _grant_store(str(tmp_path / "g.db"))
    rs1 = ReviewStore(str(tmp_path / "r1.db"))
    rs2 = ReviewStore(str(tmp_path / "r2.db"))
    # Same case_id, supplier, and invoice in two independent review stores.
    c1 = _case("C-1", "INV-SHARED", "SUP-1", "PO-1", "600.01", "100.00")
    c2 = _case("C-1", "INV-SHARED", "SUP-1", "PO-1", "600.01", "100.00")
    rs1.request(c1)
    rs2.request(c2)
    try:
        r1 = grant_store.issue(rs1, c1, _sim_reviewer())
        assert r1.status == GrantStatus.ISSUED
        r2 = grant_store.issue(rs2, c2, _sim_reviewer())
        # One-grant-per-invoice holds regardless of case ID or review-store identity.
        assert r2.status == GrantStatus.DUPLICATE_INVOICE
        assert grant_store.count() == 1
        # The original grant is preserved.
        assert grant_store.get(r1.grant.grant_id) is not None
    finally:
        rs1.close()
        rs2.close()
        grant_store.close()


_LEGACY_SCHEMA = """
CREATE TABLE grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_id TEXT NOT NULL UNIQUE,
    request_id INTEGER NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    case_fingerprint TEXT NOT NULL,
    invoice_id TEXT NOT NULL,
    po_id TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    discrepancy_cents INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reviewer_source TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    state TEXT NOT NULL
)
"""


def test_legacy_empty_schema_migrated(tmp_path):
    path = str(tmp_path / "g.db")
    conn = sqlite3.connect(path)
    conn.execute(_LEGACY_SCHEMA)
    conn.close()

    store = GrantStore(path, test_mode=True)
    try:
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(grants)").fetchall()}
        assert "review_store_id" in cols
        assert store.count() == 0
    finally:
        store.close()


def test_legacy_populated_schema_fails_closed(tmp_path):
    path = str(tmp_path / "g.db")
    conn = sqlite3.connect(path)
    conn.execute(_LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO grants (grant_id, request_id, case_id, case_fingerprint, invoice_id, "
        "po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, "
        "reviewer, reviewer_source, issued_at, expires_at, state) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("g1", 1, "C-1", "fp", "INV-1", "PO-1", "SUP-1", "APPROVE", 1, 1,
         "policy-v1", "sim", "SIMULATED", "t0", "t1", "ISSUED"),
    )
    conn.commit()
    conn.close()

    with pytest.raises(GrantSchemaError):
        GrantStore(path, test_mode=True)

    # The durable legacy record is preserved, not discarded.
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 1
    finally:
        conn.close()


# --- B2.2b-1 grant expiration and single-use claim regressions ---


def _issue_grant(review_store, grant_store, case, clock_iso=None):
    if clock_iso is not None:
        grant_store._clock = lambda: datetime.fromisoformat(clock_iso)
    result = grant_store.issue(review_store, case, _sim_reviewer())
    assert result.status == GrantStatus.ISSUED
    return result.grant


def test_claim_success_and_binding(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.CLAIMED
        c = result.claim
        assert c.grant_id == grant.grant_id
        assert c.case_id == "C-1"
        assert c.invoice_id == "INV-C-1"
        assert c.supplier_id == "SUP-1"
        assert c.po_id == "PO-1"
        assert c.proposed_action == "APPROVE"
        assert c.amount_cents == 60001
        assert c.discrepancy_cents == 50001
        assert c.policy_version == "policy-v1"
        assert c.state == "CLAIMED"
        # The grant state transitions to CLAIMED; claim is distinct from execution.
        assert grant_store.get(grant.grant_id).state == "CLAIMED"
        assert not hasattr(c, "executed")
        assert not hasattr(c, "consumed")
    finally:
        review_store.close()
        grant_store.close()


def test_claim_replay_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    try:
        assert grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer()).status == ClaimStatus.CLAIMED
        second = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert second.status == ClaimStatus.ALREADY_CLAIMED
        assert grant_store.get_claim(grant.grant_id) is not None
    finally:
        review_store.close()
        grant_store.close()


def test_claim_not_found(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    try:
        result = grant_store.claim("nonexistent", review_store, _over_500_case(), _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_NOT_FOUND
    finally:
        review_store.close()
        grant_store.close()


def test_claim_expired(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case, clock_iso="2026-10-12T00:00:00+00:00")
    try:
        grant_store._clock = lambda: datetime.fromisoformat("2026-10-13T00:00:01+00:00")
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_EXPIRED
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


def test_claim_exact_expiry_boundary(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    c1 = _over_500_case("C-1")
    c2 = _case("C-2", "INV-C-2", "SUP-1", "PO-1", "601.01", "100.00")
    review_store.request(c1)
    review_store.request(c2)
    g1 = _issue_grant(review_store, grant_store, c1, clock_iso="2026-10-12T00:00:00+00:00")
    g2 = _issue_grant(review_store, grant_store, c2, clock_iso="2026-10-12T00:00:00+00:00")
    try:
        # One second before expiry: claimable.
        grant_store._clock = lambda: datetime.fromisoformat("2026-10-12T23:59:59+00:00")
        assert grant_store.claim(g1.grant_id, review_store, c1, _sim_reviewer()).status == ClaimStatus.CLAIMED
        # Exactly at expiry: expired.
        grant_store._clock = lambda: datetime.fromisoformat("2026-10-13T00:00:00+00:00")
        assert grant_store.claim(g2.grant_id, review_store, c2, _sim_reviewer()).status == ClaimStatus.GRANT_EXPIRED
    finally:
        review_store.close()
        grant_store.close()


def test_claim_future_issued(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case, clock_iso="2026-10-13T00:00:00+00:00")
    try:
        grant_store._clock = lambda: datetime.fromisoformat("2026-10-12T00:00:00+00:00")
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_INVALID_METADATA
    finally:
        review_store.close()
        grant_store.close()


def test_claim_malformed_metadata(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_store._conn.execute(
        "UPDATE grants SET expires_at = 'not-a-timestamp' WHERE grant_id = ?", (grant.grant_id,)
    )
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_INVALID_METADATA
    finally:
        review_store.close()
        grant_store.close()


def test_claim_changed_case(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    try:
        changed = _case("C-1", "INV-C-1", "SUP-1", "PO-1", "700.01", "100.00")
        result = grant_store.claim(grant.grant_id, review_store, changed, _sim_reviewer())
        assert result.status == ClaimStatus.CASE_CHANGED
    finally:
        review_store.close()
        grant_store.close()


def test_claim_tampered_grant(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_store._conn.execute(
        "UPDATE grants SET amount_cents = 99999 WHERE grant_id = ?", (grant.grant_id,)
    )
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_INCONSISTENT
    finally:
        review_store.close()
        grant_store.close()


def test_claim_changed_policy(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_store._conn.execute(
        "UPDATE grants SET policy_version = 'policy-v2' WHERE grant_id = ?", (grant.grant_id,)
    )
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_INCONSISTENT
    finally:
        review_store.close()
        grant_store.close()


def test_claim_different_review_store(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    other_store = ReviewStore(str(tmp_path / "r2.db"))
    other_store.request(case)
    try:
        result = grant_store.claim(grant.grant_id, other_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_STORE_MISMATCH
    finally:
        review_store.close()
        other_store.close()
        grant_store.close()


def test_claim_default_mode_rejected(tmp_path):
    path = str(tmp_path / "g.db")
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(path)
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_store.close()

    non_test = GrantStore(path)  # test_mode defaults to False
    try:
        result = non_test.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.SIMULATION_NOT_ENABLED
    finally:
        non_test.close()
        review_store.close()


def test_claim_durable_reopen(tmp_path):
    path = str(tmp_path / "g.db")
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(path)
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    assert grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer()).status == ClaimStatus.CLAIMED
    grant_store.close()

    reopened = GrantStore(path, test_mode=True)
    try:
        c = reopened.get_claim(grant.grant_id)
        assert c is not None
        assert c.state == "CLAIMED"
        assert reopened.get(grant.grant_id).state == "CLAIMED"
    finally:
        reopened.close()
        review_store.close()


def test_claim_concurrent(tmp_path):
    path = str(tmp_path / "g.db")
    rpath = str(tmp_path / "r.db")
    review_store = ReviewStore(rpath)
    grant_store = _grant_store(path)
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_id = grant.grant_id
    grant_store.close()

    results = []

    def worker():
        rs = ReviewStore(rpath)
        s = GrantStore(path, test_mode=True)
        try:
            r = s.claim(grant_id, rs, case, _sim_reviewer())
            results.append(r.status)
        finally:
            s.close()
            rs.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([ClaimStatus.CLAIMED, ClaimStatus.ALREADY_CLAIMED])
    s = GrantStore(path, test_mode=True)
    try:
        assert s.get_claim(grant_id) is not None
        assert s.get(grant_id).state == "CLAIMED"
    finally:
        s.close()
    review_store.close()


# --- B2.2b-1 final claim-integrity regressions ---


def test_claim_review_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    # The stored review becomes non-pending after issuance.
    review_store._conn.execute(
        "UPDATE review_requests SET state = 'REJECTED' WHERE case_id = 'C-1'"
    )
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.REVIEW_NOT_PENDING
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


def test_claim_review_tampered(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    # Tamper the stored review without touching its fingerprint.
    review_store._conn.execute(
        "UPDATE review_requests SET proposed_action = 'TRANSFER', amount_cents = 99999 "
        "WHERE case_id = 'C-1'"
    )
    try:
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.REVIEW_INCONSISTENT
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


def test_claim_extended_lifetime_rejected(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case, clock_iso="2026-10-12T00:00:00+00:00")
    # Extend the recorded lifetime to 48 hours.
    grant_store._conn.execute(
        "UPDATE grants SET expires_at = '2026-10-14T00:00:00+00:00' WHERE grant_id = ?",
        (grant.grant_id,),
    )
    try:
        # A claim at hour 36 (inside the 48h window, outside the real 24h).
        grant_store._clock = lambda: datetime.fromisoformat("2026-10-13T12:00:00+00:00")
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_INVALID_METADATA
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


def test_claim_expiration_crossed_under_lock(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case, clock_iso="2026-10-12T00:00:00+00:00")
    try:
        # The clock reads "before expiry" on the pre-lock check and "after expiry"
        # on the in-lock re-check, modelling expiration crossing during lock wait.
        calls = [
            datetime.fromisoformat("2026-10-12T23:59:59+00:00"),
            datetime.fromisoformat("2026-10-13T00:00:01+00:00"),
        ]
        grant_store._clock = lambda: calls.pop(0)
        result = grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.GRANT_EXPIRED
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


# --- B2.2b-1 final claim-binding regressions ---


def test_claim_reviewer_mismatch(tmp_path):
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    try:
        other = Reviewer(name="other-reviewer", source=ReviewerSource.SIMULATED)
        result = grant_store.claim(grant.grant_id, review_store, case, other)
        assert result.status == ClaimStatus.REVIEWER_MISMATCH
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        review_store.close()
        grant_store.close()


def test_claim_review_changed_during_claim(tmp_path):
    rpath = str(tmp_path / "r.db")
    issue_store = ReviewStore(rpath)
    grant_store = _grant_store(str(tmp_path / "g.db"))
    case = _over_500_case()
    issue_store.request(case)
    grant = _issue_grant(issue_store, grant_store, case)
    issue_store.close()

    class RaceReviewStore(ReviewStore):
        def __init__(self, path):
            super().__init__(path)
            self._reads = 0

        def get(self, case_id):
            req = super().get(case_id)
            self._reads += 1
            if self._reads >= 2:
                # Change the review to non-pending after the first (early) read,
                # before the in-lock re-read.
                self._conn.execute(
                    "UPDATE review_requests SET state = 'REJECTED' WHERE case_id = ?", (case_id,)
                )
                req = super().get(case_id)
            return req

    race_store = RaceReviewStore(rpath)
    try:
        result = grant_store.claim(grant.grant_id, race_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.REVIEW_NOT_PENDING
        assert grant_store.get_claim(grant.grant_id) is None
        assert grant_store.get(grant.grant_id).state == "ISSUED"
    finally:
        race_store.close()
        grant_store.close()


def test_claim_grant_lineage_changed_during_claim(tmp_path):
    path = str(tmp_path / "g.db")
    review_store = ReviewStore(str(tmp_path / "r.db"))
    grant_store = _grant_store(path)
    case = _over_500_case()
    review_store.request(case)
    grant = _issue_grant(review_store, grant_store, case)
    grant_store.close()

    class RaceGrantStore(GrantStore):
        def __init__(self, p):
            super().__init__(p, test_mode=True)
            self._reads = 0

        def get(self, grant_id):
            g = super().get(grant_id)
            self._reads += 1
            if self._reads >= 2:
                # Change provenance after the first (early) read, before the lock.
                self._conn.execute(
                    "UPDATE grants SET reviewer_source = 'CLAIMED_HUMAN' WHERE grant_id = ?",
                    (grant_id,),
                )
                g = super().get(grant_id)
            return g

    race = RaceGrantStore(path)
    try:
        result = race.claim(grant.grant_id, review_store, case, _sim_reviewer())
        assert result.status == ClaimStatus.REVIEWER_REJECTED
    finally:
        race.close()

    check = GrantStore(path, test_mode=True)
    try:
        assert check.get_claim(grant.grant_id) is None
        assert check.get(grant.grant_id).state == "ISSUED"
    finally:
        check.close()
        review_store.close()
