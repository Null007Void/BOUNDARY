import sqlite3
from datetime import datetime

from boundary.authorized_verifier import (
    AuthorizedVerificationOutcome,
    verify_authorized,
)
from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.execution import AuthorizedExecutor, ExecutionStatus
from boundary.grant import GrantStore, Reviewer, ReviewerSource
from boundary.review import RequestStatus, ReviewStore

ISSUE_AT = "2026-10-12T00:00:00+00:00"
NOW = "2026-10-12T12:00:00+00:00"


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _case(case_id, invoice_id, supplier_id, po_ref, invoice_total, po_total, evidence=("ev-1",)):
    inv = Invoice(invoice_id=invoice_id, supplier_id=supplier_id, po_reference=po_ref,
                  issued_date="2026-10-12", line_items=[_line(1, "1", invoice_total)],
                  stated_total=invoice_total)
    po = PurchaseOrder(po_id="PO-1", supplier_id=supplier_id, issued_date="2026-10-12",
                       line_items=[_line(1, "1", po_total)], stated_total=po_total)
    return CaseInput(case_id=case_id, invoice=inv, po_lookup=POLookupFound(purchase_order=po),
                     supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                     evidence=list(evidence), policy_version="policy-v1", input_identity="test")


def _over_500_case(case_id="C-1"):
    return _case(case_id, "INV-" + case_id, "SUP-1", "PO-1", "600.01", "100.00")


def _sim_reviewer():
    return Reviewer(name="sim-reviewer", source=ReviewerSource.SIMULATED)


def _execute(tmp_path):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    case = _over_500_case()
    review_store.request(case)
    grant = grant_store.issue(review_store, case, _sim_reviewer()).grant
    review_store.close()
    grant_store.close()
    executor = AuthorizedExecutor(path, test_mode=True, clock=lambda: datetime.fromisoformat(NOW))
    result = executor.execute(grant.grant_id, case, _sim_reviewer())
    executor.close()
    assert result.status == ExecutionStatus.COMMITTED
    return path, case, grant.grant_id, result.action


def _mutate(path, sql, params=()):
    conn = sqlite3.connect(path)
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def test_verify_authorized_success(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    result = verify_authorized(path, case, grant_id=grant_id, receipt=receipt)
    assert result.outcome == AuthorizedVerificationOutcome.VERIFIED
    assert result.action is not None
    assert result.action.grant_id == grant_id


def test_verify_grant_without_action(tmp_path):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    case = _over_500_case()
    review_store.request(case)
    grant = grant_store.issue(review_store, case, _sim_reviewer()).grant
    review_store.close()
    grant_store.close()
    AuthorizedExecutor(path, test_mode=True).close()  # ensure authorized_actions table exists
    result = verify_authorized(path, case, grant_id=grant.grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.INCOMPLETE


def test_verify_claim_without_action(tmp_path):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    case = _over_500_case()
    review_store.request(case)
    grant = grant_store.issue(review_store, case, _sim_reviewer()).grant
    grant_store.claim(grant.grant_id, review_store, case, _sim_reviewer())
    review_store.close()
    grant_store.close()
    AuthorizedExecutor(path, test_mode=True).close()  # ensure authorized_actions table exists
    result = verify_authorized(path, case, grant_id=grant.grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.INCOMPLETE


def test_verify_not_found(tmp_path):
    path = str(tmp_path / "auth.db")
    # Create an empty database with the schema but no execution evidence.
    ReviewStore(path).close()
    GrantStore(path, test_mode=True).close()
    case = _over_500_case()
    result = verify_authorized(path, case)
    assert result.outcome == AuthorizedVerificationOutcome.NOT_FOUND


def test_verify_missing_grant(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "DELETE FROM grants WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_altered_grant(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE grants SET reviewer = 'other' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_missing_review(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "DELETE FROM review_requests WHERE case_id = ?", (case.case_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_missing_claim(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "DELETE FROM claims WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_changed_case(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    changed = _case("C-1", "INV-C-1", "SUP-1", "PO-1", "700.01", "100.00")
    result = verify_authorized(path, changed, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.RECORD_MISMATCH


def test_verify_changed_policy(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET policy_version = 'policy-v2' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.RECORD_MISMATCH


def test_verify_changed_reviewer(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET reviewer = 'other-reviewer' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_changed_amount(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET amount_cents = 99999 WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.RECORD_MISMATCH


def test_verify_changed_action(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET proposed_action = 'TRANSFER' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.RECORD_MISMATCH


def test_verify_fabricated_receipt(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    fabricated = receipt.model_copy(update={"amount_cents": 1})
    result = verify_authorized(path, case, grant_id=grant_id, receipt=fabricated)
    assert result.outcome == AuthorizedVerificationOutcome.RECEIPT_MISMATCH


def test_verify_missing_registry(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "DELETE FROM execution_registry WHERE case_id = ?", (case.case_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_contradictory_b1(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    # Create the B1 actions table, then insert a contradictory B1 approval for
    # the same invoice (bypassing the registry).
    from boundary.ledger import ActionJournal
    ActionJournal(path).close()
    _mutate(
        path,
        "INSERT INTO actions (case_id, action, invoice_id, po_id, supplier_id, amount_cents, "
        "input_fingerprint, policy_version, committed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C-B1", "APPROVE", "INV-C-1", "PO-1", "SUP-1", 60001, "fp", "policy-v1", "t"),
    )
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_readonly_and_reopen(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    # Verify twice (simulates a reopen) and confirm read-only (no mutation).
    r1 = verify_authorized(path, case, grant_id=grant_id)
    r2 = verify_authorized(path, case, grant_id=grant_id, receipt=receipt)
    assert r1.outcome == AuthorizedVerificationOutcome.VERIFIED
    assert r2.outcome == AuthorizedVerificationOutcome.VERIFIED
    # No rows added/changed by verification.
    conn = sqlite3.connect(path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0]
        assert n == 1
    finally:
        conn.close()


def test_verify_valid_chronology_no_receipt(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    result = verify_authorized(path, case, grant_id=grant_id)  # no receipt supplied
    assert result.outcome == AuthorizedVerificationOutcome.VERIFIED


def test_verify_invalid_claim_timestamp(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE claims SET claimed_at = 'not-a-timestamp' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_execution_before_issuance(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET executed_at = '2026-10-11T00:00:00+00:00' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_execution_after_expiration(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(path, "UPDATE authorized_actions SET executed_at = '2026-10-13T12:00:00+00:00' WHERE grant_id = ?", (grant_id,))
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_duplicate_b2_by_case(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    _mutate(
        path,
        "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, invoice_id, "
        "po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, reviewer, "
        "executed_at, state) VALUES ('a-dup', 'g-dup', 'C-1', 'fp', 'INV-OTHER', 'PO-1', 'SUP-1', "
        "'APPROVE', 60001, 50001, 'policy-v1', 'sim-reviewer', '2026-10-12T12:00:00+00:00', 'COMMITTED')",
    )
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED


def test_verify_duplicate_b2_by_invoice(tmp_path):
    path, case, grant_id, receipt = _execute(tmp_path)
    # Bypass the invoice uniqueness index to simulate corruption.
    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX IF EXISTS idx_action_invoice_identity")
    conn.commit()
    conn.close()
    _mutate(
        path,
        "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, invoice_id, "
        "po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, reviewer, "
        "executed_at, state) VALUES ('a-dup', 'g-dup', 'C-OTHER', 'fp', 'INV-C-1', 'PO-1', 'SUP-1', "
        "'APPROVE', 60001, 50001, 'policy-v1', 'sim-reviewer', '2026-10-12T12:00:00+00:00', 'COMMITTED')",
    )
    result = verify_authorized(path, case, grant_id=grant_id)
    assert result.outcome == AuthorizedVerificationOutcome.UNAUTHORIZED
