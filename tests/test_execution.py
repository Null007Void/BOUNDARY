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
from boundary.execution import AuthorizedExecutor, ExecutionStatus
from boundary.grant import GrantStore, Reviewer, ReviewerSource
from boundary.ledger import case_fingerprint
from boundary.review import RequestStatus, ReviewStore

ISSUE_AT = "2026-10-12T00:00:00+00:00"
NOW = "2026-10-12T12:00:00+00:00"


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


def _setup(tmp_path, now_iso=NOW):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(
        path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT)
    )
    case = _over_500_case()
    assert review_store.request(case).status == RequestStatus.CREATED
    grant = grant_store.issue(review_store, case, _sim_reviewer()).grant
    review_store.close()
    grant_store.close()
    executor = AuthorizedExecutor(
        path, test_mode=True, clock=lambda: datetime.fromisoformat(now_iso)
    )
    return executor, path, case, grant


def test_execute_success(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        result = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert result.status == ExecutionStatus.COMMITTED
        a = result.action
        assert a.grant_id == grant.grant_id
        assert a.case_id == "C-1"
        assert a.invoice_id == "INV-C-1"
        assert a.supplier_id == "SUP-1"
        assert a.proposed_action == "APPROVE"
        assert a.amount_cents == 60001
        assert a.discrepancy_cents == 50001
        assert a.state == "COMMITTED"
        # Grant transitioned to CLAIMED and a claim record exists.
        assert executor._grant.get(grant.grant_id).state == "CLAIMED"
        assert executor._grant.get_claim(grant.grant_id) is not None
    finally:
        executor.close()


def test_execute_replay(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
        second = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert second.status == ExecutionStatus.ALREADY_EXECUTED
        assert second.action is not None
        n = executor._conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0]
        assert n == 1
    finally:
        executor.close()


def test_execute_revoked_review(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    executor._conn.execute("UPDATE review_requests SET state = 'REJECTED' WHERE case_id = 'C-1'")
    try:
        result = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert result.status == ExecutionStatus.REVIEW_NOT_PENDING
        assert executor._grant.get_claim(grant.grant_id) is None
        assert executor._conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 0
        assert executor._grant.get(grant.grant_id).state == "ISSUED"
    finally:
        executor.close()


def test_execute_expired(tmp_path):
    executor, path, case, grant = _setup(tmp_path, now_iso="2026-10-13T00:00:01+00:00")
    try:
        result = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert result.status == ExecutionStatus.GRANT_EXPIRED
        assert executor._conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 0
    finally:
        executor.close()


def test_execute_reviewer_mismatch(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        other = Reviewer(name="other-reviewer", source=ReviewerSource.SIMULATED)
        result = executor.execute(grant.grant_id, case, other)
        assert result.status == ExecutionStatus.REVIEWER_MISMATCH
    finally:
        executor.close()


def test_execute_changed_case(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        changed = _case("C-1", "INV-C-1", "SUP-1", "PO-1", "700.01", "100.00")
        result = executor.execute(grant.grant_id, changed, _sim_reviewer())
        assert result.status == ExecutionStatus.CASE_CHANGED
    finally:
        executor.close()


def _raw_grant_and_review(conn, case, store_id, grant_id):
    fp = case_fingerprint(case)
    po = case.po_lookup.purchase_order
    amount = case.invoice.computed_total_cents
    discrepancy = amount - po.computed_total_cents
    conn.execute(
        "INSERT INTO review_requests (case_id, case_fingerprint, invoice_id, po_id, supplier_id, "
        "proposed_action, amount_cents, discrepancy_cents, policy_version, state, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (case.case_id, fp, case.invoice.invoice_id, po.po_id, case.invoice.supplier_id,
         "APPROVE", amount, discrepancy, case.policy_version, "WAITING_AUTHORITY",
         "x", "2026-10-12T00:00:00+00:00"),
    )
    request_id = conn.execute(
        "SELECT id FROM review_requests WHERE case_id = ?", (case.case_id,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO grants (grant_id, review_store_id, request_id, case_id, case_fingerprint, "
        "invoice_id, po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, "
        "policy_version, reviewer, reviewer_source, issued_at, expires_at, state) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (grant_id, store_id, request_id, case.case_id, fp, case.invoice.invoice_id, po.po_id,
         case.invoice.supplier_id, "APPROVE", amount, discrepancy, case.policy_version,
         "sim-reviewer", "SIMULATED", ISSUE_AT, "2026-10-13T00:00:00+00:00", "ISSUED"),
    )


def test_execute_duplicate_invoice(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
        # Bypass review/grant dedup to exercise the action-level invoice guard.
        executor._conn.execute("DROP INDEX IF EXISTS idx_review_invoice_identity")
        executor._conn.execute("DROP INDEX IF EXISTS idx_grant_invoice_identity")
        case2 = _case("C-2", "INV-C-1", "SUP-1", "PO-1", "600.01", "100.00")
        _raw_grant_and_review(executor._conn, case2, executor._review.store_id, "grant-2")
        result = executor.execute("grant-2", case2, _sim_reviewer())
        assert result.status == ExecutionStatus.DUPLICATE_INVOICE
        assert executor._conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 1
    finally:
        executor.close()


def test_execute_concurrent(tmp_path):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    case = _over_500_case()
    review_store.request(case)
    grant_id = grant_store.issue(review_store, case, _sim_reviewer()).grant.grant_id
    review_store.close()
    grant_store.close()

    results = []

    def worker():
        ex = AuthorizedExecutor(path, test_mode=True, clock=lambda: datetime.fromisoformat(NOW))
        try:
            r = ex.execute(grant_id, case, _sim_reviewer())
            results.append(r.status)
        finally:
            ex.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([ExecutionStatus.COMMITTED, ExecutionStatus.ALREADY_EXECUTED])
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_execute_crash_rollback(tmp_path):
    path = str(tmp_path / "auth.db")
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    case = _over_500_case()
    review_store.request(case)
    grant_id = grant_store.issue(review_store, case, _sim_reviewer()).grant.grant_id
    review_store.close()
    grant_store.close()

    class CrashExecutor(AuthorizedExecutor):
        def _insert_action(self, action):
            raise RuntimeError("simulated crash before commit")

    ex = CrashExecutor(path, test_mode=True, clock=lambda: datetime.fromisoformat(NOW))
    try:
        with pytest.raises(RuntimeError):
            ex.execute(grant_id, case, _sim_reviewer())
    finally:
        ex.close()

    # No partial claim or action persisted.
    check = GrantStore(path, test_mode=True)
    try:
        assert check.get(grant_id).state == "ISSUED"
        assert check.get_claim(grant_id) is None
    finally:
        check.close()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 0
    finally:
        conn.close()


def test_executor_single_database(tmp_path):
    path = str(tmp_path / "auth.db")
    ex = AuthorizedExecutor(path, test_mode=True)
    try:
        tables = {r[0] for r in ex._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
        assert {"review_requests", "grants", "claims", "authorized_actions"} <= tables
    finally:
        ex.close()


def test_execute_invoice_uniqueness_constraint(tmp_path):
    executor, path, case, grant = _setup(tmp_path)
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
        with pytest.raises(sqlite3.IntegrityError):
            executor._conn.execute(
                "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, "
                "invoice_id, po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, "
                "policy_version, reviewer, executed_at, state) "
                "VALUES ('a2', 'g2', 'C-2', 'fp', 'INV-C-1', 'PO-1', 'SUP-1', 'APPROVE', 1, 1, "
                "'policy-v1', 'sim', 't', 'COMMITTED')"
            )
    finally:
        executor.close()
