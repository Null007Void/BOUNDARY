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
from boundary.integrated import Boundary
from boundary.ledger import ActionJournal, CommitStatus
from boundary.registry import RegistryConflictError
from boundary.review import RequestStatus, ReviewStore

ISSUE_AT = "2026-10-12T00:00:00+00:00"
NOW = "2026-10-12T12:00:00+00:00"


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _exact(case_id="C-1", inv_id="INV-1", supplier="SUP-1", po_id="PO-1"):
    inv = Invoice(invoice_id=inv_id, supplier_id=supplier, po_reference=po_id,
                  issued_date="2026-10-01", line_items=[_line(1, "1", "10.00")], stated_total="10.00")
    po = PurchaseOrder(po_id=po_id, supplier_id=supplier, issued_date="2026-10-01",
                       line_items=[_line(1, "1", "10.00")], stated_total="10.00")
    return CaseInput(case_id=case_id, invoice=inv, po_lookup=POLookupFound(purchase_order=po),
                     supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                     evidence=[], policy_version="policy-v1", input_identity="cross-path")


def _exception(case_id, invoice_id, supplier_id="SUP-1", invoice_total="600.01"):
    inv = Invoice(invoice_id=invoice_id, supplier_id=supplier_id, po_reference="PO-1",
                  issued_date="2026-10-12", line_items=[_line(1, "1", invoice_total)], stated_total=invoice_total)
    po = PurchaseOrder(po_id="PO-1", supplier_id=supplier_id, issued_date="2026-10-12",
                       line_items=[_line(1, "1", "100.00")], stated_total="100.00")
    return CaseInput(case_id=case_id, invoice=inv, po_lookup=POLookupFound(purchase_order=po),
                     supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                     evidence=["ev-1"], policy_version="policy-v1", input_identity="cross-path")


def _sim_reviewer():
    return Reviewer(name="sim-reviewer", source=ReviewerSource.SIMULATED)


def _setup_b2(path, case, now_iso=NOW):
    """Create a review + grant for a B2 exception case on the shared path, then executor."""
    review_store = ReviewStore(path)
    grant_store = GrantStore(path, test_mode=True, clock=lambda: datetime.fromisoformat(ISSUE_AT))
    assert review_store.request(case).status == RequestStatus.CREATED
    grant = grant_store.issue(review_store, case, _sim_reviewer()).grant
    review_store.close()
    grant_store.close()
    executor = AuthorizedExecutor(path, test_mode=True, clock=lambda: datetime.fromisoformat(now_iso))
    return executor, grant


def _registry_count(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM execution_registry").fetchone()[0]
    finally:
        conn.close()


def test_cross_path_b1_then_b2_invoice(tmp_path):
    path = str(tmp_path / "shared.db")
    journal = ActionJournal(path)
    r = journal.approve(_exact("C-1", "INV-X", "SUP-1"))
    assert r.status == CommitStatus.COMMITTED
    journal.close()

    executor, grant = _setup_b2(path, _exception("C-2", "INV-X", "SUP-1"))
    try:
        result = executor.execute(grant.grant_id, _exception("C-2", "INV-X", "SUP-1"), _sim_reviewer())
        assert result.status == ExecutionStatus.DUPLICATE_INVOICE
        assert result.action is None
    finally:
        executor.close()
    assert _registry_count(path) == 1


def test_cross_path_b2_then_b1_invoice(tmp_path):
    path = str(tmp_path / "shared.db")
    executor, grant = _setup_b2(path, _exception("C-2", "INV-X", "SUP-1"))
    case = _exception("C-2", "INV-X", "SUP-1")
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
    finally:
        executor.close()

    journal = ActionJournal(path)
    try:
        r = journal.approve(_exact("C-1", "INV-X", "SUP-1"))
        assert r.status == CommitStatus.DUPLICATE_INVOICE
        assert r.receipt is None
    finally:
        journal.close()
    assert _registry_count(path) == 1


def test_cross_path_case_id_conflict(tmp_path):
    path = str(tmp_path / "shared.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-1", "INV-A", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()

    # B2 tries the same case id C-1 with a *different* invoice.
    executor, grant = _setup_b2(path, _exception("C-1", "INV-B", "SUP-1"))
    case = _exception("C-1", "INV-B", "SUP-1")
    try:
        result = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert result.status == ExecutionStatus.CONFLICT
        assert result.action is None
        # No partial claim or action left behind.
        assert executor._grant.get_claim(grant.grant_id) is None
        assert executor._conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 0
    finally:
        executor.close()
    assert _registry_count(path) == 1


def test_replay_legitimate(tmp_path):
    path = str(tmp_path / "auth.db")
    executor, grant = _setup_b2(path, _exception("C-1", "INV-C-1"))
    case = _exception("C-1", "INV-C-1")
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
        replay = executor.execute(grant.grant_id, case, _sim_reviewer())
        assert replay.status == ExecutionStatus.ALREADY_EXECUTED
        assert replay.action is not None
    finally:
        executor.close()


def test_replay_changed_case(tmp_path):
    path = str(tmp_path / "auth.db")
    executor, grant = _setup_b2(path, _exception("C-1", "INV-C-1"))
    try:
        assert executor.execute(grant.grant_id, _exception("C-1", "INV-C-1"), _sim_reviewer()).status == ExecutionStatus.COMMITTED
        # Replay with a different invoice total (changed case contents).
        changed = _exception("C-1", "INV-C-1", "SUP-1", "700.01")
        replay = executor.execute(grant.grant_id, changed, _sim_reviewer())
        assert replay.status == ExecutionStatus.REPLAY_MISMATCH
        assert replay.action is None
    finally:
        executor.close()


def test_replay_changed_reviewer(tmp_path):
    path = str(tmp_path / "auth.db")
    executor, grant = _setup_b2(path, _exception("C-1", "INV-C-1"))
    case = _exception("C-1", "INV-C-1")
    try:
        assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
        other = Reviewer(name="other-reviewer", source=ReviewerSource.SIMULATED)
        replay = executor.execute(grant.grant_id, case, other)
        assert replay.status == ExecutionStatus.REPLAY_MISMATCH
        assert replay.action is None
    finally:
        executor.close()


def test_cross_path_concurrent(tmp_path):
    path = str(tmp_path / "shared.db")
    # Pre-create B2 review + grant for invoice INV-X (exception case C-2).
    b2_case = _exception("C-2", "INV-X", "SUP-1")
    executor, grant = _setup_b2(path, b2_case)
    executor.close()

    b1_case = _exact("C-1", "INV-X", "SUP-1")
    results = []

    def b1_worker():
        j = ActionJournal(path)
        try:
            results.append(j.approve(b1_case).status.value)
        finally:
            j.close()

    def b2_worker():
        ex = AuthorizedExecutor(path, test_mode=True, clock=lambda: datetime.fromisoformat(NOW))
        try:
            results.append(ex.execute(grant.grant_id, b2_case, _sim_reviewer()).status.value)
        finally:
            ex.close()

    threads = [threading.Thread(target=b1_worker), threading.Thread(target=b2_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted(["COMMITTED", "DUPLICATE_INVOICE"])
    assert _registry_count(path) == 1


# --- B2.2b-2a final registry-integrity regressions ---


def _create_pre_registry_conflict(path, mode):
    """Create a pre-registry database with a cross-path conflict (no registry table)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL UNIQUE, "
        "action TEXT NOT NULL, invoice_id TEXT NOT NULL, po_id TEXT NOT NULL, supplier_id TEXT NOT NULL, "
        "amount_cents INTEGER NOT NULL, input_fingerprint TEXT NOT NULL, policy_version TEXT NOT NULL, "
        "committed_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE authorized_actions (id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL UNIQUE, "
        "grant_id TEXT NOT NULL UNIQUE, case_id TEXT NOT NULL, case_fingerprint TEXT NOT NULL, "
        "invoice_id TEXT NOT NULL, po_id TEXT NOT NULL, supplier_id TEXT NOT NULL, proposed_action TEXT NOT NULL, "
        "amount_cents INTEGER NOT NULL, discrepancy_cents INTEGER NOT NULL, policy_version TEXT NOT NULL, "
        "reviewer TEXT NOT NULL, executed_at TEXT NOT NULL, state TEXT NOT NULL)"
    )
    if mode == "case":
        conn.execute(
            "INSERT INTO actions (case_id, action, invoice_id, po_id, supplier_id, amount_cents, "
            "input_fingerprint, policy_version, committed_at) VALUES ('C-1','APPROVE','INV-A','PO-1','SUP-1',1000,'fp-a','policy-v1','t')"
        )
        conn.execute(
            "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, invoice_id, po_id, "
            "supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, reviewer, executed_at, state) "
            "VALUES ('a2','g2','C-1','fp-b','INV-B','PO-1','SUP-1','APPROVE',1000,500,'policy-v1','sim','t','COMMITTED')"
        )
    else:  # invoice
        conn.execute(
            "INSERT INTO actions (case_id, action, invoice_id, po_id, supplier_id, amount_cents, "
            "input_fingerprint, policy_version, committed_at) VALUES ('C-1','APPROVE','INV-X','PO-1','SUP-1',1000,'fp-a','policy-v1','t')"
        )
        conn.execute(
            "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, invoice_id, po_id, "
            "supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, reviewer, executed_at, state) "
            "VALUES ('a2','g2','C-2','fp-b','INV-X','PO-1','SUP-1','APPROVE',1000,500,'policy-v1','sim','t','COMMITTED')"
        )
    conn.commit()
    conn.close()


def test_historical_b1_approval_reconciled(tmp_path):
    path = str(tmp_path / "db.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-1", "INV-OLD", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()
    # Simulate a pre-registry state: the registry loses its entries.
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM execution_registry")
    conn.commit()
    conn.close()
    # Reopening reconciles the historical B1 approval; B2 duplicate is blocked.
    executor, grant = _setup_b2(path, _exception("C-2", "INV-OLD", "SUP-1"))
    try:
        result = executor.execute(grant.grant_id, _exception("C-2", "INV-OLD", "SUP-1"), _sim_reviewer())
        assert result.status == ExecutionStatus.DUPLICATE_INVOICE
    finally:
        executor.close()


def test_historical_b2_action_reconciled(tmp_path):
    path = str(tmp_path / "db.db")
    executor, grant = _setup_b2(path, _exception("C-1", "INV-OLD", "SUP-1"))
    case = _exception("C-1", "INV-OLD", "SUP-1")
    assert executor.execute(grant.grant_id, case, _sim_reviewer()).status == ExecutionStatus.COMMITTED
    executor.close()
    # Simulate a pre-registry state: the registry loses its entries.
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM execution_registry")
    conn.commit()
    conn.close()
    # Reopening reconciles the historical B2 action; B1 duplicate is blocked.
    journal = ActionJournal(path)
    try:
        r = journal.approve(_exact("C-2", "INV-OLD", "SUP-1"))
        assert r.status == CommitStatus.DUPLICATE_INVOICE
    finally:
        journal.close()


def test_historical_case_id_collision_fails_closed(tmp_path):
    path = str(tmp_path / "db.db")
    _create_pre_registry_conflict(path, "case")
    with pytest.raises(RegistryConflictError):
        ActionJournal(path)
    # No data lost.
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_historical_invoice_collision_fails_closed(tmp_path):
    path = str(tmp_path / "db.db")
    _create_pre_registry_conflict(path, "invoice")
    with pytest.raises(RegistryConflictError):
        ActionJournal(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM authorized_actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_canonical_database_config(tmp_path):
    path = str(tmp_path / "canonical.db")
    # Set up the B2 review + grant on the canonical path.
    executor, grant = _setup_b2(path, _exception("C-2", "INV-X", "SUP-1"))
    executor.close()
    # The integrated entry point wires both paths to the SAME path.
    boundary = Boundary(path, test_mode=True, clock=lambda: datetime.fromisoformat(NOW))
    try:
        assert boundary.approve_exact(_exact("C-1", "INV-X", "SUP-1")).status == CommitStatus.COMMITTED
        result = boundary.execute(grant.grant_id, _exception("C-2", "INV-X", "SUP-1"), _sim_reviewer())
        assert result.status == ExecutionStatus.DUPLICATE_INVOICE
    finally:
        boundary.close()
    assert _registry_count(path) == 1


def test_registry_case_invoice_mismatch(tmp_path):
    path = str(tmp_path / "db.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-B1", "INV-OLD", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()
    # Tamper: registry maps C-B1 to a different invoice.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE execution_registry SET invoice_id = 'INV-SPOOFED' WHERE case_id = 'C-B1'")
    conn.commit()
    conn.close()
    with pytest.raises(RegistryConflictError):
        ActionJournal(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_registry_invoice_case_mismatch(tmp_path):
    path = str(tmp_path / "db.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-B1", "INV-OLD", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()
    # Tamper: registry maps INV-OLD to a different case id.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE execution_registry SET case_id = 'C-SPOOFED' WHERE case_id = 'C-B1'")
    conn.commit()
    conn.close()
    with pytest.raises(RegistryConflictError):
        ActionJournal(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_registry_orphan_entry(tmp_path):
    path = str(tmp_path / "db.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-B1", "INV-OLD", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()
    # Insert a registry entry with no corresponding committed action.
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO execution_registry (case_id, supplier_id, invoice_id) "
        "VALUES ('C-ORPHAN', 'SUP-1', 'INV-ORPHAN')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(RegistryConflictError):
        ActionJournal(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
    finally:
        conn.close()


def test_registry_idempotent_reopen(tmp_path):
    path = str(tmp_path / "db.db")
    journal = ActionJournal(path)
    assert journal.approve(_exact("C-B1", "INV-OLD", "SUP-1")).status == CommitStatus.COMMITTED
    journal.close()
    # Reopening an already-reconciled database is a no-op (no duplicate, no error).
    j2 = ActionJournal(path)
    try:
        assert j2.count() == 1
        n = j2._conn.execute("SELECT COUNT(*) FROM execution_registry").fetchone()[0]
        assert n == 1
    finally:
        j2.close()
