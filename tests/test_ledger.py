import sqlite3
import threading

import pytest

from boundary.classifier import ReasonCode
from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupUnavailable,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.guard import ActionDecision, evaluate
from boundary.ledger import ActionJournal, CommitStatus, case_fingerprint


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _invoice(inv_id="INV-1", supplier="SUP-1", po_ref="PO-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return Invoice(invoice_id=inv_id, supplier_id=supplier, po_reference=po_ref,
                   issued_date="2026-10-01", line_items=lines, stated_total=stated)


def _po(po_id="PO-1", supplier="SUP-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return PurchaseOrder(po_id=po_id, supplier_id=supplier, issued_date="2026-10-01",
                         line_items=lines, stated_total=stated)


def _case(case_id="C-1", invoice=None, po_lookup=None, supplier_state=SupplierMatchState.MATCH, evidence=None):
    return CaseInput(
        case_id=case_id,
        invoice=invoice or _invoice(),
        po_lookup=po_lookup or POLookupFound(purchase_order=_po()),
        supplier_lookup=SupplierLookupResult(state=supplier_state),
        evidence=evidence if evidence is not None else [],
        policy_version="policy-v1",
        input_identity="fixture-builder",
    )


def _exact(case_id="C-1"):
    return _case(case_id=case_id)


# --- Guard decisions ---

def test_guard_approves_exact_match():
    g = evaluate(_exact())
    assert g.action == ActionDecision.APPROVE
    assert g.proposal.exact_match_eligible is True


def test_guard_holds_non_exact():
    g = evaluate(_case(supplier_state=SupplierMatchState.MISMATCH))
    assert g.action == ActionDecision.HOLD


# --- Approval commits ---

def test_approve_exact_match_commits(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_exact())
        assert r.status == CommitStatus.COMMITTED
        assert r.decision.action == ActionDecision.APPROVE
        assert r.receipt is not None
        assert r.receipt.case_id == "C-1"
        assert r.receipt.action == "APPROVE"
        assert r.receipt.invoice_id == "INV-1"
        assert r.receipt.po_id == "PO-1"
        assert r.receipt.supplier_id == "SUP-1"
        assert r.receipt.amount_cents == 1000
        assert r.receipt.policy_version == "policy-v1"
        assert j.count() == 1
    finally:
        j.close()


def test_hold_non_exact_discrepancy(tmp_path):
    inv = _invoice(lines=[_line(1, "1", "10.01")], stated="10.01")
    po = _po(lines=[_line(1, "1", "10.00")], stated="10.00")
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
        assert r.status == CommitStatus.HELD
        assert r.receipt is None
        assert j.count() == 0
    finally:
        j.close()


def test_hold_unavailable_po(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(po_lookup=POLookupUnavailable()))
        assert r.status == CommitStatus.HELD
        assert j.count() == 0
    finally:
        j.close()


def test_hold_unresolved_evidence(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(evidence=["ev-1"]))
        assert r.status == CommitStatus.HELD
        assert ReasonCode.EVIDENCE_PENDING_REVIEW in r.decision.proposal.reason_codes
        assert j.count() == 0
    finally:
        j.close()


def test_hold_supplier_mismatch(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(supplier_state=SupplierMatchState.MISMATCH))
        assert r.status == CommitStatus.HELD
        assert j.count() == 0
    finally:
        j.close()


def test_hold_arithmetic_invalid(tmp_path):
    inv = _invoice(lines=[_line(1, "3", "10.00")], stated="100.00")
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(invoice=inv))
        assert r.status == CommitStatus.HELD
        assert ReasonCode.INVOICE_ARITHMETIC_MISMATCH in r.decision.proposal.reason_codes
        assert j.count() == 0
    finally:
        j.close()


# --- Idempotency / conflict ---

def test_idempotent_retry(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        first = j.approve(_exact())
        assert first.status == CommitStatus.COMMITTED
        second = j.approve(_exact())
        assert second.status == CommitStatus.IDEMPOTENT
        assert second.receipt.action_id == first.receipt.action_id
        assert j.count() == 1
    finally:
        j.close()


def test_changed_input_replay_conflict(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        first = j.approve(_exact(case_id="C-1"))  # amount 1000 cents
        assert first.status == CommitStatus.COMMITTED

        # Same case_id, different (still exact-match) content.
        changed = _case(
            case_id="C-1",
            invoice=_invoice(lines=[_line(1, "1", "99.00")], stated="99.00"),
            po_lookup=POLookupFound(purchase_order=_po(lines=[_line(1, "1", "99.00")], stated="99.00")),
        )
        second = j.approve(changed)
        assert second.status == CommitStatus.CONFLICT
        assert j.count() == 1  # no second row
        # original record preserved, not overwritten
        rec = j.get("C-1")
        assert rec.amount_cents == 1000
        assert rec.invoice_id == "INV-1"
    finally:
        j.close()


# --- Persistence and receipt correspondence ---

def test_persistence_after_reopen(tmp_path):
    path = str(tmp_path / "j.db")
    j = ActionJournal(path)
    r = j.approve(_exact())
    assert r.status == CommitStatus.COMMITTED
    j.close()

    j2 = ActionJournal(path)
    try:
        rec = j2.get("C-1")
        assert rec is not None
        assert rec.case_id == "C-1"
        assert rec.amount_cents == 1000
        assert j2.count() == 1
    finally:
        j2.close()


def test_receipt_to_record_correspondence(tmp_path):
    path = str(tmp_path / "j.db")
    j = ActionJournal(path)
    r = j.approve(_exact())
    receipt = r.receipt
    j.close()

    conn = sqlite3.connect(path)
    row = conn.execute("SELECT * FROM actions WHERE case_id = 'C-1'").fetchone()
    conn.close()

    assert row is not None
    assert receipt.action_id == row[0]
    assert receipt.case_id == row[1]
    assert receipt.action == row[2]
    assert receipt.invoice_id == row[3]
    assert receipt.po_id == row[4]
    assert receipt.supplier_id == row[5]
    assert receipt.amount_cents == row[6]
    assert receipt.input_fingerprint == row[7]
    assert receipt.policy_version == row[8]
    assert receipt.committed_at == row[9]


# --- Fingerprint determinism ---

def test_fingerprint_deterministic_and_sensitive():
    assert case_fingerprint(_exact()) == case_fingerprint(_exact())
    assert case_fingerprint(_exact(case_id="C-1")) != case_fingerprint(_exact(case_id="C-2"))


# --- B1.2a SQLite safety corrections ---

def test_amount_overflow_fails_closed(tmp_path):
    huge = "92233720368547758.08"  # 2**63 cents, exceeds SQLite signed 64-bit
    inv = _invoice(lines=[_line(1, "1", huge)], stated=huge)
    po = _po(lines=[_line(1, "1", huge)], stated=huge)
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
        assert r.status == CommitStatus.AMOUNT_OVERFLOW
        assert r.receipt is None
        assert j.count() == 0
    finally:
        j.close()


def test_amount_at_max_int64_boundary_commits(tmp_path):
    max_cents = (2 ** 63) - 1
    dollars = f"{max_cents // 100}.{max_cents % 100:02d}"
    inv = _invoice(lines=[_line(1, "1", dollars)], stated=dollars)
    po = _po(lines=[_line(1, "1", dollars)], stated=dollars)
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
        assert r.status == CommitStatus.COMMITTED
        assert r.receipt.amount_cents == max_cents
        assert j.count() == 1
    finally:
        j.close()


def test_duplicate_invoice_across_case_ids(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        first = j.approve(_exact(case_id="C-1"))
        assert first.status == CommitStatus.COMMITTED
        # different case id, same invoice identity (SUP-1, INV-1), still exact match
        second = j.approve(_exact(case_id="C-2"))
        assert second.status == CommitStatus.DUPLICATE_INVOICE
        assert j.count() == 1
    finally:
        j.close()


def test_changed_input_ineligible_conflict(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        first = j.approve(_exact(case_id="C-1"))
        assert first.status == CommitStatus.COMMITTED
        # same case id, changed now-ineligible input (supplier mismatch)
        changed = _case(case_id="C-1", supplier_state=SupplierMatchState.MISMATCH)
        second = j.approve(changed)
        assert second.status == CommitStatus.CONFLICT
        assert j.count() == 1
        # original preserved, not overwritten
        assert j.get("C-1").amount_cents == 1000
    finally:
        j.close()


def test_concurrent_identical_requests(tmp_path):
    path = str(tmp_path / "j.db")
    results = []
    lock = threading.Lock()

    def worker():
        j = ActionJournal(path)
        try:
            r = j.approve(_exact())
            with lock:
                results.append(r.status)
        finally:
            j.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([CommitStatus.COMMITTED, CommitStatus.IDEMPOTENT])
    j = ActionJournal(path)
    try:
        assert j.count() == 1
    finally:
        j.close()


def test_concurrent_conflicting_requests(tmp_path):
    path = str(tmp_path / "j.db")
    results = []
    lock = threading.Lock()

    def worker(case):
        j = ActionJournal(path)
        try:
            r = j.approve(case)
            with lock:
                results.append(r.status)
        finally:
            j.close()

    c1 = _exact(case_id="C-1")  # amount 1000
    c2 = _case(
        case_id="C-1",
        invoice=_invoice(lines=[_line(1, "1", "99.00")], stated="99.00"),
        po_lookup=POLookupFound(purchase_order=_po(lines=[_line(1, "1", "99.00")], stated="99.00")),
    )
    threads = [threading.Thread(target=worker, args=(c,)) for c in (c1, c2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([CommitStatus.COMMITTED, CommitStatus.CONFLICT])
    j = ActionJournal(path)
    try:
        assert j.count() == 1
    finally:
        j.close()


# --- B1.2a final concurrency correction: forced interleaving ---

def test_invoice_identity_excludes_own_case_id(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        j.approve(_exact(case_id="C-1"))
        # own case id is not a "duplicate"
        assert j._invoice_exists("SUP-1", "INV-1", "C-1") is False
        # a different case id is
        assert j._invoice_exists("SUP-1", "INV-1", "C-2") is True
    finally:
        j.close()


def _make_race_journal(path, committed_case):
    """A journal that forces ``committed_case`` to commit between this journal's
    case-id lookup and its invoice-identity check."""

    class RaceJournal(ActionJournal):
        def __init__(self, path):
            super().__init__(path)
            self._injected = False

        def _invoice_exists(self, supplier_id, invoice_id, case_id):
            if not self._injected:
                self._injected = True
                other = ActionJournal(path)
                try:
                    other.approve(committed_case)
                finally:
                    other.close()
            return super()._invoice_exists(supplier_id, invoice_id, case_id)

    return RaceJournal(path)


def test_same_case_retry_precedes_invoice_duplicate_under_race(tmp_path):
    # R2 retries the same case id + same content as R1. R1's commit lands between
    # R2's case-id lookup (stale) and R2's invoice check. Correct result: IDEMPOTENT.
    r2 = _make_race_journal(str(tmp_path / "j.db"), _exact(case_id="C-1"))
    try:
        result = r2.approve(_exact(case_id="C-1"))
    finally:
        r2.close()

    assert result.status == CommitStatus.IDEMPOTENT
    assert result.receipt is not None

    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        assert j.count() == 1
    finally:
        j.close()


def test_conflict_precedes_invoice_duplicate_under_race(tmp_path):
    # R2 reuses the same case id with different content (different invoice). R1's
    # commit lands in R2's window. Correct result: CONFLICT, not DUPLICATE_INVOICE.
    changed = _case(
        case_id="C-1",
        invoice=_invoice(inv_id="INV-2", lines=[_line(1, "1", "99.00")], stated="99.00"),
        po_lookup=POLookupFound(purchase_order=_po(lines=[_line(1, "1", "99.00")], stated="99.00")),
    )
    r2 = _make_race_journal(str(tmp_path / "j.db"), _exact(case_id="C-1"))
    try:
        result = r2.approve(changed)
    finally:
        r2.close()

    assert result.status == CommitStatus.CONFLICT

    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        assert j.count() == 1
    finally:
        j.close()
