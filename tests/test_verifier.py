import sqlite3

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
from boundary.ledger import ActionJournal, CommitStatus, case_fingerprint
from boundary.verifier import VerificationOutcome, verify


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


_MUTABLE_COLUMNS = {"amount_cents", "supplier_id", "po_id", "input_fingerprint", "policy_version"}


def _mutate(path, column, value):
    assert column in _MUTABLE_COLUMNS
    conn = sqlite3.connect(path)
    conn.execute(f"UPDATE actions SET {column} = ? WHERE case_id = ?", (value, "C-1"))
    conn.commit()
    conn.close()


def _insert_record(path, case, action="APPROVE"):
    """Insert a committed record directly, bypassing the guarded ledger (to
    simulate a corrupted or bypassed execution path)."""
    po = case.po_lookup.purchase_order
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO actions "
        "(case_id, action, invoice_id, po_id, supplier_id, amount_cents, "
        "input_fingerprint, policy_version, committed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (case.case_id, action, case.invoice.invoice_id, po.po_id, case.invoice.supplier_id,
         case.invoice.computed_total_cents, case_fingerprint(case), case.policy_version,
         "2026-10-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()


def test_valid_committed_approval_verifies(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        r = j.approve(_exact())
        assert r.status == CommitStatus.COMMITTED
        v = verify(j, _exact(), receipt=r.receipt)
        assert v.outcome == VerificationOutcome.VERIFIED
        assert v.record is not None
        assert v.proposal.exact_match_eligible is True
    finally:
        j.close()


def test_absent_action_cannot_verify(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        v = verify(j, _exact(case_id="C-9"))  # eligible but never committed
        assert v.outcome == VerificationOutcome.NOT_FOUND
    finally:
        j.close()


def test_held_proposal_cannot_verify(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        held = _case(supplier_state=SupplierMatchState.MISMATCH)
        assert j.approve(held).status == CommitStatus.HELD
        v = verify(j, held)
        assert v.outcome == VerificationOutcome.NOT_ELIGIBLE
    finally:
        j.close()


def test_fabricated_receipt_fails(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        real = j.approve(_exact()).receipt

        fabricated = real.model_copy(update={"amount_cents": 9999})
        assert verify(j, _exact(), receipt=fabricated).outcome == VerificationOutcome.RECEIPT_MISMATCH

        altered = real.model_copy(update={"action_id": 99999})
        assert verify(j, _exact(), receipt=altered).outcome == VerificationOutcome.RECEIPT_MISMATCH

        assert verify(j, _exact(), receipt=real).outcome == VerificationOutcome.VERIFIED
    finally:
        j.close()


@pytest.mark.parametrize("column,value", [
    ("amount_cents", 9999),
    ("supplier_id", "SUP-X"),
    ("po_id", "PO-X"),
    ("input_fingerprint", "f" * 64),
    ("policy_version", "policy-v2"),
])
def test_mutated_record_field_fails(tmp_path, column, value):
    path = str(tmp_path / "j.db")
    j = ActionJournal(path)
    j.approve(_exact())
    j.close()
    _mutate(path, column, value)
    # The verifier is read-only: open the journal read-only so the write-side
    # registry reconciliation does not run against the deliberately mutated row.
    j = ActionJournal(path, read_only=True)
    try:
        v = verify(j, _exact())
        assert v.outcome == VerificationOutcome.RECORD_MISMATCH
        assert any(column in reason for reason in v.reasons)
    finally:
        j.close()


def test_changed_source_case_cannot_verify(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        j.approve(_exact(case_id="C-1"))
        changed = _case(
            case_id="C-1",
            invoice=_invoice(lines=[_line(1, "1", "99.00")], stated="99.00"),
            po_lookup=POLookupFound(purchase_order=_po(lines=[_line(1, "1", "99.00")], stated="99.00")),
        )
        v = verify(j, changed)
        assert v.outcome == VerificationOutcome.RECORD_MISMATCH
    finally:
        j.close()


def test_duplicate_and_conflict_do_not_create_extra_approvals(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        j.approve(_exact(case_id="C-1"))
        assert j.approve(_exact(case_id="C-1")).status == CommitStatus.IDEMPOTENT
        changed = _case(
            case_id="C-1",
            invoice=_invoice(inv_id="INV-2", lines=[_line(1, "1", "99.00")], stated="99.00"),
            po_lookup=POLookupFound(purchase_order=_po(lines=[_line(1, "1", "99.00")], stated="99.00")),
        )
        assert j.approve(changed).status == CommitStatus.CONFLICT
        assert j.count() == 1
        # only the original case verifies as committed
        assert verify(j, _exact(case_id="C-1")).outcome == VerificationOutcome.VERIFIED
        assert verify(j, changed).outcome == VerificationOutcome.RECORD_MISMATCH
    finally:
        j.close()


def test_verify_after_reopen(tmp_path):
    path = str(tmp_path / "j.db")
    j = ActionJournal(path)
    r = j.approve(_exact())
    assert r.status == CommitStatus.COMMITTED
    j.close()

    j2 = ActionJournal(path)
    try:
        v = verify(j2, _exact(), receipt=r.receipt)
        assert v.outcome == VerificationOutcome.VERIFIED
        assert v.record is not None
    finally:
        j2.close()


# --- B1.2b audit-visibility correction ---

def test_unauthorized_record_for_ineligible_case(tmp_path):
    path = str(tmp_path / "j.db")
    j = ActionJournal(path)
    j.close()  # create schema

    ineligible = _case(supplier_state=SupplierMatchState.MISMATCH)
    _insert_record(path, ineligible)  # bypassed path: commit an ineligible case

    j = ActionJournal(path)
    try:
        v = verify(j, ineligible)
        assert v.outcome == VerificationOutcome.UNAUTHORIZED_RECORD
        assert v.record is not None
        assert v.record.input_fingerprint == case_fingerprint(ineligible)
    finally:
        j.close()


def test_changed_ineligible_input_is_conflict_not_unauthorized(tmp_path):
    j = ActionJournal(str(tmp_path / "j.db"))
    try:
        j.approve(_exact(case_id="C-1"))  # valid historical approval
        # later presented with changed, now-ineligible input (same case id)
        changed = _case(case_id="C-1", supplier_state=SupplierMatchState.MISMATCH)
        v = verify(j, changed)
        # a record exists, but it does not match this changed input: conflict,
        # not a claim that the historical execution was unauthorized.
        assert v.outcome == VerificationOutcome.RECORD_MISMATCH
        assert v.record is not None
    finally:
        j.close()
