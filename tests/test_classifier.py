import pytest
from pydantic import ValidationError

from boundary.classifier import DecisionProposal, ReasonCode, classify
from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupNotFound,
    POLookupUnavailable,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)


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


def test_ordinary_exact_match():
    p = classify(_case())
    assert p.exact_match_eligible is True
    assert p.reason_codes == []
    assert p.execution_authority == "NONE"
    assert p.calculations.invoice_computed_total_cents == 1000
    assert p.calculations.discrepancy_cents == 0


def test_high_value_exact_match():
    lines = [_line(1, "1000", "999.99"), _line(2, "500", "1.01")]
    inv = _invoice(inv_id="INV-HIGH", lines=lines, stated="1000495.00")
    po = _po(po_id="PO-1", lines=lines, stated="1000495.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
    assert p.exact_match_eligible is True
    assert p.calculations.invoice_computed_total_cents == 100049500


def test_supplier_mismatch():
    p = classify(_case(supplier_state=SupplierMatchState.MISMATCH))
    assert p.exact_match_eligible is False
    assert ReasonCode.SUPPLIER_MISMATCH in p.reason_codes


def test_supplier_unresolved():
    p = classify(_case(supplier_state=SupplierMatchState.UNRESOLVED))
    assert p.exact_match_eligible is False
    assert ReasonCode.SUPPLIER_UNRESOLVED in p.reason_codes


def test_po_unavailable():
    p = classify(_case(po_lookup=POLookupUnavailable()))
    assert p.exact_match_eligible is False
    assert ReasonCode.PO_UNAVAILABLE in p.reason_codes
    assert p.calculations.po_computed_total_cents is None
    assert p.calculations.discrepancy_cents is None


def test_po_not_found():
    p = classify(_case(po_lookup=POLookupNotFound()))
    assert p.exact_match_eligible is False
    assert ReasonCode.PO_NOT_FOUND in p.reason_codes


def test_invoice_arithmetic_mismatch():
    inv = _invoice(inv_id="INV-ARITH", lines=[_line(1, "3", "10.00")], stated="100.00")
    p = classify(_case(invoice=inv))
    assert p.exact_match_eligible is False
    assert ReasonCode.INVOICE_ARITHMETIC_MISMATCH in p.reason_codes


def test_nonconforming_line_items():
    inv = _invoice(lines=[_line(1, "1", "10.00")], stated="10.00")
    po = _po(lines=[_line(1, "2", "10.00")], stated="20.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
    assert ReasonCode.LINE_ITEMS_NONCONFORMING in p.reason_codes
    assert p.exact_match_eligible is False


def test_duplicate_line_identifiers():
    inv_lines = [_line(1, "1", "5.00"), _line(1, "1", "5.00")]  # duplicate line_number 1
    inv = _invoice(lines=inv_lines, stated="10.00")
    po = _po(lines=[_line(1, "1", "10.00")], stated="10.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
    assert ReasonCode.DUPLICATE_LINE_IDENTIFIERS in p.reason_codes
    assert p.exact_match_eligible is False


def test_nonzero_discrepancy_not_exact():
    inv = _invoice(lines=[_line(1, "1", "10.01")], stated="10.01")
    po = _po(lines=[_line(1, "1", "10.00")], stated="10.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
    assert ReasonCode.NONZERO_DISCREPANCY in p.reason_codes
    assert p.calculations.discrepancy_cents == 1
    assert p.exact_match_eligible is False


def test_po_association_mismatch():
    inv = _invoice(po_ref="PO-9")
    p = classify(_case(invoice=inv))
    assert ReasonCode.PO_ASSOCIATION_MISMATCH in p.reason_codes
    assert p.exact_match_eligible is False


def test_po_association_missing():
    inv = _invoice(po_ref=None)
    p = classify(_case(invoice=inv))
    assert ReasonCode.PO_ASSOCIATION_MISSING in p.reason_codes
    assert p.exact_match_eligible is False


def test_supplier_id_inconsistent():
    inv = _invoice(supplier="SUP-A")
    po = _po(supplier="SUP-B")
    p = classify(_case(invoice=inv, po_lookup=POLookupFound(purchase_order=po)))
    assert ReasonCode.SUPPLIER_ID_INCONSISTENT in p.reason_codes
    assert p.exact_match_eligible is False


def test_unresolved_evidence_fails_closed():
    p = classify(_case(evidence=["ev-1"]))
    assert p.exact_match_eligible is False
    assert ReasonCode.EVIDENCE_PENDING_REVIEW in p.reason_codes


def test_empty_evidence_retains_exact_match():
    p = classify(_case(evidence=[]))
    assert p.exact_match_eligible is True
    assert p.reason_codes == []


def test_invoice_arithmetic_checked_without_po():
    inv = _invoice(lines=[_line(1, "3", "10.00")], stated="100.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupUnavailable()))
    assert ReasonCode.PO_UNAVAILABLE in p.reason_codes
    assert ReasonCode.INVOICE_ARITHMETIC_MISMATCH in p.reason_codes
    assert p.exact_match_eligible is False


def test_invoice_duplicate_lines_checked_without_po():
    inv_lines = [_line(1, "1", "5.00"), _line(1, "1", "5.00")]
    inv = _invoice(lines=inv_lines, stated="10.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupNotFound()))
    assert ReasonCode.PO_NOT_FOUND in p.reason_codes
    assert ReasonCode.DUPLICATE_LINE_IDENTIFIERS in p.reason_codes
    assert p.exact_match_eligible is False


def test_reason_code_ordering_deterministic():
    inv = _invoice(lines=[_line(1, "3", "10.00")], stated="100.00")
    p = classify(_case(invoice=inv, po_lookup=POLookupUnavailable(), evidence=["ev-1"]))
    assert p.reason_codes == [
        ReasonCode.PO_UNAVAILABLE,
        ReasonCode.EVIDENCE_PENDING_REVIEW,
        ReasonCode.INVOICE_ARITHMETIC_MISMATCH,
    ]
