import pytest
from pydantic import ValidationError

from boundary.contracts import (
    CaseInput,
    Invoice,
    LedgerSnapshot,
    LineItem,
    POState,
    POLookupFound,
    POLookupNotFound,
    POLookupUnavailable,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _invoice(inv_id="INV-1", supplier="SUP-1", lines=None, stated="10.00", issued="2026-10-01"):
    lines = lines or [_line(1, "1", "10.00")]
    return Invoice(invoice_id=inv_id, supplier_id=supplier, issued_date=issued,
                   line_items=lines, stated_total=stated)


def _po(po_id="PO-1", supplier="SUP-1", lines=None, stated="10.00", issued="2026-10-01"):
    lines = lines or [_line(1, "1", "10.00")]
    return PurchaseOrder(po_id=po_id, supplier_id=supplier, issued_date=issued,
                         line_items=lines, stated_total=stated)


def _case(case_id="C-1", invoice=None, po_lookup=None, supplier_lookup=None, **kw):
    return CaseInput(
        case_id=case_id,
        invoice=invoice or _invoice(),
        po_lookup=po_lookup or POLookupFound(purchase_order=_po()),
        supplier_lookup=supplier_lookup or SupplierLookupResult(state=SupplierMatchState.MATCH),
        policy_version="policy-v1",
        input_identity="fixture-builder",
        **kw,
    )


def test_valid_exact_match_case():
    c = _case()
    assert c.case_id == "C-1"
    assert c.po_lookup.state == "FOUND"
    assert c.supplier_lookup.state == SupplierMatchState.MATCH
    assert c.invoice.computed_total_cents == 1000


def test_valid_high_value_exact_match():
    lines = [_line(1, "1000", "999.99"), _line(2, "500", "1.01")]
    inv = _invoice(inv_id="INV-HIGH", lines=lines, stated="1000495.00")
    assert inv.computed_total_cents == 99999000 + 50500
    c = _case(invoice=inv)
    assert c.invoice.computed_total_cents == 100049500


def test_computed_total_deterministic_and_discrepancy_observable():
    # stated total does not match the line sum; the contract preserves the
    # stated value and computes independently — detection is a later concern
    inv = _invoice(inv_id="INV-ARITH", lines=[_line(1, "3", "10.00")], stated="100.00")
    assert inv.stated_total_cents == 10000
    assert inv.computed_total_cents == 3000
    assert inv.stated_total_cents != inv.computed_total_cents


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError):
        CaseInput(  # missing case_id
            invoice=_invoice(),
            po_lookup=POLookupFound(purchase_order=_po()),
            supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
            policy_version="v1",
            input_identity="x",
        )
    with pytest.raises(ValidationError):
        Invoice(invoice_id="INV-X", supplier_id="SUP-1", line_items=[], stated_total="1.00")
    with pytest.raises(ValidationError):
        _line(1, "0", "10.00")     # zero quantity
    with pytest.raises(ValidationError):
        _line(1, "-2", "10.00")    # negative quantity


def test_po_unavailable_distinct_from_not_found():
    unavail = POLookupUnavailable()
    notfound = POLookupNotFound()
    assert unavail.state == POState.UNAVAILABLE
    assert notfound.state == POState.NOT_FOUND
    assert unavail.state != notfound.state
    assert not hasattr(unavail, "purchase_order")
    assert not hasattr(notfound, "purchase_order")


def test_supplier_unresolved_distinct_from_mismatch():
    unresolved = SupplierLookupResult(state=SupplierMatchState.UNRESOLVED)
    mismatch = SupplierLookupResult(state=SupplierMatchState.MISMATCH)
    assert unresolved.state != mismatch.state
    assert unresolved.state == SupplierMatchState.UNRESOLVED
    assert mismatch.state == SupplierMatchState.MISMATCH


def test_monetary_boundary_handling():
    with pytest.raises(ValidationError):
        _line(1, "1", 1.50)        # float unit_price forbidden
    with pytest.raises(ValidationError):
        _line(1, "1", "1.999")     # sub-cent precision forbidden
    with pytest.raises(ValidationError):
        _line(1, 1.5, "10.00")     # float quantity forbidden


def test_line_total_rounding_half_up():
    assert _line(1, "3", "0.01").line_total_cents == 3
    assert _line(1, "1.5", "10.00").line_total_cents == 1500
    assert _line(1, "1.5", "0.01").line_total_cents == 2    # 0.015 -> 0.02


def test_no_ground_truth_labels_at_runtime():
    for label in ["expected_outcome", "scenario_family", "split", "label", "ground_truth"]:
        with pytest.raises(ValidationError):
            _case(**{label: "LEAK"})


def test_opaque_case_id_not_parsed():
    c = _case(case_id="a/b/c|whatever$123")
    assert c.case_id == "a/b/c|whatever$123"


def test_starting_ledger_optional_and_cents():
    ledger = LedgerSnapshot(balances={"acct-1": 500, "acct-2": -25})
    c = _case(starting_ledger=ledger)
    assert c.starting_ledger.balances["acct-1"] == 500
    assert c.starting_ledger.balances["acct-2"] == -25
    assert _case().starting_ledger is None


# --- B1.1a corrective-pass regressions ---

def test_line_number_strict_int():
    for bad in [1.0, True, "1"]:
        with pytest.raises(ValidationError):
            LineItem(line_number=bad, description="x", quantity="1", unit_price="1.00")


def test_ledger_balances_strict_int():
    assert LedgerSnapshot(balances={"a": 5, "b": -1}).balances["a"] == 5
    for bad in [1.0, True, "5"]:
        with pytest.raises(ValidationError):
            LedgerSnapshot(balances={"a": bad})


def test_ledger_balances_strict_int_json():
    assert LedgerSnapshot.model_validate_json('{"balances": {"a": 5}}').balances["a"] == 5
    with pytest.raises(ValidationError):
        LedgerSnapshot.model_validate_json('{"balances": {"a": 5.0}}')
    with pytest.raises(ValidationError):
        LedgerSnapshot.model_validate_json('{"balances": {"a": "5"}}')


def test_currency_restricted_to_usd():
    inv = Invoice(invoice_id="I", supplier_id="S", issued_date="2026-01-01",
                  line_items=[_line(1, "1", "1.00")], stated_total="1.00")
    assert inv.currency == "USD"
    with pytest.raises(ValidationError):
        Invoice(invoice_id="I", supplier_id="S", issued_date="2026-01-01",
                line_items=[_line(1, "1", "1.00")], stated_total="1.00", currency="EUR")
    with pytest.raises(ValidationError):
        PurchaseOrder(po_id="P", supplier_id="S", issued_date="2026-01-01",
                      line_items=[_line(1, "1", "1.00")], stated_total="1.00", currency="GBP")


def test_whitespace_only_identifiers_rejected():
    with pytest.raises(ValidationError):
        _case(case_id="   ")
    with pytest.raises(ValidationError):
        Invoice(invoice_id=" ", supplier_id="S", issued_date="2026-01-01",
                line_items=[_line(1, "1", "1.00")], stated_total="1.00")
    with pytest.raises(ValidationError):
        _line(1, "1", "1.00", desc="\t")


def test_opaque_identifier_content_preserved_not_stripped():
    c = _case(case_id="  C-7  ")
    assert c.case_id == "  C-7  "


def test_exact_monetary_arithmetic_no_precision_loss():
    assert _line(1, "1.00499999999999999999999999999", "1.00").line_total_cents == 100


def test_quantity_magnitude_fails_controlled():
    huge = "1" + "0" * 45  # 46 significant digits > MAX_QUANTITY_DIGITS
    with pytest.raises(ValidationError):
        _line(1, huge, "1.00")


def test_quantity_exponential_magnitude_rejected():
    with pytest.raises(ValidationError):
        _line(1, "1E+29", "1.00")


def test_quantity_accepted_large_value_works():
    assert _line(1, "1E+20", "1.00").line_total_cents == 10 ** 22
