"""S4E.25 — interactive synthetic invoice intake.

Lets a judge supply an invoice (reference, supplier, PO reference, USD amount) and have
BOUNDARY run the *accepted deterministic pipeline* against it, using clearly declared
synthetic purchase-order/supplier fixtures. This is a real vertical slice — the same
``classify``/``evaluate``/routing rules as every other BOUNDARY path — never a toy
decision engine that switches between hardcoded outcomes.

Safety and honesty (fixed here, not optional):
- **No provider calls.** The intake never contacts AMD or Evolus, and never invokes the
  offline mock adapter. A case that would ordinarily require advisory inference is shown
  as *required but unavailable in public offline mode* — a model response is never
  fabricated and historical inference is never implied to be "just executed".
- **Authority NONE.** The intake only classifies and holds; it never writes the journal,
  never approves, and never mutates any financial system. ``execution_authority`` stays
  ``NONE``.
- **No persistence.** Visitor submissions are evaluated in-memory and never stored; no
  cross-visitor state exists.
- **Bounded input.** Field length, numeric precision, and amount magnitude are capped.

The PO/supplier evidence comes only from ``SYNTHETIC_PO`` / ``SYNTHETIC_SUPPLIERS`` —
it is never manufactured from the submitted invoice amount.
"""

from __future__ import annotations

import re
from decimal import Decimal

from .classifier import ReasonCode, classify
from .contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupNotFound,
    POState,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from .guard import ActionDecision, evaluate

# --- Declared synthetic fixtures (never derived from submitted values) ----------------

SYNTHETIC_PO = PurchaseOrder(
    po_id="PO-SYN-100",
    supplier_id="SUP-SYN-100",
    issued_date="2026-10-01",
    line_items=[LineItem(line_number=1, description="Synthetic widget",
                         quantity="1", unit_price="1000.00")],
    stated_total="1000.00",
)

SYNTHETIC_SUPPLIERS = frozenset({"SUP-SYN-100"})

# --- Input bounds (resource + customer safety) ----------------------------------------

MAX_FIELD_LEN = 100
MIN_AMOUNT = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000000")  # $1B — engineering bound, not business policy


class IntakeError(ValueError):
    """A judge-supplied input failed validation; safe to display to the visitor."""


def _parse_field(raw: str, name: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise IntakeError(f"{name} is required")
    if len(value) > MAX_FIELD_LEN:
        raise IntakeError(f"{name} must be at most {MAX_FIELD_LEN} characters")
    return value


_AMOUNT_RE = re.compile(r"^\d+(?:\.\d{1,2})?$")


def _parse_amount(raw: str) -> Decimal:
    value = (raw or "").strip()
    # Require a conventional unsigned USD decimal string (digits, optional '.' plus one or
    # two fractional digits) BEFORE any numeric interpretation, so "1000.000" and "1E+3" are
    # rejected rather than silently equal to "1000.00".
    if not value.isascii() or not _AMOUNT_RE.fullmatch(value):
        raise IntakeError(
            "amount must be a plain USD amount like 1000.00 "
            "(at most 2 decimal places, no sign, no scientific notation)")
    amount = Decimal(value)
    if amount < MIN_AMOUNT or amount > MAX_AMOUNT:
        raise IntakeError(f"amount must be between {MIN_AMOUNT} and {MAX_AMOUNT}")
    return amount


# --- Case construction -----------------------------------------------------------------


def build_case(
    invoice_id: str,
    supplier_id: str,
    po_reference: str,
    amount: str,
    case_id: str = "INTAKE",
) -> CaseInput:
    """Build a validated ``CaseInput`` from judge fields against the synthetic fixtures.

    The invoice has a single synthetic line item whose unit price equals the submitted
    amount, so the invoice's stated total always equals its computed total (no internal
    arithmetic mismatch) and any discrepancy is against the synthetic PO — matching the
    accepted demo-case shape.
    """
    inv_ref = _parse_field(invoice_id, "invoice reference")
    sup = _parse_field(supplier_id, "supplier")
    po_ref = _parse_field(po_reference, "purchase-order reference")
    parsed_amount = _parse_amount(amount)

    invoice = Invoice(
        invoice_id=inv_ref,
        supplier_id=sup,
        po_reference=po_ref,
        issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="Synthetic widget",
                             quantity="1", unit_price=f"{parsed_amount}")],
        stated_total=f"{parsed_amount}",
    )

    supplier_lookup = SupplierLookupResult(
        state=SupplierMatchState.MATCH if sup in SYNTHETIC_SUPPLIERS
        else SupplierMatchState.MISMATCH,
    )

    if po_ref in (SYNTHETIC_PO.po_id,):
        po_lookup = POLookupFound(purchase_order=SYNTHETIC_PO)
    else:
        po_lookup = POLookupNotFound()

    return CaseInput(
        case_id=case_id,
        invoice=invoice,
        po_lookup=po_lookup,
        supplier_lookup=supplier_lookup,
        policy_version="policy-v1",
        input_identity="public-synthetic-intake",
    )


def _would_require_inference(proposal, case: CaseInput) -> bool:
    """True when the case falls in the bounded advisory-inference category.

    Mirrors ``router.route``'s INFERENCE gating (nonzero discrepancy, supplier MATCH, PO
    FOUND, association intact, no structural defect) — but does NOT invoke a model, so the
    intake can honestly report "required but unavailable" instead of fabricating a response.
    """
    rc = proposal.reason_codes
    return (
        ReasonCode.NONZERO_DISCREPANCY in rc
        and ReasonCode.INVOICE_ARITHMETIC_MISMATCH not in rc
        and ReasonCode.PO_ARITHMETIC_MISMATCH not in rc
        and ReasonCode.DUPLICATE_LINE_IDENTIFIERS not in rc
        and ReasonCode.SUPPLIER_ID_INCONSISTENT not in rc
        and ReasonCode.PO_ASSOCIATION_MISSING not in rc
        and ReasonCode.PO_ASSOCIATION_MISMATCH not in rc
        and case.supplier_lookup.state == SupplierMatchState.MATCH
        and case.po_lookup.state == POState.FOUND
    )


def evaluate_intake(case: CaseInput) -> dict:
    """Run the accepted pipeline and return a non-persisted, read-only evaluation."""
    proposal = classify(case)
    guard = evaluate(case)

    if proposal.exact_match_eligible:
        route_label = "DETERMINISTIC"
        inference = {
            "required": False,
            "note": "handled deterministically — zero model calls",
        }
    elif _would_require_inference(proposal, case):
        route_label = "INFERENCE"
        inference = {
            "required": True,
            "note": "advisory inference required but UNAVAILABLE in public offline mode — "
                    "no model response is fabricated",
        }
    else:
        route_label = "HELD"
        inference = {
            "required": False,
            "note": "held for deterministic reasons — no model call",
        }

    return {
        "case": case,
        "proposal": proposal,
        "guard": guard,
        "route": route_label,
        "inference": inference,
        "execution_authority": "NONE",
    }
