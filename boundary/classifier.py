"""Deterministic exact-match classifier for BOUNDARY.

Given a validated :class:`~boundary.contracts.CaseInput`, produce a
:class:`DecisionProposal` — a structured, non-executing statement of whether the
trusted facts establish an unambiguous, conforming exact match between an invoice
and a purchase order.

A ``DecisionProposal`` is a *proposal*, never an accounting action or an
execution-authority grant (``execution_authority`` is fixed to ``"NONE"``). It
performs no inference, no authentication, and no approval: every reason code
traces to a deterministic fact derived from the contract fields. All arithmetic
is exact ``Decimal``/integer-cent work; no floating point is used.

Exact-match eligibility is proposed only when *all* of the following hold and no
reason code is raised:
  - the supplier lookup state is ``MATCH``;
  - the PO lookup state is ``FOUND``;
  - the invoice and PO reference the same supplier;
  - the invoice references the found PO (association established);
  - both documents' stated totals equal their computed line totals;
  - neither document has duplicate line identifiers;
  - the invoice line items conform exactly to the PO line items;
  - the invoice and PO totals are equal (zero discrepancy);
  - the case carries no unresolved evidence.

Any unresolved, contradictory, incomplete, or non-exact case yields a proposal
with ``exact_match_eligible == False`` and explicit reason codes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CaseInput, LineItem, SupplierMatchState


class ReasonCode(str, Enum):
    """Deterministic reason a case is not an exact match."""
    SUPPLIER_UNRESOLVED = "SUPPLIER_UNRESOLVED"
    SUPPLIER_MISMATCH = "SUPPLIER_MISMATCH"
    SUPPLIER_ID_INCONSISTENT = "SUPPLIER_ID_INCONSISTENT"
    PO_UNAVAILABLE = "PO_UNAVAILABLE"
    PO_NOT_FOUND = "PO_NOT_FOUND"
    PO_ASSOCIATION_MISSING = "PO_ASSOCIATION_MISSING"
    PO_ASSOCIATION_MISMATCH = "PO_ASSOCIATION_MISMATCH"
    INVOICE_ARITHMETIC_MISMATCH = "INVOICE_ARITHMETIC_MISMATCH"
    PO_ARITHMETIC_MISMATCH = "PO_ARITHMETIC_MISMATCH"
    DUPLICATE_LINE_IDENTIFIERS = "DUPLICATE_LINE_IDENTIFIERS"
    LINE_ITEMS_NONCONFORMING = "LINE_ITEMS_NONCONFORMING"
    NONZERO_DISCREPANCY = "NONZERO_DISCREPANCY"
    EVIDENCE_PENDING_REVIEW = "EVIDENCE_PENDING_REVIEW"


class Calculations(BaseModel):
    """Traceable integer-cent calculations backing a proposal."""
    model_config = ConfigDict(extra="forbid")

    invoice_stated_total_cents: int
    invoice_computed_total_cents: int
    po_stated_total_cents: Optional[int] = None
    po_computed_total_cents: Optional[int] = None
    discrepancy_cents: Optional[int] = None  # invoice_computed - po_computed; None w/o PO
    invoice_line_count: int
    po_line_count: Optional[int] = None


class DecisionProposal(BaseModel):
    """Structured, non-executing decision proposal."""
    model_config = ConfigDict(extra="forbid")

    case_id: str
    exact_match_eligible: bool
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    calculations: Calculations
    execution_authority: Literal["NONE"] = "NONE"  # a proposal is never an action grant


def _duplicate_line_numbers(items: list[LineItem]) -> bool:
    seen: set[int] = set()
    for item in items:
        if item.line_number in seen:
            return True
        seen.add(item.line_number)
    return False


def _conformity_key(item: LineItem) -> tuple:
    # Substantive identity of a line item (exact Decimal comparisons).
    return (item.description, item.quantity, item.unit_price)


def _line_items_conform(invoice_items: list[LineItem], po_items: list[LineItem]) -> bool:
    inv_keys = sorted(_conformity_key(i) for i in invoice_items)
    po_keys = sorted(_conformity_key(i) for i in po_items)
    return inv_keys == po_keys


def classify(case: CaseInput) -> DecisionProposal:
    """Classify a case into an exact-match proposal or a non-executing outcome.

    Reason codes are emitted in a fixed, deterministic order: supplier state,
    PO state, evidence, invoice arithmetic, duplicate line identifiers, then
    (only when a PO was retrieved) supplier identity, PO association, PO
    arithmetic, line conformity, and discrepancy. Invoice-internal checks run
    regardless of PO availability.
    """
    reasons: list[ReasonCode] = []

    # Supplier trusted lookup state (authoritative for supplier match).
    if case.supplier_lookup.state == SupplierMatchState.UNRESOLVED:
        reasons.append(ReasonCode.SUPPLIER_UNRESOLVED)
    elif case.supplier_lookup.state == SupplierMatchState.MISMATCH:
        reasons.append(ReasonCode.SUPPLIER_MISMATCH)

    # PO trusted lookup state.
    po = None
    if case.po_lookup.state == "UNAVAILABLE":
        reasons.append(ReasonCode.PO_UNAVAILABLE)
    elif case.po_lookup.state == "NOT_FOUND":
        reasons.append(ReasonCode.PO_NOT_FOUND)
    else:  # FOUND
        po = case.po_lookup.purchase_order

    # Unresolved evidence fails closed (opaque IDs are not authenticated).
    if case.evidence:
        reasons.append(ReasonCode.EVIDENCE_PENDING_REVIEW)

    # Invoice-internal checks (run regardless of PO availability).
    if case.invoice.stated_total_cents != case.invoice.computed_total_cents:
        reasons.append(ReasonCode.INVOICE_ARITHMETIC_MISMATCH)

    duplicate_lines = _duplicate_line_numbers(case.invoice.line_items)
    if po is not None:
        duplicate_lines = duplicate_lines or _duplicate_line_numbers(po.line_items)
    if duplicate_lines:
        reasons.append(ReasonCode.DUPLICATE_LINE_IDENTIFIERS)

    # PO-dependent checks (only meaningful when a PO was retrieved).
    if po is not None:
        if case.invoice.supplier_id != po.supplier_id:
            reasons.append(ReasonCode.SUPPLIER_ID_INCONSISTENT)

        if case.invoice.po_reference is None:
            reasons.append(ReasonCode.PO_ASSOCIATION_MISSING)
        elif case.invoice.po_reference != po.po_id:
            reasons.append(ReasonCode.PO_ASSOCIATION_MISMATCH)

        if po.stated_total_cents != po.computed_total_cents:
            reasons.append(ReasonCode.PO_ARITHMETIC_MISMATCH)

        if not _line_items_conform(case.invoice.line_items, po.line_items):
            reasons.append(ReasonCode.LINE_ITEMS_NONCONFORMING)

        if case.invoice.computed_total_cents != po.computed_total_cents:
            reasons.append(ReasonCode.NONZERO_DISCREPANCY)

    calculations = Calculations(
        invoice_stated_total_cents=case.invoice.stated_total_cents,
        invoice_computed_total_cents=case.invoice.computed_total_cents,
        po_stated_total_cents=po.stated_total_cents if po else None,
        po_computed_total_cents=po.computed_total_cents if po else None,
        discrepancy_cents=(case.invoice.computed_total_cents - po.computed_total_cents) if po else None,
        invoice_line_count=len(case.invoice.line_items),
        po_line_count=len(po.line_items) if po else None,
    )

    return DecisionProposal(
        case_id=case.case_id,
        exact_match_eligible=not reasons,
        reason_codes=reasons,
        calculations=calculations,
    )
