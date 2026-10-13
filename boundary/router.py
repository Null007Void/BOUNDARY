"""Governed selective inference routing for BOUNDARY (B3.3).

Deterministic-first routing: a case is classified with the accepted exact-match
classifier *first*, and model inference is invoked only for a bounded, explicitly
identifiable category. The router never executes, approves, or writes to the
ledger — every decision carries ``execution_authority == "NONE"`` and is advisory
only.

Routing policy (in order):

1. **DETERMINISTIC** — an exact match is handled by the deterministic pipeline
   with zero model calls.
2. **HELD** — authoritative business facts are unresolved (supplier lookup not
   ``MATCH``, or PO lookup not ``FOUND``). The model cannot manufacture trusted
   PO, supplier, or payment evidence.
3. **HELD** — the case has a known structural defect (invoice/PO arithmetic
   mismatch or duplicate line identifiers), which is deterministically
   detectable and not a question for the model.
4. **INFERENCE** — the case has a nonzero discrepancy, consistent supplier
   identity and PO association, and actual evidence *document text*. Exactly one
   advisory model call is made, strictly validated against the source-case facts
   (including the trusted PO association), and recorded as non-authoritative
   evidence. Opaque evidence identifiers alone — without usable document text —
   stay held; an identifier is not document text and never justifies inference.
5. **HELD** — otherwise (no document text, adapter not configured, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from .classifier import DecisionProposal, ReasonCode, classify
from .contracts import CaseInput, POState, SupplierMatchState
from .inference import (
    EvidenceInterpretation,
    InferenceAdapter,
    InferenceOutcome,
)


class Route(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    HELD = "HELD"
    INFERENCE = "INFERENCE"


class HandlingStatus(str, Enum):
    ADVISORY_ONLY = "ADVISORY_ONLY"


class RoutingEvidence(BaseModel):
    """Durable, exportable record of the routing decision (B5-comparable)."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    model_calls: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    elapsed_ms: Optional[int] = None
    advisory_outcome: Optional[InferenceOutcome] = None


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    route: Route
    reason: str
    proposal: DecisionProposal
    evidence: RoutingEvidence
    interpretation: Optional[EvidenceInterpretation] = None
    advisory_consistent: Optional[bool] = None  # None unless inference attempted
    handling_status: HandlingStatus = HandlingStatus.ADVISORY_ONLY
    execution_authority: Literal["NONE"] = "NONE"


def _deterministic(case: CaseInput, proposal: DecisionProposal) -> RoutingDecision:
    reason = "exact match handled deterministically; zero model calls"
    return RoutingDecision(
        case_id=case.case_id,
        route=Route.DETERMINISTIC,
        reason=reason,
        proposal=proposal,
        evidence=RoutingEvidence(reason=reason, model_calls=0),
    )


def _held(case: CaseInput, proposal: DecisionProposal, reason: str) -> RoutingDecision:
    return RoutingDecision(
        case_id=case.case_id,
        route=Route.HELD,
        reason=reason,
        proposal=proposal,
        evidence=RoutingEvidence(reason=reason, model_calls=0),
    )


def _validate_against_facts(
    interpretation: EvidenceInterpretation, case: CaseInput
) -> tuple[bool, Optional[str]]:
    """An advisory claim must not override known amounts or identities."""
    po = case.po_lookup.purchase_order
    expected_amount = case.invoice.computed_total_cents
    expected_discrepancy = case.invoice.computed_total_cents - po.computed_total_cents

    if interpretation.invoice_id != case.invoice.invoice_id:
        return False, "model invoice_id contradicts source case"
    if interpretation.supplier_id != case.invoice.supplier_id:
        return False, "model supplier_id contradicts source case"
    if interpretation.amount_cents != expected_amount:
        return False, "model amount contradicts source case"
    if interpretation.discrepancy_cents != expected_discrepancy:
        return False, "model discrepancy contradicts source case"
    # An explicitly supplied PO reference must match the trusted association.
    if interpretation.po_reference is not None and interpretation.po_reference != po.po_id:
        return False, "model po_reference contradicts source case"
    return True, None


def route(
    case: CaseInput,
    evidence_text: Optional[str] = None,
    adapter: Optional[InferenceAdapter] = None,
) -> RoutingDecision:
    """Route a case to deterministic processing, hold, or bounded inference.

    ``evidence_text`` is actual synthetic evidence *document content*, distinct
    from the opaque identifiers in ``case.evidence``. ``adapter`` is an optional
    :class:`InferenceAdapter`; when the bounded inference category applies but no
    adapter is supplied, the case is held.
    """
    proposal = classify(case)

    # 1. Exact match -> deterministic, zero model calls.
    if proposal.exact_match_eligible:
        return _deterministic(case, proposal)

    # 2. Authoritative business facts unresolved -> held.
    if case.supplier_lookup.state != SupplierMatchState.MATCH:
        return _held(case, proposal, "supplier identity unresolved by trusted lookup")
    if case.po_lookup.state != POState.FOUND:
        return _held(case, proposal, "purchase order unresolved by trusted lookup")

    # 3. Known structural defects -> held (deterministically detectable, not a
    #    question to put to the model).
    if ReasonCode.INVOICE_ARITHMETIC_MISMATCH in proposal.reason_codes:
        return _held(case, proposal, "invoice arithmetic mismatch is a structural defect")
    if ReasonCode.PO_ARITHMETIC_MISMATCH in proposal.reason_codes:
        return _held(case, proposal, "purchase order arithmetic mismatch is a structural defect")
    if ReasonCode.DUPLICATE_LINE_IDENTIFIERS in proposal.reason_codes:
        return _held(case, proposal, "duplicate line identifiers are a structural defect")

    # 4. Bounded interpretation category.
    if ReasonCode.NONZERO_DISCREPANCY not in proposal.reason_codes:
        return _held(case, proposal, "no nonzero discrepancy requiring interpretation")
    if ReasonCode.SUPPLIER_ID_INCONSISTENT in proposal.reason_codes:
        return _held(case, proposal, "supplier identity inconsistent between invoice and PO")
    if (
        ReasonCode.PO_ASSOCIATION_MISSING in proposal.reason_codes
        or ReasonCode.PO_ASSOCIATION_MISMATCH in proposal.reason_codes
    ):
        return _held(case, proposal, "purchase order association unresolved")
    if not evidence_text or not evidence_text.strip():
        # Opaque evidence identifiers alone, with no usable document text, stay held.
        return _held(case, proposal, "no evidence document text supplied")
    if adapter is None:
        return _held(case, proposal, "inference adapter not configured")

    # 5. Exactly one advisory model call.
    result = adapter.interpret(evidence_text)
    telemetry = result.telemetry
    # Preflight rejection (INPUT_TOO_LARGE) makes zero outbound model attempts.
    model_calls = 0 if result.outcome == InferenceOutcome.INPUT_TOO_LARGE else 1
    evidence = RoutingEvidence(
        reason="nonzero discrepancy interpreted via advisory model",
        model_calls=model_calls,
        prompt_tokens=telemetry.prompt_tokens if model_calls else None,
        completion_tokens=telemetry.completion_tokens if model_calls else None,
        total_tokens=telemetry.total_tokens if model_calls else None,
        elapsed_ms=telemetry.elapsed_ms if model_calls else None,
        advisory_outcome=result.outcome,
    )

    # Fail closed on any non-success or missing interpretation.
    if result.outcome != InferenceOutcome.SUCCESS or result.interpretation is None:
        return RoutingDecision(
            case_id=case.case_id,
            route=Route.INFERENCE,
            reason="model output failed or was invalid",
            proposal=proposal,
            evidence=evidence,
            advisory_consistent=False,
        )

    # Strictly validate the advisory interpretation against source-case facts.
    consistent, contradiction = _validate_against_facts(result.interpretation, case)
    if not consistent:
        return RoutingDecision(
            case_id=case.case_id,
            route=Route.INFERENCE,
            reason=contradiction or "model interpretation contradicts source case",
            proposal=proposal,
            evidence=evidence,
            interpretation=result.interpretation,
            advisory_consistent=False,
        )

    return RoutingDecision(
        case_id=case.case_id,
        route=Route.INFERENCE,
        reason="advisory interpretation consistent with source facts",
        proposal=proposal,
        evidence=evidence,
        interpretation=result.interpretation,
        advisory_consistent=True,
    )
