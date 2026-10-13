"""Read-only independent verifier for BOUNDARY.

Establishes whether a simulated invoice approval actually committed to the SQLite
journal and whether the durable record agrees with the source case, the execution
constraints, and (optionally) the returned action receipt.

The verifier trusts neither a proposal, a guard decision, nor a claimed receipt:
it re-classifies the case internally and reads the journal directly. It is
strictly read-only — it never commits, authorizes, repairs, or deletes a record.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .classifier import DecisionProposal, classify
from .contracts import CaseInput
from .ledger import ActionJournal, ActionReceipt, case_fingerprint

# Fields compared field-by-field between a supplied receipt and the record.
_RECEIPT_FIELDS = (
    "action_id", "case_id", "action", "invoice_id", "po_id", "supplier_id",
    "amount_cents", "input_fingerprint", "policy_version", "committed_at",
)


class VerificationOutcome(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"            # ineligible case, no committed record
    NOT_FOUND = "NOT_FOUND"                  # eligible case, no committed record
    RECORD_MISMATCH = "RECORD_MISMATCH"      # record disagrees with the source case
    RECEIPT_MISMATCH = "RECEIPT_MISMATCH"    # supplied receipt disagrees with the record
    UNAUTHORIZED_RECORD = "UNAUTHORIZED_RECORD"  # ineligible case has a matching committed record


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: VerificationOutcome
    case_id: str
    reasons: list[str] = Field(default_factory=list)
    record: Optional[ActionReceipt] = None
    proposal: Optional[DecisionProposal] = None


def verify(
    journal: ActionJournal,
    case: CaseInput,
    receipt: Optional[ActionReceipt] = None,
) -> VerificationResult:
    """Independently verify a committed approval against its source case.

    Always reads the persisted journal first, so a committed record for an
    ineligible case is surfaced as an explicit audit failure rather than being
    indistinguishable from a case with no record.
    """
    proposal = classify(case)  # independent classification; no external flag trusted
    record = journal.get(case.case_id)  # read-only journal access, always performed

    if not proposal.exact_match_eligible:
        if record is None:
            return VerificationResult(
                outcome=VerificationOutcome.NOT_ELIGIBLE,
                case_id=case.case_id,
                reasons=["case is not an eligible exact match"],
                proposal=proposal,
            )
        # A committed record exists for an ineligible case.
        if record.input_fingerprint == case_fingerprint(case):
            # The ineligible case itself was committed through some (corrupted or
            # bypassed) path: explicit audit failure.
            return VerificationResult(
                outcome=VerificationOutcome.UNAUTHORIZED_RECORD,
                case_id=case.case_id,
                reasons=["a committed action exists for an ineligible case"],
                record=record,
                proposal=proposal,
            )
        # The record does not match the supplied (changed, now-ineligible) input.
        # This is a record/source conflict, not a claim that the historical
        # execution was unauthorized.
        return VerificationResult(
            outcome=VerificationOutcome.RECORD_MISMATCH,
            case_id=case.case_id,
            reasons=["persisted record disagrees with the supplied source case"],
            record=record,
            proposal=proposal,
        )

    if record is None:
        return VerificationResult(
            outcome=VerificationOutcome.NOT_FOUND,
            case_id=case.case_id,
            reasons=["no committed action for this case"],
            proposal=proposal,
        )

    # The persisted record must agree with the source case.
    expected = {
        "case_id": case.case_id,
        "action": "APPROVE",
        "invoice_id": case.invoice.invoice_id,
        "supplier_id": case.invoice.supplier_id,
        "po_id": case.po_lookup.purchase_order.po_id,
        "amount_cents": case.invoice.computed_total_cents,
        "input_fingerprint": case_fingerprint(case),
        "policy_version": case.policy_version,
    }
    mismatches = [name for name, value in expected.items() if getattr(record, name) != value]
    if mismatches:
        return VerificationResult(
            outcome=VerificationOutcome.RECORD_MISMATCH,
            case_id=case.case_id,
            reasons=[f"persisted {name} disagrees with the source case" for name in mismatches],
            record=record,
            proposal=proposal,
        )

    # A supplied receipt must agree field-by-field with the persisted record.
    if receipt is not None:
        receipt_mismatches = [
            f for f in _RECEIPT_FIELDS if getattr(receipt, f) != getattr(record, f)
        ]
        if receipt_mismatches:
            return VerificationResult(
                outcome=VerificationOutcome.RECEIPT_MISMATCH,
                case_id=case.case_id,
                reasons=[f"supplied receipt {f} disagrees with the persisted record"
                         for f in receipt_mismatches],
                record=record,
                proposal=proposal,
            )

    return VerificationResult(
        outcome=VerificationOutcome.VERIFIED,
        case_id=case.case_id,
        record=record,
        proposal=proposal,
    )
