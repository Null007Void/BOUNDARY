"""Deterministic execution guard for BOUNDARY.

The guard evaluates an actual :class:`~boundary.contracts.CaseInput`, obtains
its classification *internally* (never trusting an externally supplied
eligibility flag or model assertion), and decides whether a simulated
``APPROVE`` action is permitted.

For this stage, ``APPROVE`` is permitted only for a fully valid exact-match
case: a classification with ``exact_match_eligible == True``, no reason codes,
and no unresolved evidence. Every other case is ``HOLD`` (non-executing); human
authorization for non-exact cases is a later stage (B2), not implemented here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from .classifier import DecisionProposal, classify
from .contracts import CaseInput


class ActionDecision(str, Enum):
    APPROVE = "APPROVE"
    HOLD = "HOLD"


class GuardDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    action: ActionDecision
    proposal: DecisionProposal


def evaluate(case: CaseInput) -> GuardDecision:
    """Classify internally and return the guarded action decision."""
    proposal = classify(case)

    approved = (
        proposal.exact_match_eligible
        and not proposal.reason_codes
        # Defensive re-assertion of the invariant: unresolved (opaque,
        # unauthenticated) evidence can never justify an approval.
        and not case.evidence
    )
    action = ActionDecision.APPROVE if approved else ActionDecision.HOLD
    return GuardDecision(case_id=case.case_id, action=action, proposal=proposal)
