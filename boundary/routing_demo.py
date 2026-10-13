"""Offline B3 routing demonstration for the Signal Prism interface (B6 S2).

Computes real ``RoutingDecision`` results through the accepted ``router.route``
using the *actual* ``CaseInput`` the web route renders, with an OFFLINE injected
inference adapter. No live AMD or Evolus request is ever made.

The demonstration is bound to the case object passed in — never a cached copy — so
a changed case (same identifier, different facts) produces a routing result that
matches the displayed source facts. The advisory interpretation fixture is derived
from the case's own facts, so it cannot be presented as source-consistent when the
case facts differ. The authority-claim example is returned only when the accepted
local validator actually produced ``AUTHORITY_CLAIM`` for that case.

B1 journal actions and B3 advisory routing are strictly separate: the B3 route is
advisory-only (``execution_authority == "NONE"``) and never changes the B1 trace.
"""

from __future__ import annotations

import json
from typing import Optional

from .contracts import CaseInput
from .inference import InferenceAdapter, InferenceConfig, InferenceOutcome
from .router import route


def _evidence_text(case: CaseInput) -> str:
    """A synthetic evidence document derived from the ACTUAL case facts."""
    inv = case.invoice
    text = (f"Invoice {inv.invoice_id} from supplier {inv.supplier_id} "
            f"with a stated total of {inv.stated_total}.")
    if case.po_lookup.state == "FOUND":
        po = case.po_lookup.purchase_order
        text += f" The referenced purchase order {po.po_id} states {po.stated_total}."
    return text


def _interpretation_content(case: CaseInput, authority: bool = False) -> str:
    """A mock model interpretation derived from the case's own trusted facts."""
    po = case.po_lookup.purchase_order
    data = {
        "supplier_id": case.invoice.supplier_id,
        "invoice_id": case.invoice.invoice_id,
        "po_reference": po.po_id,
        "amount_cents": case.invoice.computed_total_cents,
        "discrepancy_cents": case.invoice.computed_total_cents - po.computed_total_cents,
        "confidence": 0.9,
        "notes": "",
    }
    if authority:
        data["approve"] = True
    return json.dumps(data)


def _offline_adapter_for(case: CaseInput, authority: bool = False) -> InferenceAdapter:
    """A local mock adapter; content is computed lazily (only when inference runs)."""

    def transport(payload):
        return {"choices": [{"message": {"content": _interpretation_content(case, authority)}}]}

    return InferenceAdapter(InferenceConfig(model_id="DeepSeek-V4-Flash"), transport)


def _summarize(decision, case_id: str) -> dict:
    ev = decision.evidence
    interp = decision.interpretation
    return {
        "case_id": case_id,
        "route": decision.route.value,
        "reason": decision.reason,
        "model_calls": ev.model_calls,
        "advisory_outcome": ev.advisory_outcome.value if ev.advisory_outcome else None,
        "advisory_consistent": decision.advisory_consistent,
        "execution_authority": decision.execution_authority,
        "interpretation": (
            {
                "supplier_id": interp.supplier_id,
                "invoice_id": interp.invoice_id,
                "po_reference": interp.po_reference,
                "amount_cents": interp.amount_cents,
                "discrepancy_cents": interp.discrepancy_cents,
                "confidence": interp.confidence,
            }
            if interp else None
        ),
    }


def routing_for_case(case: CaseInput) -> dict:
    """Return the B3 routing demonstration for the ACTUAL rendered case.

    The router itself decides DETERMINISTIC / HELD / INFERENCE from the case facts;
    the offline adapter is invoked only when the case genuinely routes to inference.
    """
    decision = route(case, evidence_text=_evidence_text(case),
                     adapter=_offline_adapter_for(case))
    return _summarize(decision, case.case_id)


def authority_claim_rejection(case: CaseInput) -> Optional[dict]:
    """Return the AUTHORITY_CLAIM rejection demo, or None when not applicable.

    Returns a demo only when the accepted local validator actually produced
    ``AUTHORITY_CLAIM`` for this case (i.e. the case routes to inference and the
    injected authority assertion is rejected). A deterministic or held case returns
    None — there is no model invocation to reject.
    """
    decision = route(case, evidence_text=_evidence_text(case),
                     adapter=_offline_adapter_for(case, authority=True))
    if decision.evidence.advisory_outcome == InferenceOutcome.AUTHORITY_CLAIM:
        return _summarize(decision, case.case_id)
    return None
