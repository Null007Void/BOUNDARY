"""B3.3 governed selective inference routing tests."""

import pytest

from boundary.classifier import ReasonCode
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
from boundary.inference import (
    EvidenceInterpretation,
    InferenceOutcome,
    InferenceResult,
    InferenceTelemetry,
)
from boundary.router import HandlingStatus, Route, RoutingDecision, route


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _invoice(inv_id="INV-1", supplier="SUP-1", po_ref="PO-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return Invoice(
        invoice_id=inv_id,
        supplier_id=supplier,
        po_reference=po_ref,
        issued_date="2026-10-01",
        line_items=lines,
        stated_total=stated,
    )


def _po(po_id="PO-1", supplier="SUP-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return PurchaseOrder(
        po_id=po_id,
        supplier_id=supplier,
        issued_date="2026-10-01",
        line_items=lines,
        stated_total=stated,
    )


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


def _discrepancy_case(evidence=None):
    # Invoice $1250.00 vs PO $1245.00 -> 500-cent ($5.00) discrepancy.
    inv = _invoice(
        inv_id="INV-2048",
        supplier="SUP-9001",
        po_ref="PO-77",
        lines=[_line(1, "1", "1250.00")],
        stated="1250.00",
    )
    po = _po(po_id="PO-77", supplier="SUP-9001", lines=[_line(1, "1", "1245.00")], stated="1245.00")
    return _case(invoice=inv, po_lookup=POLookupFound(purchase_order=po), evidence=evidence)


def _interpretation(**overrides):
    data = dict(
        supplier_id="SUP-9001",
        invoice_id="INV-2048",
        po_reference="PO-77",
        amount_cents=125000,
        discrepancy_cents=500,
        confidence=0.6,
        notes="",
    )
    data.update(overrides)
    return EvidenceInterpretation(**data)


def _telemetry(outcome=InferenceOutcome.SUCCESS, total=30):
    return InferenceTelemetry(
        model_id="DeepSeek-V4-Flash",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=total,
        elapsed_ms=123,
        outcome=outcome,
    )


def _result(outcome=InferenceOutcome.SUCCESS, interpretation=None):
    return InferenceResult(
        outcome=outcome,
        interpretation=interpretation,
        telemetry=_telemetry(outcome),
        reasons=[],
    )


class _StubAdapter:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def interpret(self, evidence):
        self.calls += 1
        return self._result


# --- Deterministic bypass ---------------------------------------------------------


def test_exact_match_routes_deterministic():
    adapter = _StubAdapter(None)
    decision = route(_case(), adapter=adapter)
    assert decision.route == Route.DETERMINISTIC
    assert decision.evidence.model_calls == 0
    assert adapter.calls == 0
    assert decision.execution_authority == "NONE"
    assert decision.handling_status == HandlingStatus.ADVISORY_ONLY


# --- Selective inference invocation ------------------------------------------------


def test_discrepancy_with_text_routes_inference():
    adapter = _StubAdapter(_result(interpretation=_interpretation()))
    decision = route(_discrepancy_case(), evidence_text="ambiguous evidence text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert adapter.calls == 1
    assert decision.evidence.model_calls == 1
    assert decision.advisory_consistent is True
    assert decision.execution_authority == "NONE"


def test_consistent_interpretation_records_tokens_and_facts():
    adapter = _StubAdapter(_result(interpretation=_interpretation()))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.evidence.total_tokens == 30
    assert decision.evidence.prompt_tokens == 10
    assert decision.evidence.elapsed_ms == 123
    assert decision.interpretation.amount_cents == 125000
    assert decision.interpretation.discrepancy_cents == 500


# --- Authoritative facts unresolved -> held, zero LLM ------------------------------


@pytest.mark.parametrize("state", [SupplierMatchState.UNRESOLVED, SupplierMatchState.MISMATCH])
def test_supplier_unresolved_held(state):
    adapter = _StubAdapter(None)
    decision = route(_case(supplier_state=state), evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0
    assert decision.evidence.model_calls == 0


def test_po_unavailable_held():
    adapter = _StubAdapter(None)
    decision = route(_case(po_lookup=POLookupUnavailable()), evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


def test_po_not_found_held():
    adapter = _StubAdapter(None)
    decision = route(_case(po_lookup=POLookupNotFound()), evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


# --- Opaque evidence IDs are not document text -------------------------------------


def test_opaque_evidence_ids_with_document_text_routes_inference():
    adapter = _StubAdapter(_result(interpretation=_interpretation()))
    decision = route(_discrepancy_case(evidence=["ev-1"]), evidence_text="actual document text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert adapter.calls == 1
    assert decision.advisory_consistent is True


def test_opaque_evidence_ids_alone_held():
    adapter = _StubAdapter(None)
    decision = route(_discrepancy_case(evidence=["ev-1"]), evidence_text=None, adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0
    assert decision.evidence.model_calls == 0


def test_no_document_text_held():
    adapter = _StubAdapter(None)
    decision = route(_discrepancy_case(), evidence_text=None, adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


def test_no_adapter_held():
    decision = route(_discrepancy_case(), evidence_text="text", adapter=None)
    assert decision.route == Route.HELD
    assert decision.evidence.model_calls == 0


def test_no_discrepancy_held():
    # Exact-match inputs plus a document text but no discrepancy: still deterministic.
    adapter = _StubAdapter(None)
    decision = route(_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.DETERMINISTIC
    assert adapter.calls == 0


def test_supplier_id_inconsistent_held():
    inv = _invoice(supplier="SUP-A", lines=[_line(1, "1", "10.05")], stated="10.05")
    po = _po(supplier="SUP-B", lines=[_line(1, "1", "10.00")], stated="10.00")
    case = _case(invoice=inv, po_lookup=POLookupFound(purchase_order=po))
    adapter = _StubAdapter(None)
    decision = route(case, evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


# --- Fail closed on invalid / contradictory / unavailable --------------------------


def test_malformed_output_fails_closed():
    adapter = _StubAdapter(_result(outcome=InferenceOutcome.MALFORMED_RESPONSE))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert decision.advisory_consistent is False
    assert decision.interpretation is None
    assert decision.evidence.advisory_outcome == InferenceOutcome.MALFORMED_RESPONSE
    assert decision.execution_authority == "NONE"


def test_unavailable_service_fails_closed():
    adapter = _StubAdapter(_result(outcome=InferenceOutcome.UNAVAILABLE))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert decision.advisory_consistent is False
    assert decision.evidence.advisory_outcome == InferenceOutcome.UNAVAILABLE


@pytest.mark.parametrize(
    "override,field",
    [
        ({"invoice_id": "INV-WRONG"}, "invoice_id"),
        ({"supplier_id": "SUP-WRONG"}, "supplier_id"),
        ({"amount_cents": 999999}, "amount"),
        ({"discrepancy_cents": 0}, "discrepancy"),
    ],
)
def test_source_fact_contradiction_fails_closed(override, field):
    adapter = _StubAdapter(_result(interpretation=_interpretation(**override)))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert decision.advisory_consistent is False
    assert field in decision.reason
    assert decision.execution_authority == "NONE"


# --- Exportable routing evidence ---------------------------------------------------


def test_routing_decision_is_exportable():
    adapter = _StubAdapter(_result(interpretation=_interpretation()))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    dumped = decision.model_dump_json()
    assert isinstance(dumped, str)
    assert "model_calls" in dumped
    assert "execution_authority" in dumped


# --- Finding 2: PO-reference contradiction -----------------------------------------


def test_po_reference_contradiction_fails_closed():
    adapter = _StubAdapter(_result(interpretation=_interpretation(po_reference="PO-OTHER")))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert decision.advisory_consistent is False
    assert "po_reference" in decision.reason


def test_null_po_reference_preserved():
    # Null po_reference is permitted by the optional-field contract.
    adapter = _StubAdapter(_result(interpretation=_interpretation(po_reference=None)))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.advisory_consistent is True


# --- Finding 3: structural defects held before inference ---------------------------


def test_invoice_arithmetic_mismatch_held():
    inv = _invoice(
        inv_id="INV-2048", supplier="SUP-9001", po_ref="PO-77",
        lines=[_line(1, "1", "1250.00")], stated="9999.00",
    )
    po = _po(po_id="PO-77", supplier="SUP-9001", lines=[_line(1, "1", "1245.00")], stated="1245.00")
    case = _case(invoice=inv, po_lookup=POLookupFound(purchase_order=po))
    adapter = _StubAdapter(None)
    decision = route(case, evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


def test_duplicate_lines_held():
    inv = _invoice(
        inv_id="INV-2048", supplier="SUP-9001", po_ref="PO-77",
        lines=[_line(1, "1", "625.00"), _line(1, "1", "625.00")],
        stated="1250.00",
    )
    po = _po(po_id="PO-77", supplier="SUP-9001", lines=[_line(1, "1", "1245.00")], stated="1245.00")
    case = _case(invoice=inv, po_lookup=POLookupFound(purchase_order=po))
    adapter = _StubAdapter(None)
    decision = route(case, evidence_text="text", adapter=adapter)
    assert decision.route == Route.HELD
    assert adapter.calls == 0


# --- Finding 4: preflight rejection accounts zero model calls -----------------------


def test_preflight_oversized_input_zero_model_calls():
    adapter = _StubAdapter(_result(outcome=InferenceOutcome.INPUT_TOO_LARGE))
    decision = route(_discrepancy_case(), evidence_text="text", adapter=adapter)
    assert decision.route == Route.INFERENCE
    assert decision.advisory_consistent is False
    assert decision.evidence.model_calls == 0
    assert decision.evidence.total_tokens is None
    assert decision.evidence.elapsed_ms is None
