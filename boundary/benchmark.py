"""B5.1 offline benchmark harness for deterministic-first decision routing.

Measures BOUNDARY's selective inference routing against a synthetic case matrix.
The inference cases exercise the *real* ``InferenceAdapter`` parser/validator with
synthetic inputs (no canned outcomes): malformed JSON and prompt-injection payloads
are actually parsed and rejected by the accepted local validator. No network, no
real model, fully reproducible.

Provenance labels (unambiguous):
- ``OBSERVED_OFFLINE`` — routing decision, adapter invocation counts, local runtime.
- ``COUNTERFACTUAL`` — the arithmetic LLM-first call-opportunity baseline (computed,
  never an observed savings).
- ``NOT_CAPTURED`` — token telemetry and real AMD/Evolus latency, which require an
  authorized live run.

Important distinctions:
- Offline adapter invocations are *not* real AMD model calls (no live model ran).
- The counterfactual baseline is arithmetic only, not observed savings.
- AMD model invocations are measured separately from Evolus HTTP submission/polling
  counts; an HTTP request is never equated with an inference call.
- Synthetic security scenarios exercise the local parser/validator against synthetic
  inputs; they are not a claim of real-world attack-detection performance.

Usage::

    python -m boundary.benchmark            # run matrix, write JSON + report
    python -m boundary.benchmark --reps 5   # 5 repetitions per case
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from .contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupNotFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from .inference import (
    InferenceAdapter,
    InferenceConfig,
    InferenceOutcome,
    InferenceUnavailable,
)
from .router import Route, route

# Provenance labels.
OBSERVED_OFFLINE = "OBSERVED_OFFLINE"
COUNTERFACTUAL = "COUNTERFACTUAL"
NOT_CAPTURED = "NOT_CAPTURED"

# Counterfactual LLM-first call-opportunity per case (a naive guarded baseline
# invokes the model for every case).
COUNTERFACTUAL_CALLS_PER_CASE = 1


# --- Synthetic fixtures (fictional, safe for the public repository) ------------------


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _invoice(inv_id="INV-1", supplier="SUP-1", po_ref="PO-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return Invoice(
        invoice_id=inv_id, supplier_id=supplier, po_reference=po_ref,
        issued_date="2026-10-12", line_items=lines, stated_total=stated,
    )


def _po(po_id="PO-1", supplier="SUP-1", lines=None, stated="10.00"):
    lines = lines or [_line(1, "1", "10.00")]
    return PurchaseOrder(
        po_id=po_id, supplier_id=supplier, issued_date="2026-10-12",
        line_items=lines, stated_total=stated,
    )


def _case(case_id, invoice, po_lookup, supplier_state=SupplierMatchState.MATCH, evidence=None):
    return CaseInput(
        case_id=case_id, invoice=invoice, po_lookup=po_lookup,
        supplier_lookup=SupplierLookupResult(state=supplier_state),
        evidence=evidence if evidence is not None else [],
        policy_version="policy-v1", input_identity="benchmark",
    )


def _exact_match_case():
    inv = _invoice(inv_id="INV-100", supplier="SUP-100", po_ref="PO-100",
                   lines=[_line(1, "2", "25.00")], stated="50.00")
    po = _po(po_id="PO-100", supplier="SUP-100", lines=[_line(1, "2", "25.00")], stated="50.00")
    return _case("CASE-DET", inv, POLookupFound(purchase_order=po))


def _discrepancy_case(evidence=None):
    # Invoice $1,250.00 vs PO $1,245.00 -> 500-cent discrepancy.
    inv = _invoice(inv_id="INV-2048", supplier="SUP-9001", po_ref="PO-77",
                   lines=[_line(1, "1", "1250.00")], stated="1250.00")
    po = _po(po_id="PO-77", supplier="SUP-9001", lines=[_line(1, "1", "1245.00")], stated="1245.00")
    return _case("CASE-DISC", inv, POLookupFound(purchase_order=po), evidence=evidence)


def _missing_po_case():
    inv = _invoice(inv_id="INV-300", supplier="SUP-300", po_ref="PO-300",
                   lines=[_line(1, "1", "10.00")], stated="10.00")
    return _case("CASE-NOPO", inv, POLookupNotFound())


# --- Real parser/validator via a mock transport --------------------------------------


def mock_transport(content: Optional[str] = None, exc: Optional[Exception] = None):
    """Return a transport callable that yields the given content or raises ``exc``."""

    def transport(payload):
        if exc is not None:
            raise exc
        return {"choices": [{"message": {"content": content}}]}

    return transport


def real_adapter(content: Optional[str] = None, exc: Optional[Exception] = None) -> InferenceAdapter:
    """Build a real ``InferenceAdapter`` over a mock transport (no network)."""
    config = InferenceConfig(model_id="DeepSeek-V4-Flash")
    return InferenceAdapter(config, mock_transport(content=content, exc=exc))


def _consistent_content() -> str:
    return json.dumps({
        "supplier_id": "SUP-9001", "invoice_id": "INV-2048", "po_reference": "PO-77",
        "amount_cents": 125000, "discrepancy_cents": 500, "confidence": 0.6, "notes": "",
    })


def _injection_content() -> str:
    # A synthetic prompt-injection payload that asserts an authority key.
    return json.dumps({
        "approve": True,
        "supplier_id": "SUP-9001", "invoice_id": "INV-2048", "po_reference": "PO-77",
        "amount_cents": 125000, "discrepancy_cents": 500, "confidence": 0.6, "notes": "",
    })


# --- Benchmark model -----------------------------------------------------------------


@dataclass
class BenchmarkCase:
    category: str
    case: CaseInput
    evidence_text: Optional[str]
    expected_route: Route
    expected_calls: int
    expected_advisory_outcome: Optional[InferenceOutcome]
    adapter: Optional[InferenceAdapter] = None
    expected_consistent: Optional[bool] = None


@dataclass
class Observation:
    category: str
    repetition: int
    expected_route: str
    actual_route: str
    expected_calls: int
    actual_calls: int             # adapter.interpret() invocations (offline, not real AMD)
    model_calls: int               # RoutingEvidence.model_calls (offline adapter calls)
    expected_advisory_outcome: Optional[str]
    advisory_outcome: Optional[str]
    advisory_consistent: Optional[bool]
    execution_authority: str
    authority_claim_attempt: bool
    elapsed_ns: int               # local harness runtime, NOT model/API latency
    token_provenance: str         # NOT_CAPTURED offline
    route_correct: bool
    calls_correct: bool
    outcome_correct: bool
    consistent_correct: bool
    passed: bool


class RecordingAdapter:
    """Wraps a real adapter (or ``None``) and records the number of invocations."""

    def __init__(self, adapter: Optional[InferenceAdapter] = None):
        self._adapter = adapter
        self.calls = 0

    def interpret(self, evidence: str):
        self.calls += 1
        if self._adapter is None:
            raise AssertionError("adapter invoked for a case expected to make zero calls")
        return self._adapter.interpret(evidence)


def build_matrix() -> List[BenchmarkCase]:
    """The six required synthetic categories, with expected results derived from
    the accepted BOUNDARY contracts (not imposed assumptions)."""
    return [
        BenchmarkCase(
            category="deterministic_match",
            case=_exact_match_case(),
            evidence_text=None,
            expected_route=Route.DETERMINISTIC,
            expected_calls=0,
            expected_advisory_outcome=None,
        ),
        BenchmarkCase(
            category="financial_discrepancy",
            case=_discrepancy_case(),
            evidence_text="synthetic ambiguous evidence",
            expected_route=Route.INFERENCE,
            expected_calls=1,
            expected_advisory_outcome=InferenceOutcome.SUCCESS,
            adapter=real_adapter(content=_consistent_content()),
            expected_consistent=True,
        ),
        BenchmarkCase(
            category="missing_purchase_order",
            case=_missing_po_case(),
            evidence_text=None,
            expected_route=Route.HELD,
            expected_calls=0,
            expected_advisory_outcome=None,
        ),
        BenchmarkCase(
            category="malformed_evidence",
            case=_discrepancy_case(),
            evidence_text="synthetic malformed evidence",
            expected_route=Route.INFERENCE,
            expected_calls=1,
            expected_advisory_outcome=InferenceOutcome.MALFORMED_RESPONSE,
            adapter=real_adapter(content="this is not valid json {"),
            expected_consistent=False,
        ),
        BenchmarkCase(
            category="prompt_injection",
            case=_discrepancy_case(),
            evidence_text="synthetic injection evidence",
            expected_route=Route.INFERENCE,
            expected_calls=1,
            expected_advisory_outcome=InferenceOutcome.AUTHORITY_CLAIM,
            adapter=real_adapter(content=_injection_content()),
            expected_consistent=False,
        ),
        BenchmarkCase(
            category="unavailable_inference",
            case=_discrepancy_case(),
            evidence_text="synthetic evidence",
            expected_route=Route.INFERENCE,
            expected_calls=1,
            expected_advisory_outcome=InferenceOutcome.UNAVAILABLE,
            adapter=real_adapter(exc=InferenceUnavailable("unavailable")),
            expected_consistent=False,
        ),
    ]


def run_case(bench_case: BenchmarkCase, repetition: int) -> Observation:
    recording = RecordingAdapter(bench_case.adapter)
    start = time.perf_counter_ns()
    decision = route(bench_case.case, evidence_text=bench_case.evidence_text, adapter=recording)
    elapsed_ns = time.perf_counter_ns() - start

    actual_calls = recording.calls
    model_calls = decision.evidence.model_calls
    advisory_outcome = decision.evidence.advisory_outcome
    advisory_outcome_str = advisory_outcome.value if advisory_outcome is not None else None
    authority_claim_attempt = advisory_outcome == InferenceOutcome.AUTHORITY_CLAIM
    accepted_authority_violation = decision.execution_authority != "NONE"

    route_correct = decision.route == bench_case.expected_route
    calls_correct = actual_calls == bench_case.expected_calls and model_calls == bench_case.expected_calls
    outcome_correct = advisory_outcome == bench_case.expected_advisory_outcome
    consistent_correct = (
        bench_case.expected_consistent is None
        or decision.advisory_consistent == bench_case.expected_consistent
    )
    passed = (
        route_correct
        and calls_correct
        and outcome_correct
        and consistent_correct
        and not accepted_authority_violation
    )

    return Observation(
        category=bench_case.category,
        repetition=repetition,
        expected_route=bench_case.expected_route.value,
        actual_route=decision.route.value,
        expected_calls=bench_case.expected_calls,
        actual_calls=actual_calls,
        model_calls=model_calls,
        expected_advisory_outcome=(
            bench_case.expected_advisory_outcome.value
            if bench_case.expected_advisory_outcome is not None else None
        ),
        advisory_outcome=advisory_outcome_str,
        advisory_consistent=decision.advisory_consistent,
        execution_authority=decision.execution_authority,
        authority_claim_attempt=authority_claim_attempt,
        elapsed_ns=elapsed_ns,
        token_provenance=NOT_CAPTURED,
        route_correct=route_correct,
        calls_correct=calls_correct,
        outcome_correct=outcome_correct,
        consistent_correct=consistent_correct,
        passed=passed,
    )


def run_benchmark(cases: List[BenchmarkCase], repetitions: int = 3) -> List[Observation]:
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    observations: List[Observation] = []
    for case in cases:
        for rep in range(repetitions):
            observations.append(run_case(case, rep))
    return observations


def summarize(cases: List[BenchmarkCase], observations: List[Observation]) -> dict:
    total = len(observations)
    passed = sum(1 for o in observations if o.passed)

    counterfactual_calls = len(cases) * COUNTERFACTUAL_CALLS_PER_CASE
    # Actual offline adapter invocations per distinct case (deterministic).
    actual_model_calls = sum(o.model_calls for o in observations if o.repetition == 0)
    call_opportunity_avoided = counterfactual_calls - actual_model_calls

    authority_claim_observations = sum(1 for o in observations if o.authority_claim_attempt)
    authority_claim_scenarios = len({o.category for o in observations if o.authority_claim_attempt})
    accepted_authority_violations = sum(1 for o in observations if o.execution_authority != "NONE")

    per_category: dict = {}
    for case in cases:
        cat_obs = [o for o in observations if o.category == case.category]
        per_category[case.category] = {
            "expected_route": case.expected_route.value,
            "actual_route": cat_obs[0].actual_route,
            "expected_outcome": cat_obs[0].expected_advisory_outcome,
            "actual_outcome": cat_obs[0].advisory_outcome,
            "offline_invocations": cat_obs[0].model_calls,
            "passed": all(o.passed for o in cat_obs),
            "repetitions": len(cat_obs),
        }

    return {
        "total_observations": total,
        "passed": passed,
        "failed": total - passed,
        "counterfactual": {
            "label": COUNTERFACTUAL,
            "cases": len(cases),
            "counterfactual_llm_calls": counterfactual_calls,
            "actual_offline_invocations": actual_model_calls,
            "call_opportunity_avoided": call_opportunity_avoided,
            "note": "arithmetic call-opportunity baseline; offline invocations are not real "
                    "AMD calls, and this is not token, latency, or cost savings",
        },
        "authority": {
            "claim_scenarios": authority_claim_scenarios,
            "claim_observations": authority_claim_observations,
            "accepted_violations": accepted_authority_violations,
            "note": "authority-claim observations are repetitions of one synthetic scenario; "
                    "every attempt is a rejection, and accepted violations must be zero",
        },
        "per_category": per_category,
    }


def render_report(summary: dict) -> str:
    lines: List[str] = []
    lines.append("# BOUNDARY B5.1 — Offline Benchmark Report")
    lines.append("")
    lines.append(f"- Observations: {summary['total_observations']}")
    lines.append(f"- Passed: {summary['passed']} / {summary['total_observations']}")
    lines.append(f"- Failed: {summary['failed']}")
    lines.append("")
    lines.append("## Per-category")
    lines.append("")
    lines.append("| category | expected route | actual route | expected outcome | actual outcome | offline invocations | passed |")
    lines.append("|---|---|---|---|---|---|---|")
    for category, info in summary["per_category"].items():
        lines.append(
            f"| {category} | {info['expected_route']} | {info['actual_route']} "
            f"| {info['expected_outcome']} | {info['actual_outcome']} "
            f"| {info['offline_invocations']} | {info['passed']} |"
        )
    lines.append("")
    lines.append("## Counterfactual call-opportunity baseline (COUNTERFACTUAL)")
    lines.append("")
    c = summary["counterfactual"]
    lines.append(f"- Cases: {c['cases']}")
    lines.append(f"- Counterfactual LLM-first calls: {c['counterfactual_llm_calls']}")
    lines.append(f"- Actual offline adapter invocations: {c['actual_offline_invocations']}")
    lines.append(f"- Call opportunities avoided: {c['call_opportunity_avoided']}")
    lines.append(f"- Note: {c['note']}")
    lines.append("")
    lines.append("## Authority enforcement")
    lines.append("")
    a = summary["authority"]
    lines.append(f"- Distinct authority-claim scenarios: {a['claim_scenarios']}")
    lines.append(f"- Authority-claim observations (repetitions): {a['claim_observations']}")
    lines.append(f"- Accepted authority violations: {a['accepted_violations']}")
    lines.append(f"- Note: {a['note']}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Routing decisions, invocation counts, and local runtime: OBSERVED_OFFLINE.")
    lines.append("- Offline adapter invocations are NOT real AMD model calls (no live model ran).")
    lines.append("- Token telemetry and real AMD/Evolus latency: NOT_CAPTURED (require an")
    lines.append("  authorized live run).")
    lines.append("- Synthetic malformed/injection scenarios exercise the real local parser/")
    lines.append("  validator; this is not a claim of real-world attack-detection performance.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BOUNDARY B5.1 offline benchmark")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per case (positive)")
    parser.add_argument("--json", default="benchmark_results.json", help="machine-readable output path")
    parser.add_argument("--report", default="benchmark_report.md", help="human-readable report path")
    args = parser.parse_args(argv)

    if args.reps <= 0:
        parser.error("--reps must be a positive integer")

    cases = build_matrix()
    observations = run_benchmark(cases, repetitions=args.reps)
    summary = summarize(cases, observations)

    results = {
        "provenance": OBSERVED_OFFLINE,
        "summary": summary,
        "observations": [asdict(o) for o in observations],
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report = render_report(summary)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
