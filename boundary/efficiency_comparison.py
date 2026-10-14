"""S4E.2 — Controlled DeepSeek efficiency comparison (offline).

Compares two execution strategies over the *same* synthetic case matrix and the *same*
designated model (``DeepSeek-V4-Flash``), isolating the benefit of BOUNDARY's
deterministic-first routing from any difference between language models.

Offline comparison (this task)
------------------------------

- **Strategy A — LLM-first baseline.** A conventional architecture invokes the model for
  every case before any deterministic resolution. Recorded as a ``COUNTERFACTUAL`` call
  opportunity and **never executed**.
- **Strategy B — BOUNDARY deterministic-first.** The accepted ``router.route`` resolves
  exact matches deterministically, holds unresolved/structurally-defective evidence, and
  invokes inference only for eligible exceptions. Recorded as ``OBSERVED_OFFLINE`` via an
  injected mock adapter.

The offline comparison is a *modeled* call-opportunity comparison. It computes **no**
token, latency, or cost savings.

Future live experiment (specification only)
-------------------------------------------

A genuine same-model comparison measuring **both** strategies on the three safe
synthetic scenarios (exact match, financial discrepancy, missing purchase order) via
DeepSeek-V4-Flash on AMD Token Factory. See ``LIVE_EXPERIMENT_SPEC``.

Usage::

    python -m boundary.efficiency_comparison
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from .benchmark import (
    BenchmarkCase,
    COUNTERFACTUAL,
    OBSERVED_OFFLINE,
    RecordingAdapter,
    build_matrix,
)
from .router import Route, route

DESIGNATED_MODEL = "DeepSeek-V4-Flash"
# A naive LLM-first architecture invokes the model for every case.
LLM_FIRST_CALLS_PER_CASE = 1


def _disposition(r: Route) -> str:
    if r == Route.DETERMINISTIC:
        return "avoided"  # exact match resolved with zero inference
    if r == Route.HELD:
        return "held"  # model cannot resolve; inference not selected
    return "selected"  # bounded inference invoked


@dataclass
class ComparisonRecord:
    category: str
    case_id: str
    designated_model: str
    # Strategy A — LLM-first baseline (hypothetical).
    llm_first_call_opportunities: int
    llm_first_provenance: str  # COUNTERFACTUAL
    # Strategy B — BOUNDARY deterministic-first (observed offline).
    boundary_route: str
    boundary_adapter_invocations: int
    boundary_provenance: str  # OBSERVED_OFFLINE
    disposition: str  # "selected" / "avoided" / "held"
    reason_codes: List[str]
    advisory_outcome: Optional[str]
    advisory_consistent: Optional[bool]
    execution_authority: str
    calls_avoided: int


def run_comparison(cases: List[BenchmarkCase]) -> List[ComparisonRecord]:
    """Run Strategy B (real routing) against each case and record the Strategy A
    (LLM-first) hypothetical call opportunity for the same case facts."""
    records: List[ComparisonRecord] = []
    for bc in cases:
        recording = RecordingAdapter(bc.adapter)
        decision = route(bc.case, evidence_text=bc.evidence_text, adapter=recording)
        boundary_calls = decision.evidence.model_calls
        outcome = decision.evidence.advisory_outcome
        records.append(
            ComparisonRecord(
                category=bc.category,
                case_id=bc.case.case_id,
                designated_model=DESIGNATED_MODEL,
                llm_first_call_opportunities=LLM_FIRST_CALLS_PER_CASE,
                llm_first_provenance=COUNTERFACTUAL,
                boundary_route=decision.route.value,
                boundary_adapter_invocations=boundary_calls,
                boundary_provenance=OBSERVED_OFFLINE,
                disposition=_disposition(decision.route),
                reason_codes=[rc.value for rc in decision.proposal.reason_codes],
                advisory_outcome=(outcome.value if outcome is not None else None),
                advisory_consistent=decision.advisory_consistent,
                execution_authority=decision.execution_authority,
                calls_avoided=LLM_FIRST_CALLS_PER_CASE - boundary_calls,
            )
        )
    return records


def summarize(records: List[ComparisonRecord]) -> dict:
    total = len(records)
    llm_first_total = sum(r.llm_first_call_opportunities for r in records)
    boundary_total = sum(r.boundary_adapter_invocations for r in records)
    avoided = llm_first_total - boundary_total
    authority_violations = sum(1 for r in records if r.execution_authority != "NONE")

    per_category = {
        r.category: {
            "disposition": r.disposition,
            "llm_first_calls": r.llm_first_call_opportunities,
            "boundary_adapter_invocations": r.boundary_adapter_invocations,
            "calls_avoided": r.calls_avoided,
        }
        for r in records
    }

    return {
        "designated_model": DESIGNATED_MODEL,
        "total_cases": total,
        "strategy_a": {
            "label": "LLM-first baseline",
            "provenance": COUNTERFACTUAL,
            "total_call_opportunities": llm_first_total,
            "note": "hypothetical one-model-call-per-case; never executed",
        },
        "strategy_b": {
            "label": "BOUNDARY deterministic-first",
            "provenance": OBSERVED_OFFLINE,
            "total_adapter_invocations": boundary_total,
            "note": "offline adapter invocations, not genuine model requests",
        },
        "calls_avoided": avoided,
        "calls_avoided_note": (
            "modeled comparison of call opportunities — NOT measured token, latency, "
            "or cost savings"
        ),
        "authority_violations_accepted": authority_violations,
        "per_category": per_category,
    }


# --- Reusable live-experiment specification (future bounded AMD run) -----------------

LIVE_EXPERIMENT_SPEC = {
    "experiment_id": "deepseek-efficiency-live-v1",
    "purpose": (
        "genuine same-model comparison: a measured LLM-first baseline vs a measured "
        "BOUNDARY deterministic-first strategy, both on DeepSeek-V4-Flash via AMD "
        "Token Factory"
    ),
    "designated_model": DESIGNATED_MODEL,
    "provider": "AMD Token Factory",
    "strategies": {
        "A": {
            "name": "Measured LLM-first baseline",
            "behavior": (
                "one bounded advisory-only model request per scenario, every repetition"
            ),
            "requests_per_repetition": 3,
            "authority": (
                "advisory-only; execution_authority=NONE; baseline responses never "
                "authorize journal entries or financial actions"
            ),
        },
        "B": {
            "name": "Measured BOUNDARY deterministic-first",
            "behavior": (
                "route per the accepted contract; invoke the model only for the "
                "eligible financial discrepancy"
            ),
            "requests_per_repetition": 1,
            "authority": "advisory-only; execution_authority=NONE",
        },
    },
    "scenarios": ["exact_match", "financial_discrepancy", "missing_purchase_order"],
    "offline_only_categories": [
        "malformed_evidence",
        "prompt_injection",
        "unavailable_inference",
    ],
    "repetitions": 2,
    "request_budget": {
        "strategy_a": 6,  # 3 scenarios x 2 repetitions
        "strategy_b": 2,  # 1 eligible scenario x 2 repetitions
        "maximum_total": 8,
        "note": "PROPOSED budget, not authorization to execute",
    },
    "derived_request_contract": {
        "source": "B3.2 InferenceAdapter + B5.2 build_live_adapter/SingleRequestGuard",
        "endpoint": "AMD /chat/completions (via build_amd_transport)",
        "config": (
            "InferenceConfig.from_env() — model_id, api_key (BOUNDARY_AMD_API_KEY), "
            "base_url, timeout_seconds"
        ),
        "prompt_generation": (
            "fixed advisory-only system prompt + user evidence text "
            "(InferenceAdapter._build_payload)"
        ),
        "evidence_inputs": (
            "reuse B5.2 SCENARIOS (exact_match/financial_discrepancy/missing_po) and "
            "SYNTHETIC_EVIDENCE (boundary.smoke_test)"
        ),
        "generation_parameters": (
            "model_id, max_input_chars, max_output_chars; any additional generation "
            "config and sampling settings to be pinned before execution"
        ),
        "model_identity_verification": (
            "verify the endpoint-reported model identity matches DeepSeek-V4-Flash"
        ),
        "request_limit": (
            "at most one outbound request per scenario (SingleRequestGuard); no "
            "automatic retries"
        ),
        "timeout": "config.timeout_seconds (default 30.0)",
        "failure_handling": (
            "fail closed to documented outcomes (TIMEOUT, AUTH_FAILURE, UNAVAILABLE, "
            "MALFORMED_RESPONSE, etc.); no silent retry"
        ),
        "repetition_order": "fixed scenario order per repetition; call order recorded",
    },
    "tracked_metrics": [
        "actual vs hypothetical model requests (per strategy)",
        "prompt/completion/total tokens (only when genuinely measured)",
        "adapter latency and measurement scope",
        "successful responses vs failures",
        "advisory validity and source consistency",
        "authority violations and rejection outcomes",
        "case-level route and final governed disposition",
    ],
    "provenance": {
        "OBSERVED_LIVE": "confirmed HTTP response over a real transport (success or failure)",
        "LIVE_UNCONFIRMED": "real transport attempted but no confirmed response",
        "LIVE_ZERO_REQUEST": "live transport configured but zero requests (deterministic/held)",
        "OBSERVED_OFFLINE": "mock/WSGI/custom transport or dry-run",
    },
    "reporting_scope": (
        "a small controlled experiment, not universal savings or statistically "
        "established performance"
    ),
    "do_not": [
        "invent missing telemetry",
        "attribute baseline measurements to BOUNDARY",
        "compute percentage token or cost savings without actual comparable measurements",
        "allow baseline model responses to authorize journal entries or financial actions",
    ],
}


def render_report(summary: dict, records: List[ComparisonRecord]) -> str:
    lines: List[str] = []
    lines.append("# BOUNDARY — Controlled DeepSeek Efficiency Comparison (offline)")
    lines.append("")
    lines.append(f"- Designated model (both strategies): **{summary['designated_model']}**")
    lines.append(f"- Cases: {summary['total_cases']}")
    lines.append("")
    lines.append("## Strategies")
    lines.append("")
    a = summary["strategy_a"]
    b = summary["strategy_b"]
    lines.append(f"- **Strategy A — {a['label']}** ({a['provenance']}):")
    lines.append(f"  {a['total_call_opportunities']} hypothetical model-call opportunities "
                 f"({a['note']}).")
    lines.append(f"- **Strategy B — {b['label']}** ({b['provenance']}):")
    lines.append(f"  {b['total_adapter_invocations']} offline adapter invocations "
                 f"({b['note']}).")
    lines.append("")
    lines.append(f"**Call opportunities avoided: {summary['calls_avoided']}** "
                 f"(modeled comparison — not measured savings).")
    lines.append("")
    lines.append("## Per-category")
    lines.append("")
    lines.append("| category | disposition | LLM-first calls | BOUNDARY invocations | avoided |")
    lines.append("|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r.category} | {r.disposition} | {r.llm_first_call_opportunities} "
            f"| {r.boundary_adapter_invocations} | {r.calls_avoided} |"
        )
    lines.append("")
    lines.append("## Authority")
    lines.append("")
    lines.append(f"- Accepted authority violations: {summary['authority_violations_accepted']}")
    lines.append("- Every model output carries `execution_authority = NONE`; a simulated "
                 "authority claim is rejected by the accepted validator.")
    lines.append("")
    lines.append("## What is observed vs. modeled")
    lines.append("")
    lines.append("- **OBSERVED_OFFLINE** — Strategy B routing decisions and offline adapter "
                 "invocations.")
    lines.append("- **COUNTERFACTUAL** — Strategy A call opportunities (arithmetic, hypothetical).")
    lines.append("- **RECORDED_LIVE** — none in this task; deferred to a separately authorized "
                 "live run.")
    lines.append("")
    lines.append(summary["calls_avoided_note"])
    return "\n".join(lines)


def render_live_spec(spec: dict) -> str:
    lines: List[str] = []
    lines.append("# Future Live AMD Comparison — Experiment Specification")
    lines.append("")
    lines.append(f"- Experiment: {spec['experiment_id']}")
    lines.append(f"- Model: {spec['designated_model']} via {spec['provider']}")
    lines.append(f"- {spec['purpose']}")
    lines.append("")
    lines.append("## Strategies (both measured)")
    for key in ("A", "B"):
        s = spec["strategies"][key]
        lines.append(f"- **{key} — {s['name']}**: {s['behavior']} "
                     f"({s['requests_per_repetition']} request(s)/repetition).")
        lines.append(f"  Authority: {s['authority']}.")
    lines.append("")
    lines.append("## Scenarios")
    lines.append("- Live: " + ", ".join(spec["scenarios"]))
    lines.append("- Offline-only: " + ", ".join(spec["offline_only_categories"]))
    lines.append("")
    lines.append("## Request budget (PROPOSED, not authorized)")
    b = spec["request_budget"]
    lines.append(f"- Strategy A: {b['strategy_a']} genuine requests")
    lines.append(f"- Strategy B: {b['strategy_b']} genuine requests")
    lines.append(f"- Maximum total: **{b['maximum_total']}**")
    lines.append(f"- Repetitions: {spec['repetitions']}")
    lines.append("")
    lines.append("## Derived request contract")
    for k, v in spec["derived_request_contract"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Tracked metrics")
    for m in spec["tracked_metrics"]:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## Provenance")
    for k, v in spec["provenance"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Constraints")
    lines.append(f"- Reporting scope: {spec['reporting_scope']}.")
    for d in spec["do_not"]:
        lines.append(f"- Do not {d}.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BOUNDARY S4E.2 efficiency comparison")
    parser.add_argument("--json", default="efficiency_comparison_results.json",
                        help="machine-readable output path")
    parser.add_argument("--report", default="efficiency_comparison_report.md",
                        help="human-readable report path")
    args = parser.parse_args(argv)

    cases = build_matrix()
    records = run_comparison(cases)
    summary = summarize(records)

    results = {
        "provenance_note": (
            "mixed provenance — Strategy A opportunities are COUNTERFACTUAL "
            "(hypothetical); Strategy B routing is OBSERVED_OFFLINE; no RECORDED_LIVE "
            "measurements exist"
        ),
        "summary": summary,
        "records": [asdict(r) for r in records],
        "live_experiment_spec": LIVE_EXPERIMENT_SPEC,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report = render_report(summary, records) + "\n\n" + render_live_spec(LIVE_EXPERIMENT_SPEC)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
