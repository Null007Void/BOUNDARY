"""B5.3 benchmark evidence reconciliation and reporting.

Reconciles the three B5.2 live-measurement JSON records (financial discrepancy,
exact match, missing purchase order) into a judge-readable report and a
machine-readable validated summary.

Each record is validated for scenario identity, route, provenance, request count,
authority state, measurement result, and telemetry consistency before it is
accepted. Records that are missing, duplicated, contradictory, malformed, or that
claim success beyond the evidence are rejected.

This tool is offline: it reads only explicit file paths and performs no network I/O.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Provenance labels (mirror the accepted benchmark_live module).
OBSERVED_LIVE = "OBSERVED_LIVE"
OBSERVED_OFFLINE = "OBSERVED_OFFLINE"
LIVE_ZERO_REQUEST = "LIVE_ZERO_REQUEST"

SCENARIOS = ("financial_discrepancy", "exact_match", "missing_po")

# Expected contract per scenario, derived from the accepted B5.2 semantics.
# provenance is the single REQUIRED label: any other value (including offline or
# unconfirmed) is rejected.
EXPECTED = {
    "financial_discrepancy": {
        "route": "INFERENCE",
        "provenance": OBSERVED_LIVE,
        "outbound_requests": 1,
        "confirmed_response": True,
        "model_calls": 1,
        "requires_tokens": True,
    },
    "exact_match": {
        "route": "DETERMINISTIC",
        "provenance": LIVE_ZERO_REQUEST,
        "outbound_requests": 0,
        "confirmed_response": False,
        "model_calls": 0,
        "requires_tokens": False,
    },
    "missing_po": {
        "route": "HELD",
        "provenance": LIVE_ZERO_REQUEST,
        "outbound_requests": 0,
        "confirmed_response": False,
        "model_calls": 0,
        "requires_tokens": False,
    },
}


@dataclass
class ValidatedRecord:
    scenario: str
    valid: bool
    issues: List[str]
    record: dict


@dataclass
class ReconcileResult:
    records: List[ValidatedRecord]
    cross_record_issues: List[str]
    all_valid: bool


def _is_nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_true(value) -> bool:
    return value is True


def _is_false(value) -> bool:
    return value is False


def validate_record(record) -> ValidatedRecord:
    if not isinstance(record, dict):
        return ValidatedRecord(
            scenario="<not-an-object>", valid=False,
            issues=["record is not a JSON object"], record={},
        )

    scenario = record.get("scenario")
    if not isinstance(scenario, str) or scenario not in EXPECTED:
        return ValidatedRecord(
            scenario=repr(scenario), valid=False,
            issues=[f"unknown or missing scenario: {scenario!r}"], record=record,
        )

    exp = EXPECTED[scenario]
    issues: List[str] = []

    # Route and internal routing-consistency fields.
    route = record.get("route")
    if route != exp["route"]:
        issues.append(f"route {route!r} != expected {exp['route']!r}")
    expected_route = record.get("expected_route")
    if expected_route != exp["route"]:
        issues.append(f"expected_route {expected_route!r} != expected {exp['route']!r}")
    if not _is_true(record.get("route_correct")):
        issues.append(f"route_correct is not True: {record.get('route_correct')!r}")

    # Provenance and transport/dry-run consistency.
    provenance = record.get("provenance")
    if provenance != exp["provenance"]:
        issues.append(
            f"provenance {provenance!r} != expected {exp['provenance']!r} "
            "(offline/mock/unconfirmed is not a genuine live measurement)"
        )
    transport_kind = record.get("transport_kind")
    if transport_kind != "live":
        issues.append(f"transport_kind {transport_kind!r} != 'live' (provenance is live)")
    if not _is_false(record.get("dry_run")):
        issues.append(f"dry_run is not False: {record.get('dry_run')!r}")

    # Authority.
    if record.get("execution_authority") != "NONE":
        issues.append(f"execution_authority {record.get('execution_authority')!r} != 'NONE'")
    if not _is_true(record.get("authority_preserved")):
        issues.append(f"authority_preserved is not True: {record.get('authority_preserved')!r}")

    # Measurement result and internal consistency.
    if not _is_true(record.get("harness_safe")):
        issues.append(f"harness_safe is not True: {record.get('harness_safe')!r}")
    if not _is_true(record.get("measurement_success")):
        issues.append(f"measurement_success is not True: {record.get('measurement_success')!r}")
    if not _is_true(record.get("outbound_correct")):
        issues.append(f"outbound_correct is not True: {record.get('outbound_correct')!r}")

    # Request/count fields: correct integer types (reject booleans).
    ob = record.get("outbound_requests")
    if not _is_nonnegative_int(ob) or ob != exp["outbound_requests"]:
        issues.append(f"outbound_requests {ob!r} != expected {exp['outbound_requests']}")
    mc = record.get("model_calls")
    if not _is_nonnegative_int(mc) or mc != exp["model_calls"]:
        issues.append(f"model_calls {mc!r} != expected {exp['model_calls']}")

    # confirmed_response: correct boolean type.
    cr = record.get("confirmed_response")
    cr_ok = _is_true(cr) if exp["confirmed_response"] is True else _is_false(cr)
    if not cr_ok:
        issues.append(
            f"confirmed_response {cr!r} != expected {exp['confirmed_response']}"
        )

    # harness_elapsed_ns: non-negative integer.
    hen = record.get("harness_elapsed_ns")
    if not _is_nonnegative_int(hen):
        issues.append(f"harness_elapsed_ns is not a non-negative integer: {hen!r}")

    if exp["requires_tokens"]:
        # Confirmed live inference: model identity, consistent token telemetry, latency.
        model_id = record.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            issues.append(f"model_id missing or empty: {model_id!r}")

        prompt = record.get("prompt_tokens")
        completion = record.get("completion_tokens")
        total = record.get("total_tokens")
        for name, v in (("prompt_tokens", prompt), ("completion_tokens", completion),
                        ("total_tokens", total)):
            if not _is_nonnegative_int(v):
                issues.append(f"{name} is not a non-negative integer: {v!r}")
        if (_is_nonnegative_int(prompt) and _is_nonnegative_int(completion)
                and _is_nonnegative_int(total) and total != prompt + completion):
            issues.append(f"total_tokens {total} != prompt {prompt} + completion {completion}")

        elapsed = record.get("elapsed_ms")
        if not _is_nonnegative_int(elapsed):
            issues.append(f"elapsed_ms is not a non-negative integer: {elapsed!r}")

        if record.get("advisory_outcome") != "SUCCESS":
            issues.append(f"advisory_outcome {record.get('advisory_outcome')!r} != 'SUCCESS'")
        if not _is_true(record.get("advisory_consistent")):
            issues.append(f"advisory_consistent is not True: {record.get('advisory_consistent')!r}")
    else:
        # Zero-request scenario: no tokens, no adapter latency, no advisory outcome or
        # consistency claim. A configured model_id may legitimately be present and is
        # not rejected.
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_ms"):
            v = record.get(name)
            if v is not None:
                issues.append(f"{name} should be absent for a zero-request scenario, got {v!r}")
        if record.get("advisory_outcome") is not None:
            issues.append(
                f"advisory_outcome should be absent for a zero-request scenario, "
                f"got {record.get('advisory_outcome')!r}"
            )
        if record.get("advisory_consistent") is not None:
            issues.append(
                f"advisory_consistent should be absent for a zero-request scenario, "
                f"got {record.get('advisory_consistent')!r}"
            )

    return ValidatedRecord(scenario=scenario, valid=(len(issues) == 0),
                           issues=issues, record=record)


def reconcile(records: List[dict]) -> ReconcileResult:
    validated = [validate_record(r) for r in records]
    cross: List[str] = []

    counts = Counter(vr.scenario for vr in validated if vr.scenario in SCENARIOS)
    for s in SCENARIOS:
        c = counts.get(s, 0)
        if c == 0:
            cross.append(f"missing scenario: {s}")
        elif c > 1:
            cross.append(f"duplicate scenario {s}: {c} records")

    all_valid = all(vr.valid for vr in validated) and not cross
    return ReconcileResult(records=validated, cross_record_issues=cross, all_valid=all_valid)


def _by_scenario(result: ReconcileResult):
    return {vr.scenario: vr for vr in result.records if vr.scenario in SCENARIOS}


def build_summary(result: ReconcileResult) -> dict:
    scenarios = {}
    for s in SCENARIOS:
        vr = _by_scenario(result).get(s)
        if vr is None:
            scenarios[s] = {"valid": False, "issues": ["missing"]}
            continue
        r = vr.record
        scenarios[s] = {
            "valid": vr.valid,
            "issues": vr.issues,
            "route": r.get("route"),
            "provenance": r.get("provenance"),
            "outbound_requests": r.get("outbound_requests"),
            "model_calls": r.get("model_calls"),
            "execution_authority": r.get("execution_authority"),
            "model_id": r.get("model_id"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "total_tokens": r.get("total_tokens"),
            "elapsed_ms": r.get("elapsed_ms"),
            "harness_elapsed_ns": r.get("harness_elapsed_ns"),
        }

    summary = {
        "validated": result.all_valid,
        "scenarios": scenarios,
        "cross_record_issues": result.cross_record_issues,
        "offline_benchmark": {
            "routing_observations_provenance": "OBSERVED_OFFLINE",
            "llm_first_comparator_provenance": "COUNTERFACTUAL",
            "call_opportunities_avoided": None,
            "note": (
                "The offline six-category routing observations are OBSERVED_OFFLINE "
                "(actual offline measurements). Only the hypothetical LLM-first "
                "call-opportunity comparator is COUNTERFACTUAL: it posits one model call "
                "per scenario that deterministic/held routing avoided. This is arithmetic, "
                "NOT measured token, latency, or cost savings."
            ),
        },
        "attestation_note": (
            "Structural reconciliation validates record shape and internal consistency; "
            "it does not independently authenticate the AMD server or cryptographically "
            "attest to the origin of a JSON record."
        ),
        "limitations": [
            "Sample size: three scenarios, one observation each; no repeated trials, "
            "no statistical significance, no averages.",
            "Synthetic fixtures only.",
            "Model output variability is not characterized.",
            "No measured financial savings (counterfactual call avoidance only).",
            "No comprehensive security certification.",
        ],
    }

    if result.all_valid:
        fin = _by_scenario(result)["financial_discrepancy"].record
        # Measured AMD consumption (one confirmed request).
        summary["measured"] = {
            "amd_requests": fin.get("outbound_requests"),
            "prompt_tokens": fin.get("prompt_tokens"),
            "completion_tokens": fin.get("completion_tokens"),
            "total_tokens": fin.get("total_tokens"),
            "elapsed_ms": fin.get("elapsed_ms"),
            "model_id": fin.get("model_id"),
        }
        # Counterfactual call avoidance: 3 scenarios, 1 actual model call.
        summary["offline_benchmark"]["call_opportunities_avoided"] = len(SCENARIOS) - 1

    return summary


def render_report(result: ReconcileResult, summary: dict) -> str:
    lines: List[str] = []
    lines.append("# BOUNDARY B5.3 — Validated Benchmark Evidence Report")
    lines.append("")

    by = _by_scenario(result)

    lines.append("## B5.2 Live Measurements (three scenarios, one observation each)")
    lines.append("")
    for s in SCENARIOS:
        vr = by.get(s)
        if vr is None:
            lines.append(f"- {s}: MISSING.")
            continue
        r = vr.record
        status = "VALID" if vr.valid else "INVALID: " + "; ".join(vr.issues)
        lines.append(f"- {s}: {r.get('route')} | {r.get('provenance')} | "
                     f"{r.get('outbound_requests')} request(s) | authority {r.get('execution_authority')} "
                     f"| {status}")
        if s == "financial_discrepancy" and vr.valid:
            lines.append(f"    measured: {r.get('prompt_tokens')} prompt + "
                         f"{r.get('completion_tokens')} completion = {r.get('total_tokens')} tokens, "
                         f"{r.get('elapsed_ms')} ms adapter latency, model {r.get('model_id')}.")
        elif vr.valid:
            lines.append(f"    local runtime: {r.get('harness_elapsed_ns')} ns (harness wall-clock).")
    lines.append("")

    lines.append("## B5.1 Offline Benchmark (OBSERVED_OFFLINE routing; COUNTERFACTUAL comparator)")
    lines.append("")
    lines.append(summary["offline_benchmark"]["note"])
    lines.append("")
    lines.append("The COUNTERFACTUAL comparator is NOT measured AMD token consumption and is not")
    lines.append("comparable to it; the two are reported separately above and below.")
    lines.append("")

    lines.append("## Latency note")
    lines.append("")
    lines.append("Zero-request local runtimes (exact_match and missing_po) are local harness")
    lines.append("wall-clock timings measured in nanoseconds. They are NOT comparable to the")
    lines.append("real adapter latency of the inference scenario, which is measured in")
    lines.append("milliseconds against the AMD endpoint.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for lim in summary["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## Attestation scope")
    lines.append("")
    lines.append(summary["attestation_note"])
    lines.append("")

    lines.append("## Validation")
    lines.append("")
    lines.append(f"- Records accepted: {len([vr for vr in result.records if vr.valid])} / {len(result.records)}")
    if result.cross_record_issues:
        lines.append("- Cross-record issues: " + "; ".join(result.cross_record_issues))
    else:
        lines.append("- Cross-record issues: none (all three scenarios present exactly once).")
    lines.append(f"- Overall: {'VALID' if result.all_valid else 'INVALID'}")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BOUNDARY B5.3 evidence reconciliation")
    parser.add_argument("--records", nargs="+", required=True,
                        help="paths to the live-measurement JSON files")
    parser.add_argument("--report", default=None, help="write the report to a file")
    parser.add_argument("--summary", default=None, help="write the validated summary to a JSON file")
    args = parser.parse_args(argv)

    records: List[dict] = []
    read_errors: List[str] = []
    for path in args.records:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except FileNotFoundError:
            read_errors.append(f"not found: {path}")
        except json.JSONDecodeError as e:
            read_errors.append(f"malformed JSON in {path}: {e}")
        except OSError as e:
            read_errors.append(f"cannot read {path}: {e}")

    if read_errors:
        for msg in read_errors:
            print(msg)
        return 1

    result = reconcile(records)
    summary = build_summary(result)
    report = render_report(result, summary)

    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0 if result.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
