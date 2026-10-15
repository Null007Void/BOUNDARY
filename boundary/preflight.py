"""S4E.14 — controlled two-model comparative preflight (offline-only).

Prepares a bounded, live-ready comparison of DeepSeek and Gemma under LLM-first and BOUNDARY
strategies, spanning the three live-safe invoice-routing families, over a SMALL development-only
sample of the frozen workload-v2 (held-out split is never consumed).

Key properties demonstrated offline with mocks:
  - Gemma uses ONLY the opt-in code-fence normalization (config gate + GEMMA_MODEL_ID check);
    DeepSeek is unchanged (no fence, no normalization).
  - Hard request + token budgets, completion-token cap, timeout, no retry, no fallback, and
    fail-closed stopping via ``BudgetedTransport`` + ``RunBudgets``.
  - Separate configured vs endpoint-reported model identity (``unverified_mismatch`` preserved).
  - Independent five-axis scoring (integrity / routing / interpretation / authority / task
    success), with complete accounting for made calls, skipped calls, and failures.

No live inference, credentials, or network activity occur here.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .experiment_engine import (
    DEEPSEEK_ENDPOINT_ID,
    DEEPSEEK_MOCK_USAGE,
    DESIGNATED_MODEL_DEEPSEEK,
    DESIGNATED_MODEL_GEMMA,
    DEV_PER_FAMILY,
    DEFAULT_REPETITIONS,
    GEMMA_ENDPOINT_ID,
    GEMMA_MOCK_USAGE,
    LIVE_SAFE_FAMILIES,
    OBSERVED_OFFLINE,
    STRATEGY_BOUNDARY,
    STRATEGY_LLM_FIRST,
    EXPECTED_ROUTE,
    ObservationState,
    ExperimentRecord,
    RunBudgets,
    BudgetExceeded,
    BudgetedTransport,
    Split,
    WorkloadCase,
    WorkloadManifest,
    _classify_model_identity,
    _interpretation_dict,
    _mock_interpretation,
    _RecordingTransport,
    build_workload_v2,
    evidence_id,
    interpretation_is_correct,
    verify_records,
)
from .inference import InferenceAdapter, InferenceConfig
from .live_experiment import build_evidence
from .router import route

# Small development-only sample: N dev cases per routing family (held-out never touched).
SAMPLE_PER_FAMILY = 2

# Expected model-call counts per case, by strategy and family. LLM-first always consults the model
# (1 call). BOUNDARY is deterministic-first: only financial_discrepancy routes to INFERENCE (1
# call); deterministic_match and missing_purchase_order resolve with zero model calls.
EXPECTED_CALLS = {
    STRATEGY_LLM_FIRST: {
        "deterministic_match": 1,
        "financial_discrepancy": 1,
        "missing_purchase_order": 1,
    },
    STRATEGY_BOUNDARY: {
        "deterministic_match": 0,
        "financial_discrepancy": 1,
        "missing_purchase_order": 0,
    },
}

# Hard per-run budget margins (request headroom + token headroom) so the exact expected calls fit
# cleanly without tripping the stop condition; the budget is a CAP, not an exact count.
REQUEST_HEADROOM = 1
TOKEN_HEADROOM = 512


def select_dev_sample(
    manifest: WorkloadManifest, per_family: int = SAMPLE_PER_FAMILY
) -> List[WorkloadCase]:
    """Return a deterministic, development-only sample: the first ``per_family`` dev cases per
    family. Never includes held-out cases."""
    by_family: Dict[str, List[WorkloadCase]] = {}
    for wc in manifest.cases:
        if wc.split == Split.DEVELOPMENT:
            by_family.setdefault(wc.family, []).append(wc)
    sample: List[WorkloadCase] = []
    for family in LIVE_SAFE_FAMILIES:
        sample.extend(by_family.get(family, [])[:per_family])
    return sample


def expected_request_count(sample: List[WorkloadCase], strategy: str) -> int:
    """The exact number of outbound model calls a strategy should make over the sample."""
    return sum(EXPECTED_CALLS[strategy][wc.family] for wc in sample)


def expected_token_count(sample: List[WorkloadCase], strategy: str, model_id: str) -> int:
    per_call = (
        GEMMA_MOCK_USAGE["total_tokens"]
        if model_id == DESIGNATED_MODEL_GEMMA
        else DEEPSEEK_MOCK_USAGE["total_tokens"]
    )
    return expected_request_count(sample, strategy) * per_call


def _build_transport(case_map: Dict[str, str], model_id: str):
    """A source-consistent mock transport. Gemma wraps its JSON in a ```json fence (S4E.12/13
    observed behavior); DeepSeek returns plain JSON."""
    is_gemma = model_id == DESIGNATED_MODEL_GEMMA
    endpoint = GEMMA_ENDPOINT_ID if is_gemma else DEEPSEEK_ENDPOINT_ID
    usage = GEMMA_MOCK_USAGE if is_gemma else DEEPSEEK_MOCK_USAGE

    def transport(payload: dict) -> dict:
        user_msg = payload["messages"][1]["content"]
        body = case_map.get(user_msg, "{}")
        if is_gemma:
            body = f"```json\n{body}\n```"
        return {"model": endpoint, "choices": [{"message": {"content": body}}], "usage": usage}

    return transport


def _run_cell(
    model_id: str,
    strategy: str,
    sample: List[WorkloadCase],
    rep: int,
    run_id: str,
    request_limit: int,
    token_limit: int,
) -> Tuple[List[ExperimentRecord], RunBudgets, Optional[str]]:
    """Run one (model, strategy, repetition) cell offline, fail-closed on budget exhaustion."""
    case_map = {build_evidence(wc.case): _mock_interpretation(wc.case) for wc in sample}
    inner = _build_transport(case_map, model_id)
    recording = _RecordingTransport(inner)
    budgets = RunBudgets(request_limit=request_limit, token_limit=token_limit)
    budgeted = BudgetedTransport(recording, budgets)
    config = InferenceConfig(
        model_id=model_id,
        code_fence_normalization=(model_id == DESIGNATED_MODEL_GEMMA),
        timeout_seconds=30.0,
    )
    adapter = InferenceAdapter(config, budgeted)

    records: List[ExperimentRecord] = []
    stop_reason: Optional[str] = None
    for wc in sample:
        # Reset the recording transport so a zero-call case does not inherit a stale endpoint id
        # (or content) from a previous case's call.
        recording.last_endpoint_model = None
        recording.last_content = None
        eid = evidence_id(run_id, model_id, strategy, rep, wc.case_id)
        try:
            if strategy == STRATEGY_BOUNDARY:
                decision = route(wc.case, evidence_text=build_evidence(wc.case), adapter=adapter)
                ev = decision.evidence
                records.append(
                    ExperimentRecord(
                        run_id=run_id, model_id=model_id, strategy=strategy, repetition=rep,
                        case_id=wc.case_id, family=wc.family, split=wc.split.value,
                        state=ObservationState.COMPLETED, evidence_id=eid,
                        route=decision.route.value,
                        route_correct=(decision.route.value == EXPECTED_ROUTE[wc.family]),
                        model_calls=ev.model_calls,
                        advisory_outcome=(ev.advisory_outcome.value if ev.advisory_outcome else None),
                        advisory_consistent=decision.advisory_consistent,
                        interpretation=_interpretation_dict(decision.interpretation),
                        prompt_tokens=ev.prompt_tokens,
                        completion_tokens=ev.completion_tokens,
                        total_tokens=ev.total_tokens,
                        elapsed_ms=ev.elapsed_ms,
                        endpoint_model_id=recording.last_endpoint_model,
                        model_verified=_classify_model_identity(recording.last_endpoint_model, model_id),
                        execution_authority=decision.execution_authority,
                        provenance=OBSERVED_OFFLINE,
                    )
                )
            else:  # STRATEGY_LLM_FIRST: consult the model for every case.
                result = adapter.interpret(build_evidence(wc.case))
                interp = _interpretation_dict(result.interpretation)
                records.append(
                    ExperimentRecord(
                        run_id=run_id, model_id=model_id, strategy=strategy, repetition=rep,
                        case_id=wc.case_id, family=wc.family, split=wc.split.value,
                        state=ObservationState.COMPLETED, evidence_id=eid,
                        route=None, route_correct=None,
                        model_calls=1,
                        advisory_outcome=result.outcome.value,
                        advisory_consistent=(
                            interpretation_is_correct(interp, wc.case)
                            if interp is not None else None
                        ),
                        interpretation=interp,
                        prompt_tokens=result.telemetry.prompt_tokens,
                        completion_tokens=result.telemetry.completion_tokens,
                        total_tokens=result.telemetry.total_tokens,
                        elapsed_ms=result.telemetry.elapsed_ms,
                        endpoint_model_id=recording.last_endpoint_model,
                        model_verified=_classify_model_identity(recording.last_endpoint_model, model_id),
                        execution_authority="NONE",
                        provenance=OBSERVED_OFFLINE,
                    )
                )
        except BudgetExceeded as exc:
            stop_reason = str(exc)
            records.append(
                ExperimentRecord(
                    run_id=run_id, model_id=model_id, strategy=strategy, repetition=rep,
                    case_id=wc.case_id, family=wc.family, split=wc.split.value,
                    state=ObservationState.FAILED, evidence_id=eid, provenance=OBSERVED_OFFLINE,
                )
            )
            break  # fail-closed: stop this cell, do not continue burning budget
    return records, budgets, stop_reason


def run_preflight(
    manifest: WorkloadManifest,
    repetitions: int = DEFAULT_REPETITIONS,
    run_id: str = "s4e14-preflight",
) -> Tuple[List[ExperimentRecord], Dict[str, dict]]:
    """Run the full matrix (2 models x 2 strategies x reps) offline over the dev sample.

    Returns the records plus per-cell budget/stop accounting."""
    sample = select_dev_sample(manifest)
    records: List[ExperimentRecord] = []
    accounting: Dict[str, dict] = {}
    for model_id in (DESIGNATED_MODEL_DEEPSEEK, DESIGNATED_MODEL_GEMMA):
        for strategy in (STRATEGY_LLM_FIRST, STRATEGY_BOUNDARY):
            exp_req = expected_request_count(sample, strategy)
            exp_tok = expected_token_count(sample, strategy, model_id)
            for rep in range(repetitions):
                cell_records, budgets, stop_reason = _run_cell(
                    model_id, strategy, sample, rep, run_id,
                    request_limit=exp_req + REQUEST_HEADROOM,
                    token_limit=exp_tok + TOKEN_HEADROOM,
                )
                records.extend(cell_records)
                accounting[f"{model_id}|{strategy}|rep{rep}"] = {
                    "expected_requests": exp_req,
                    "expected_tokens": exp_tok,
                    "made_requests": budgets.requests,
                    "charged_tokens": budgets.tokens,
                    "rejected_requests": budgets.rejected_requests,
                    "stop_reason": stop_reason,
                }
    return records, accounting


def account_calls(records: List[ExperimentRecord]) -> dict:
    """Summarize made vs skipped model calls and failures across the matrix."""
    made = sum(r.model_calls or 0 for r in records if r.state != ObservationState.FAILED)
    failures = sum(1 for r in records if r.state == ObservationState.FAILED)
    # Skipped = LLM-first opportunity (1 call/case) minus BOUNDARY actual calls (deterministic +
    # held resolve with zero calls). Summed over every BOUNDARY record.
    boundary_skipped = 0
    for r in records:
        if r.strategy == STRATEGY_BOUNDARY and r.state != ObservationState.FAILED:
            boundary_skipped += 1 - (r.model_calls or 0)  # 1 avoided where model_calls == 0
    return {
        "made_calls": made,
        "skipped_calls_boundary": boundary_skipped,
        "failed_cells": failures,
    }


def estimate_live_budget(manifest: WorkloadManifest, repetitions: int = DEFAULT_REPETITIONS) -> dict:
    """Estimated MAXIMUM live request/token budget over the full development split (42 cases).

    ESTIMATES only; not measured. Completion tokens are capped at 512 per request (request-side).
    """
    dev = [wc for wc in manifest.cases if wc.split == Split.DEVELOPMENT]
    n_dev = len(dev)
    n_disc = sum(1 for wc in dev if wc.family == "financial_discrepancy")
    n_models = 2

    llm_first_requests = n_dev * n_models * repetitions
    boundary_requests = n_disc * n_models * repetitions
    total_requests = llm_first_requests + boundary_requests

    # Per-model request split and estimated token totals (mock usage as per-call estimate).
    deepseek_calls = (n_dev + n_disc) * repetitions
    gemma_calls = (n_dev + n_disc) * repetitions
    deepseek_tokens = deepseek_calls * DEEPSEEK_MOCK_USAGE["total_tokens"]
    gemma_tokens = gemma_calls * GEMMA_MOCK_USAGE["total_tokens"]

    return {
        "scope": f"development split only ({n_dev} cases; held-out excluded)",
        "llm_first_requests": llm_first_requests,
        "boundary_requests": boundary_requests,
        "max_requests": total_requests,
        "completion_token_cap_per_request": 512,
        "completion_upper_bound": total_requests * 512,
        "estimated_total_tokens": deepseek_tokens + gemma_tokens,
        "estimated_deepseek_tokens": deepseek_tokens,
        "estimated_gemma_tokens": gemma_tokens,
        "timeout_seconds_per_request": 30.0,
        "note": "ESTIMATE — request counts are exact from the matrix; tokens use recorded mock "
                "usage and the 512 completion cap; prompt tokens vary with evidence length.",
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the offline comparative preflight and print a concise report (no network)."""
    import argparse

    parser = argparse.ArgumentParser(description="S4E.14 controlled two-model comparative preflight (offline)")
    parser.add_argument("--out", help="optional path to write the full evidence JSON")
    args = parser.parse_args(argv)

    manifest = build_workload_v2()
    sample = select_dev_sample(manifest)
    records, accounting = run_preflight(manifest)
    verdicts, vsummary = verify_records(manifest, records)

    print(f"sample: {len(sample)} dev-only cases ({[wc.case_id for wc in sample]})")
    print(f"records: {len(records)} (2 models x 2 strategies x {DEFAULT_REPETITIONS} reps x {len(sample)} cases)")
    print(f"call accounting: {account_calls(records)}")
    print(f"verification: {vsummary}")
    print("live budget estimate:")
    for k, v in estimate_live_budget(manifest).items():
        print(f"  {k}: {v}")

    if args.out:
        from dataclasses import asdict

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "workload_version": manifest.version,
                    "sample_case_ids": [wc.case_id for wc in sample],
                    "total_records": len(records),
                    "accounting": accounting,
                    "call_accounting": account_calls(records),
                    "verification_summary": vsummary,
                    "live_budget_estimate": estimate_live_budget(manifest),
                },
                f, indent=2, sort_keys=True,
            )
        print(f"evidence written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
