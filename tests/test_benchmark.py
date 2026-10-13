"""B5.1 offline benchmark harness tests."""

import pytest

from boundary.benchmark import (
    COUNTERFACTUAL,
    NOT_CAPTURED,
    BenchmarkCase,
    build_matrix,
    real_adapter,
    render_report,
    run_benchmark,
    run_case,
    summarize,
    _discrepancy_case,
    _injection_content,
)
from boundary.inference import InferenceOutcome, InferenceUnavailable
from boundary.router import Route


def test_matrix_has_all_six_categories():
    categories = {c.category for c in build_matrix()}
    assert categories == {
        "deterministic_match",
        "financial_discrepancy",
        "missing_purchase_order",
        "malformed_evidence",
        "prompt_injection",
        "unavailable_inference",
    }


def test_all_categories_pass():
    matrix = build_matrix()
    observations = run_benchmark(matrix, repetitions=2)
    assert all(o.passed for o in observations)
    assert summarize(matrix, observations)["failed"] == 0


def test_expected_routes_derived_from_contracts():
    expected = {c.category: c.expected_route for c in build_matrix()}
    assert expected["deterministic_match"] == Route.DETERMINISTIC
    assert expected["missing_purchase_order"] == Route.HELD
    assert expected["financial_discrepancy"] == Route.INFERENCE


def test_expected_advisory_outcomes_specified():
    expected = {c.category: c.expected_advisory_outcome for c in build_matrix()}
    assert expected["deterministic_match"] is None
    assert expected["financial_discrepancy"] == InferenceOutcome.SUCCESS
    assert expected["malformed_evidence"] == InferenceOutcome.MALFORMED_RESPONSE
    assert expected["prompt_injection"] == InferenceOutcome.AUTHORITY_CLAIM
    assert expected["unavailable_inference"] == InferenceOutcome.UNAVAILABLE


def test_invocation_counts_observable():
    observations = run_benchmark(build_matrix(), repetitions=1)
    by_cat = {o.category: o for o in observations}
    assert by_cat["deterministic_match"].model_calls == 0
    assert by_cat["deterministic_match"].actual_calls == 0
    assert by_cat["missing_purchase_order"].model_calls == 0
    assert by_cat["financial_discrepancy"].model_calls == 1


def test_no_accepted_authority_violation():
    observations = run_benchmark(build_matrix(), repetitions=3)
    assert all(o.execution_authority == "NONE" for o in observations)


def test_authority_claim_attempt_rejected_not_accepted():
    observations = run_benchmark(build_matrix(), repetitions=1)
    injection = next(o for o in observations if o.category == "prompt_injection")
    assert injection.authority_claim_attempt is True
    assert injection.advisory_outcome == "AUTHORITY_CLAIM"
    assert injection.passed is True  # rejected, never accepted


def test_incorrect_outcome_fails_even_with_correct_route_and_calls():
    # Negative regression: route and call counts are correct, but the actual
    # advisory outcome (UNAVAILABLE) differs from the expected (AUTHORITY_CLAIM).
    case = BenchmarkCase(
        category="prompt_injection",
        case=_discrepancy_case(),
        evidence_text="synthetic",
        expected_route=Route.INFERENCE,
        expected_calls=1,
        expected_advisory_outcome=InferenceOutcome.AUTHORITY_CLAIM,
        adapter=real_adapter(exc=InferenceUnavailable("down")),
        expected_consistent=False,
    )
    obs = run_case(case, 0)
    assert obs.route_correct is True
    assert obs.calls_correct is True
    assert obs.outcome_correct is False
    assert obs.passed is False


def test_real_parser_detects_authority_key():
    # The prompt-injection scenario exercises the real parser/validator, not a
    # canned outcome.
    adapter = real_adapter(content=_injection_content())
    result = adapter.interpret("synthetic")
    assert result.outcome == InferenceOutcome.AUTHORITY_CLAIM


def test_real_parser_detects_malformed():
    adapter = real_adapter(content="this is not valid json {")
    result = adapter.interpret("synthetic")
    assert result.outcome == InferenceOutcome.MALFORMED_RESPONSE


def test_counterfactual_baseline_labeled_and_computed():
    matrix = build_matrix()
    summary = summarize(matrix, run_benchmark(matrix, repetitions=3))
    c = summary["counterfactual"]
    assert c["label"] == COUNTERFACTUAL
    assert c["counterfactual_llm_calls"] == len(matrix)
    assert c["actual_offline_invocations"] == 4  # four inference cases
    assert c["call_opportunity_avoided"] == 2  # deterministic + missing-PO


def test_token_telemetry_not_captured_offline():
    observations = run_benchmark(build_matrix(), repetitions=1)
    assert all(o.token_provenance == NOT_CAPTURED for o in observations)


def test_authority_claim_scenarios_vs_observations():
    matrix = build_matrix()
    summary = summarize(matrix, run_benchmark(matrix, repetitions=3))
    a = summary["authority"]
    assert a["claim_scenarios"] == 1  # one synthetic scenario
    assert a["claim_observations"] == 3  # three repetitions
    assert a["accepted_violations"] == 0


def test_render_report_sections_present():
    matrix = build_matrix()
    summary = summarize(matrix, run_benchmark(matrix, repetitions=1))
    report = render_report(summary)
    assert "Counterfactual" in report
    assert "Authority enforcement" in report
    assert "NOT_CAPTURED" in report
    assert "not real AMD" in report


def test_deterministic_across_repetitions():
    matrix = build_matrix()
    routes1 = {o.category: o.actual_route for o in run_benchmark(matrix, repetitions=1)}
    routes2 = {o.category: o.actual_route for o in run_benchmark(matrix, repetitions=1)}
    assert routes1 == routes2


def test_invalid_repetitions_rejected():
    with pytest.raises(ValueError):
        run_benchmark(build_matrix(), repetitions=0)
    with pytest.raises(ValueError):
        run_benchmark(build_matrix(), repetitions=-1)


def test_elapsed_timing_is_nanoseconds():
    observations = run_benchmark(build_matrix(), repetitions=1)
    assert all(isinstance(o.elapsed_ns, int) for o in observations)
    assert all(o.elapsed_ns >= 0 for o in observations)
