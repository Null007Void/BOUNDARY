"""S4E.2 efficiency-comparison regression tests (offline).

Maps to the required verifications: same inputs, designated model, deterministic
avoidance, held evidence, no authority, no authority bypass, no mislabeled-live, no
credentials, B5 unchanged, reproducibility — plus the review corrections: provenance
split, live-experiment specification, request-budget arithmetic, consistent inputs.
"""

import json
from dataclasses import asdict

from boundary.benchmark import COUNTERFACTUAL, OBSERVED_OFFLINE, build_matrix
from boundary.efficiency_comparison import (
    DESIGNATED_MODEL,
    LIVE_EXPERIMENT_SPEC,
    main,
    run_comparison,
    summarize,
)


def _records():
    return run_comparison(build_matrix())


# 1. Both strategies use the same synthetic input set.
def test_same_input_set_for_both_strategies():
    cases = build_matrix()
    records = run_comparison(cases)
    assert [r.category for r in records] == [c.category for c in cases]
    for r in records:
        assert r.llm_first_call_opportunities == 1
        assert r.boundary_route in ("DETERMINISTIC", "HELD", "INFERENCE")


# 2. DeepSeek is the designated model for both strategies.
def test_designated_model_is_deepseek_for_both():
    for r in _records():
        assert r.designated_model == DESIGNATED_MODEL


# 3. Deterministic cases avoid unnecessary inference.
def test_deterministic_cases_avoid_inference():
    det = {r.category: r for r in _records()}["deterministic_match"]
    assert det.disposition == "avoided"
    assert det.boundary_route == "DETERMINISTIC"
    assert det.boundary_adapter_invocations == 0
    assert det.calls_avoided == 1


# 4. Insufficient trusted evidence remains held.
def test_insufficient_evidence_is_held():
    missing = {r.category: r for r in _records()}["missing_purchase_order"]
    assert missing.disposition == "held"
    assert missing.boundary_route == "HELD"
    assert missing.boundary_adapter_invocations == 0


# 5. Advisory outputs cannot authorize actions.
def test_advisory_outputs_cannot_authorize_actions():
    for r in _records():
        assert r.execution_authority == "NONE"


# 6. Baseline authority-claim attempts do not bypass checks.
def test_authority_claim_does_not_bypass_checks():
    injection = {r.category: r for r in _records()}["prompt_injection"]
    assert injection.advisory_outcome == "AUTHORITY_CLAIM"
    assert injection.advisory_consistent is False
    assert injection.execution_authority == "NONE"


# 7. Mock measurements are never mislabeled as live AMD measurements.
def test_mock_measurements_not_labeled_live():
    for r in _records():
        assert r.boundary_provenance == OBSERVED_OFFLINE
        assert r.llm_first_provenance == COUNTERFACTUAL
    s = summarize(_records())
    assert s["strategy_b"]["provenance"] == OBSERVED_OFFLINE
    assert s["strategy_a"]["provenance"] == COUNTERFACTUAL
    assert "RECORDED_LIVE" not in (s["strategy_a"]["provenance"], s["strategy_b"]["provenance"])


# 8. No credentials or external network calls are required.
def test_no_credentials_required(monkeypatch):
    for var in (
        "BOUNDARY_AMD_API_KEY",
        "BOUNDARY_EVOLUS_API_KEY",
        "BOUNDARY_EVOLUS_APPLICATION_ID",
        "BOUNDARY_EVOLUS_WORKFLOW_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    records = run_comparison(build_matrix())
    assert len(records) == 6


# 9. Accepted B5 measurements remain unchanged.
def test_b5_matrix_unchanged():
    cases = build_matrix()
    expected = {
        "deterministic_match": "DETERMINISTIC",
        "financial_discrepancy": "INFERENCE",
        "missing_purchase_order": "HELD",
        "malformed_evidence": "INFERENCE",
        "prompt_injection": "INFERENCE",
        "unavailable_inference": "INFERENCE",
    }
    assert {c.category: c.expected_route.value for c in cases} == expected
    records = run_comparison(cases)
    assert sum(r.llm_first_call_opportunities for r in records) == 6
    assert sum(r.boundary_adapter_invocations for r in records) == 4


# 10. Repeated runs produce reproducible offline results.
def test_reproducible_offline_results():
    a = run_comparison(build_matrix())
    b = run_comparison(build_matrix())
    assert [asdict(r) for r in a] == [asdict(r) for r in b]


# --- Review corrections --------------------------------------------------------------

# 11. Provenance is split across record/summary levels; no misleading top-level label.
def test_provenance_split_no_misleading_top_level(tmp_path):
    records = _records()
    # Records carry explicit per-strategy provenance.
    for r in records:
        assert r.llm_first_provenance == COUNTERFACTUAL
        assert r.boundary_provenance == OBSERVED_OFFLINE
    # Summary carries explicit per-strategy provenance, no single top-level label.
    s = summarize(records)
    assert "provenance" not in s  # no misleading top-level designation
    assert s["strategy_a"]["provenance"] == COUNTERFACTUAL
    assert s["strategy_b"]["provenance"] == OBSERVED_OFFLINE
    # The serialized JSON uses a provenance_note, not a top-level provenance label.
    jp = tmp_path / "r.json"
    rp = tmp_path / "r.md"
    assert main(["--json", str(jp), "--report", str(rp)]) == 0
    data = json.loads(jp.read_text())
    assert "provenance" not in data
    assert "provenance_note" in data
    assert "COUNTERFACTUAL" in data["provenance_note"]
    assert "OBSERVED_OFFLINE" in data["provenance_note"]


# 12. The live experiment measures BOTH strategies on the three safe scenarios.
def test_live_spec_measures_both_strategies():
    spec = LIVE_EXPERIMENT_SPEC
    assert set(spec["scenarios"]) == {
        "exact_match", "financial_discrepancy", "missing_purchase_order",
    }
    assert spec["repetitions"] == 2
    assert spec["strategies"]["A"]["name"].startswith("Measured")
    assert spec["strategies"]["B"]["name"].startswith("Measured")
    # Adversarial categories stay offline-only.
    assert set(spec["offline_only_categories"]) == {
        "malformed_evidence", "prompt_injection", "unavailable_inference",
    }


# 13. Request-budget arithmetic is consistent (A=6, B=2, max=8).
def test_request_budget_arithmetic():
    spec = LIVE_EXPERIMENT_SPEC
    reps = spec["repetitions"]
    a = spec["strategies"]["A"]["requests_per_repetition"]
    b = spec["strategies"]["B"]["requests_per_repetition"]
    budget = spec["request_budget"]
    assert a == 3 and b == 1
    assert budget["strategy_a"] == a * reps == 6
    assert budget["strategy_b"] == b * reps == 2
    assert budget["maximum_total"] == budget["strategy_a"] + budget["strategy_b"] == 8
    # The budget is a proposal, not authorization.
    assert "PROPOSED" in budget["note"].upper()


# 14. Live spec uses the accepted three safe B5.2 scenarios, not the six-case matrix.
def test_live_spec_consistent_synthetic_inputs():
    spec = LIVE_EXPERIMENT_SPEC
    assert len(spec["scenarios"]) == 3
    # Evidence inputs are explicitly derived from B5.2 SCENARIOS + SYNTHETIC_EVIDENCE.
    assert "B5.2 SCENARIOS" in spec["derived_request_contract"]["evidence_inputs"]
    assert "SYNTHETIC_EVIDENCE" in spec["derived_request_contract"]["evidence_inputs"]
