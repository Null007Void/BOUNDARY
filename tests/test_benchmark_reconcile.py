"""B5.3 evidence reconciliation tests (offline, synthetic fixtures)."""

import json

from boundary.benchmark_reconcile import (
    OBSERVED_LIVE,
    OBSERVED_OFFLINE,
    LIVE_ZERO_REQUEST,
    build_summary,
    main,
    reconcile,
    render_report,
    validate_record,
)

FINANCIAL = {
    "scenario": "financial_discrepancy",
    "provenance": OBSERVED_LIVE,
    "transport_kind": "live",
    "dry_run": False,
    "route": "INFERENCE",
    "expected_route": "INFERENCE",
    "reason": "advisory interpretation consistent with source facts",
    "model_calls": 1,
    "outbound_requests": 1,
    "confirmed_response": True,
    "model_id": "DeepSeek-V4-Flash",
    "prompt_tokens": 127,
    "completion_tokens": 118,
    "total_tokens": 245,
    "elapsed_ms": 3974,
    "harness_elapsed_ns": 123456,
    "advisory_outcome": "SUCCESS",
    "advisory_consistent": True,
    "execution_authority": "NONE",
    "route_correct": True,
    "authority_preserved": True,
    "outbound_correct": True,
    "harness_safe": True,
    "measurement_success": True,
}

EXACT = {
    "scenario": "exact_match",
    "provenance": LIVE_ZERO_REQUEST,
    "transport_kind": "live",
    "dry_run": False,
    "route": "DETERMINISTIC",
    "expected_route": "DETERMINISTIC",
    "reason": "exact match handled deterministically; zero model calls",
    "model_calls": 0,
    "outbound_requests": 0,
    "confirmed_response": False,
    "model_id": "DeepSeek-V4-Flash",
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
    "elapsed_ms": None,
    "harness_elapsed_ns": 148300,
    "advisory_outcome": None,
    "advisory_consistent": None,
    "execution_authority": "NONE",
    "route_correct": True,
    "authority_preserved": True,
    "outbound_correct": True,
    "harness_safe": True,
    "measurement_success": True,
}

MISSING = {
    "scenario": "missing_po",
    "provenance": LIVE_ZERO_REQUEST,
    "transport_kind": "live",
    "dry_run": False,
    "route": "HELD",
    "expected_route": "HELD",
    "reason": "purchase order unresolved by trusted lookup",
    "model_calls": 0,
    "outbound_requests": 0,
    "confirmed_response": False,
    "model_id": "DeepSeek-V4-Flash",
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
    "elapsed_ms": None,
    "harness_elapsed_ns": 159400,
    "advisory_outcome": None,
    "advisory_consistent": None,
    "execution_authority": "NONE",
    "route_correct": True,
    "authority_preserved": True,
    "outbound_correct": True,
    "harness_safe": True,
    "measurement_success": True,
}

ALL = [FINANCIAL, EXACT, MISSING]


def _with(record, **overrides):
    out = dict(record)
    out.update(overrides)
    return out


# --- acceptance of valid records ---------------------------------------------------


def test_valid_records_accepted():
    result = reconcile(ALL)
    assert result.all_valid is True
    assert all(vr.valid for vr in result.records)
    assert result.cross_record_issues == []


def test_summary_has_measured_and_counterfactual():
    summary = build_summary(reconcile(ALL))
    assert summary["validated"] is True
    assert summary["measured"]["total_tokens"] == 245
    assert summary["measured"]["elapsed_ms"] == 3974
    assert summary["measured"]["amd_requests"] == 1
    assert summary["offline_benchmark"]["llm_first_comparator_provenance"] == "COUNTERFACTUAL"
    assert summary["offline_benchmark"]["routing_observations_provenance"] == "OBSERVED_OFFLINE"
    assert summary["offline_benchmark"]["call_opportunities_avoided"] == 2
    assert "attestation_note" in summary


def test_report_mentions_limitations_and_separation():
    report = render_report(reconcile(ALL), build_summary(reconcile(ALL)))
    assert "Limitations" in report
    assert "NOT comparable" in report
    assert "COUNTERFACTUAL" in report
    assert "245" in report  # measured tokens present
    assert "one observation each" in report


# --- rejection of missing / duplicate / unknown ------------------------------------


def test_missing_scenario_rejected():
    result = reconcile([FINANCIAL, EXACT])
    assert result.all_valid is False
    assert any("missing scenario: missing_po" in i for i in result.cross_record_issues)


def test_duplicate_scenario_rejected():
    result = reconcile([FINANCIAL, FINANCIAL, EXACT, MISSING])
    assert result.all_valid is False
    assert any("duplicate scenario financial_discrepancy" in i for i in result.cross_record_issues)


def test_unknown_scenario_rejected():
    result = reconcile([_with(FINANCIAL, scenario="bogus"), EXACT, MISSING])
    assert result.all_valid is False


# --- rejection of misleading / false-success records ------------------------------


def test_offline_provenance_rejected():
    # A record claiming success but with offline provenance is misleading.
    vr = validate_record(_with(FINANCIAL, provenance=OBSERVED_OFFLINE))
    assert vr.valid is False
    assert any("provenance" in i for i in vr.issues)


def test_route_mismatch_rejected():
    vr = validate_record(_with(EXACT, route="INFERENCE"))
    assert vr.valid is False
    assert any("route" in i for i in vr.issues)


def test_authority_violation_rejected():
    vr = validate_record(_with(FINANCIAL, execution_authority="APPROVE"))
    assert vr.valid is False
    assert any("authority" in i for i in vr.issues)


def test_false_measurement_success_rejected():
    # measurement_success=True but the advisory outcome is not SUCCESS.
    vr = validate_record(_with(FINANCIAL, advisory_outcome="AUTH_FAILURE"))
    assert vr.valid is False
    assert any("advisory_outcome" in i for i in vr.issues)


def test_telemetry_inconsistency_rejected():
    vr = validate_record(_with(FINANCIAL, total_tokens=999))
    assert vr.valid is False
    assert any("total_tokens" in i for i in vr.issues)


def test_zero_request_with_tokens_rejected():
    vr = validate_record(_with(EXACT, prompt_tokens=5))
    assert vr.valid is False
    assert any("prompt_tokens" in i for i in vr.issues)


def test_missing_tokens_rejected():
    vr = validate_record(_with(FINANCIAL, prompt_tokens=None))
    assert vr.valid is False
    assert any("prompt_tokens" in i for i in vr.issues)


def test_malformed_record_rejected():
    vr = validate_record(["not", "a", "dict"])
    assert vr.valid is False


# --- internal-consistency / contradiction rejection ---------------------------------


def test_internally_contradictory_record_rejected():
    # The exact defect from review: internally contradictory fields must not pass.
    bad = _with(FINANCIAL, dry_run=True, transport_kind="mock",
                route_correct=False, authority_preserved=False)
    vr = validate_record(bad)
    assert vr.valid is False
    joined = " ".join(vr.issues)
    assert "dry_run" in joined
    assert "transport_kind" in joined
    assert "route_correct" in joined
    assert "authority_preserved" in joined


def test_transport_kind_mock_rejected():
    vr = validate_record(_with(FINANCIAL, transport_kind="mock"))
    assert vr.valid is False
    assert any("transport_kind" in i for i in vr.issues)


def test_dry_run_rejected():
    vr = validate_record(_with(FINANCIAL, dry_run=True))
    assert vr.valid is False
    assert any("dry_run" in i for i in vr.issues)


def test_route_correct_false_rejected():
    vr = validate_record(_with(FINANCIAL, route_correct=False))
    assert vr.valid is False
    assert any("route_correct" in i for i in vr.issues)


def test_authority_preserved_false_rejected():
    vr = validate_record(_with(FINANCIAL, authority_preserved=False))
    assert vr.valid is False
    assert any("authority_preserved" in i for i in vr.issues)


def test_outbound_correct_false_rejected():
    vr = validate_record(_with(FINANCIAL, outbound_correct=False))
    assert vr.valid is False
    assert any("outbound_correct" in i for i in vr.issues)


def test_expected_route_mismatch_rejected():
    vr = validate_record(_with(FINANCIAL, expected_route="HELD"))
    assert vr.valid is False
    assert any("expected_route" in i for i in vr.issues)


def test_boolean_as_count_rejected():
    # A boolean True must not be accepted as integer 1 in request/count fields.
    vr = validate_record(_with(FINANCIAL, outbound_requests=True))
    assert vr.valid is False
    assert any("outbound_requests" in i for i in vr.issues)


def test_negative_harness_elapsed_rejected():
    vr = validate_record(_with(FINANCIAL, harness_elapsed_ns=-1))
    assert vr.valid is False
    assert any("harness_elapsed_ns" in i for i in vr.issues)


def test_zero_request_advisory_fabrication_rejected():
    vr = validate_record(_with(EXACT, advisory_outcome="SUCCESS"))
    assert vr.valid is False
    assert any("advisory_outcome" in i for i in vr.issues)

    vr2 = validate_record(_with(EXACT, advisory_consistent=True))
    assert vr2.valid is False
    assert any("advisory_consistent" in i for i in vr2.issues)


def test_zero_request_configured_model_id_preserved():
    # The real zero-request files carry a configured model_id; it must be accepted.
    vr = validate_record(EXACT)
    assert vr.valid is True
    assert EXACT["model_id"] == "DeepSeek-V4-Flash"


def test_summary_does_not_declare_success_for_contradictory():
    bad = _with(FINANCIAL, dry_run=True, transport_kind="mock",
                route_correct=False, authority_preserved=False)
    result = reconcile([bad, EXACT, MISSING])
    summary = build_summary(result)
    assert result.all_valid is False
    assert summary["validated"] is False
    assert summary["scenarios"]["financial_discrepancy"]["valid"] is False
    assert "measured" not in summary


def test_contradictory_cli_nonzero(tmp_path):
    bad = tmp_path / "fin.json"
    bad.write_text(json.dumps(_with(FINANCIAL, dry_run=True, transport_kind="mock",
                                     route_correct=False, authority_preserved=False)), encoding="utf-8")
    e = tmp_path / "exact.json"
    e.write_text(json.dumps(EXACT), encoding="utf-8")
    m = tmp_path / "missing.json"
    m.write_text(json.dumps(MISSING), encoding="utf-8")
    assert main(["--records", str(bad), str(e), str(m)]) == 1


# --- CLI ---------------------------------------------------------------------------


def test_cli_accepts_three_files(tmp_path):
    paths = []
    for record in ALL:
        p = tmp_path / f"{record['scenario']}.json"
        p.write_text(json.dumps(record), encoding="utf-8")
        paths.append(str(p))
    rc = main(["--records", *paths])
    assert rc == 0


def test_cli_rejects_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_with(FINANCIAL, provenance=OBSERVED_OFFLINE)), encoding="utf-8")
    good1 = tmp_path / "exact.json"
    good1.write_text(json.dumps(EXACT), encoding="utf-8")
    good2 = tmp_path / "missing.json"
    good2.write_text(json.dumps(MISSING), encoding="utf-8")
    rc = main(["--records", str(bad), str(good1), str(good2)])
    assert rc == 1


def test_cli_missing_file(tmp_path):
    rc = main(["--records", str(tmp_path / "nope.json")])
    assert rc == 1
