"""S4E.3 live-experiment preflight regression tests (offline, mock-only).

Covers the eight original verifications, the B3 payload contract, and the five
evidence-integrity review corrections (isolated state, confirmation-by-behavior, mock
provenance, shared-budget wiring, baseline source consistency, lookup terminology).
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from boundary.inference import (
    EvidenceInterpretation,
    InferenceAdapter,
    InferenceAuthError,
    InferenceConfig,
    InferenceTimeout,
)
from boundary.benchmark_live import SCENARIOS
from boundary.live_experiment import (
    APPROVED_AMD_ENDPOINT,
    DESIGNATED_MODEL,
    LIVE_PROVENANCE,
    LIVE_UNCONFIRMED_PROVENANCE,
    LIVE_ZERO_REQUEST_PROVENANCE,
    MOCK_PROVENANCE,
    SAFE_SCENARIOS,
    ExperimentTransport,
    LiveConfigError,
    RequestBudget,
    baseline_source_consistency,
    build_live_experiment,
    build_mock_transport,
    build_preflight,
    main,
    run_experiment,
    scenario_evidence,
    validate_live_config,
    verify_model_identity,
    _live_main,
    _new_run_id,
    _write_exclusive,
)


def _run(model=DESIGNATED_MODEL, content=None, exc=None):
    adapter, transport, budget = build_preflight(model=model, content=content, exc=exc)
    events, route_records = run_experiment(adapter)
    return events, route_records, budget


# --- Original verifications --------------------------------------------------------

# 1. All three safe scenarios have complete, case-derived synthetic evidence.
def test_all_three_scenarios_have_evidence():
    for name in SAFE_SCENARIOS:
        ev = scenario_evidence(name)
        assert ev and ev.strip()


def test_evidence_derived_from_case_facts():
    em = scenario_evidence("exact_match")
    assert "INV-100" in em and "SUP-100" in em and "PO-100" in em
    assert "match" in em.lower()
    fd = scenario_evidence("financial_discrepancy")
    assert "INV-2048" in fd and "SUP-9001" in fd and "PO-77" in fd
    assert "ambiguous" in fd.lower()
    mp = scenario_evidence("missing_po")
    assert "INV-300" in mp
    assert "found no purchase order" in mp
    assert em != fd and fd != mp and em != mp


# 2. Corresponding prompts match for the shared discrepancy request.
def test_corresponding_prompts_match():
    events, _, _ = _run()
    a = [e for e in events if e.strategy == "A" and e.scenario == "financial_discrepancy"]
    b = [e for e in events if e.strategy == "B" and e.scenario == "financial_discrepancy"]
    assert len(a) == 2 and len(b) == 2
    for rep in range(2):
        assert a[rep].payload["messages"] == b[rep].payload["messages"]


# 3. Exactly six baseline and two BOUNDARY requests are scheduled.
def test_exact_request_schedule():
    events, route_records, budget = _run()
    assert sum(1 for e in events if e.strategy == "A") == 6
    assert sum(1 for e in events if e.strategy == "B") == 2
    assert len(events) == 8
    assert budget.attempted == 8
    assert budget.rejected == 0
    rr = {(r["scenario"], r["repetition"]): r for r in route_records}
    assert all(rr[(s, rep)]["requests_made"] == 0 for s in ("exact_match", "missing_po") for rep in range(2))
    assert all(rr[("financial_discrepancy", rep)]["requests_made"] == 1 for rep in range(2))


# 4. A ninth request is rejected.
def test_ninth_request_rejected():
    adapter, transport, budget = build_preflight()
    run_experiment(adapter)
    assert budget.attempted == 8
    result = adapter.interpret(scenario_evidence("financial_discrepancy"))
    assert result.outcome.value == "UNAVAILABLE"
    assert budget.rejected == 1


# 5. Missing or mismatched model identity is never reported verified.
def test_model_identity_classification():
    assert verify_model_identity(None) == "unverified_missing"
    assert verify_model_identity("SomeOtherModel") == "unverified_mismatch"
    assert verify_model_identity(DESIGNATED_MODEL) == "verified"


def test_mismatched_model_identity_unverified():
    events, _, _ = _run(model="Wrong-Model-7B")
    for e in events:
        assert e.endpoint_model_id == "Wrong-Model-7B"
        assert e.model_verified == "unverified_mismatch"


def test_missing_model_identity_unverified():
    def no_model_transport(payload):
        return {
            "choices": [{"message": {"content": '{"supplier_id":"SUP-9001","invoice_id":"INV-2048","po_reference":"PO-77","amount_cents":125000,"discrepancy_cents":500,"confidence":0.9,"notes":""}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    budget = RequestBudget()
    transport = ExperimentTransport(no_model_transport, budget)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    events, _ = run_experiment(adapter)
    for e in events:
        assert e.endpoint_model_id is None
        assert e.model_verified == "unverified_missing"


# 6. Timeouts and malformed responses are recorded and fail closed.
def test_timeout_recorded_fail_closed():
    events, _, _ = _run(exc=InferenceTimeout("timeout"))
    for e in events:
        assert e.outcome == "TIMEOUT"
        assert e.confirmed is False


def test_malformed_response_recorded():
    events, _, _ = _run(content="this is not valid json {")
    for e in events:
        assert e.outcome == "MALFORMED_RESPONSE"


# 7. No model response can authorize execution.
def test_no_model_response_authorizes_execution():
    events, route_records, _ = _run()
    for e in events:
        assert e.execution_authority == "NONE"
    for rr in route_records:
        assert rr["execution_authority"] == "NONE"


# 8. A dry run cannot contact a real endpoint, even with credentials present.
def test_dry_run_no_live_even_with_credentials(monkeypatch):
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", "fake-secret-key")
    adapter, transport, budget = build_preflight()
    assert callable(transport._transport)
    assert not isinstance(transport._transport, httpx.BaseTransport)
    run_experiment(adapter)
    assert budget.attempted == 8


# 9. The accepted B3 payload contract is used.
def test_b3_payload_parameters():
    events, _, _ = _run()
    for e in events:
        assert e.payload["temperature"] == 0.0
        assert e.payload["max_tokens"] == 512
        assert e.payload["model"] == DESIGNATED_MODEL


# --- Review corrections ------------------------------------------------------------

# Correction 1: a budget-rejected request never inherits prior state.
def test_rejected_request_no_stale_state():
    budget = RequestBudget(limit=1)
    transport = ExperimentTransport(build_mock_transport(), budget)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    adapter.interpret(scenario_evidence("financial_discrepancy"))
    assert transport.last_endpoint_model == DESIGNATED_MODEL
    result = adapter.interpret(scenario_evidence("exact_match"))
    assert result.outcome.value == "UNAVAILABLE"
    assert transport.last_endpoint_model is None  # not inherited
    # The payload reflects the rejected request's own evidence, not the prior request's.
    assert transport.last_payload is not None
    assert "INV-100" in transport.last_payload["messages"][1]["content"]


# Correction 2: confirmation is based on observed transport behavior, not outcomes.
def test_auth_failure_not_confirmed():
    events, _, _ = _run(exc=InferenceAuthError("auth failed"))
    for e in events:
        assert e.outcome == "AUTH_FAILURE"
        assert e.confirmed is False  # transport raised; no response was returned


def test_missing_telemetry_stays_missing():
    events, _, _ = _run()  # default mock omits usage
    for e in events:
        assert e.prompt_tokens is None
        assert e.completion_tokens is None
        assert e.total_tokens is None


# Correction 3: the runner cannot operate with mismatched transport/budget wiring.
def test_mismatched_transport_rejected():
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), build_mock_transport())
    with pytest.raises(TypeError):
        run_experiment(adapter)


# Correction 4: baseline source consistency is distinguished from schema parsing.
def test_baseline_source_consistency_case_appropriate():
    events, _, _ = _run()
    a0 = {e.scenario: e for e in events if e.strategy == "A" and e.repetition == 0}
    assert a0["exact_match"].source_consistency == "source_consistent"
    assert a0["financial_discrepancy"].source_consistency == "source_consistent"
    assert a0["missing_po"].source_consistency == "not_verifiable"


def test_baseline_wrong_interpretation_source_inconsistent():
    wrong = json.dumps(
        {"supplier_id": "SUP-9001", "invoice_id": "INV-2048", "po_reference": "PO-77",
         "amount_cents": 125000, "discrepancy_cents": 500, "confidence": 0.9, "notes": ""}
    )
    events, _, _ = _run(content=wrong)
    a0 = {e.scenario: e for e in events if e.strategy == "A" and e.repetition == 0}
    # Schema parses (SUCCESS) but refers to the wrong invoice -> source_inconsistent.
    assert a0["exact_match"].outcome == "SUCCESS"
    assert a0["exact_match"].source_consistency == "source_inconsistent"
    assert a0["financial_discrepancy"].source_consistency == "source_consistent"
    # A wrong interpretation on a missing-PO case contradicts known invoice facts now,
    # so it is source_inconsistent (not merely not_verifiable).
    assert a0["missing_po"].source_consistency == "source_inconsistent"


# Correction 5: NOT_FOUND is not conflated with UNAVAILABLE.
def test_missing_po_not_found_terminology():
    mp = scenario_evidence("missing_po")
    assert "found no purchase order" in mp
    assert "unavailable" not in mp.lower()


# Final correction: baseline source consistency validates invoice facts before PO state.
def _missing_po_case():
    return SCENARIOS["missing_po"].case


def _interp(supplier="SUP-300", invoice="INV-300", po_ref=None, amount=1000, disc=0):
    return EvidenceInterpretation(
        supplier_id=supplier, invoice_id=invoice, po_reference=po_ref,
        amount_cents=amount, discrepancy_cents=disc, confidence=0.5, notes="",
    )


def test_missing_po_wrong_supplier_source_inconsistent():
    r = baseline_source_consistency(_missing_po_case(), _interp(supplier="SUP-999"))
    assert r == "source_inconsistent"


def test_missing_po_wrong_invoice_source_inconsistent():
    r = baseline_source_consistency(_missing_po_case(), _interp(invoice="INV-999"))
    assert r == "source_inconsistent"


def test_missing_po_wrong_amount_source_inconsistent():
    r = baseline_source_consistency(_missing_po_case(), _interp(amount=9999))
    assert r == "source_inconsistent"


def test_missing_po_wrong_po_reference_source_inconsistent():
    r = baseline_source_consistency(_missing_po_case(), _interp(po_ref="PO-999"))
    assert r == "source_inconsistent"


def test_missing_po_correct_invoice_facts_not_verifiable():
    r = baseline_source_consistency(_missing_po_case(), _interp())
    assert r == "not_verifiable"


# Final-final correction: with a trusted PO present, an omitted reference is not enough.
def _exact_match_case():
    return SCENARIOS["exact_match"].case


def _found_interp(po_ref, amount=5000, disc=0):
    return EvidenceInterpretation(
        supplier_id="SUP-100", invoice_id="INV-100", po_reference=po_ref,
        amount_cents=amount, discrepancy_cents=disc, confidence=0.9, notes="",
    )


def test_found_po_omitted_reference_source_inconsistent():
    r = baseline_source_consistency(_exact_match_case(), _found_interp(po_ref=None))
    assert r == "source_inconsistent"


def test_found_po_wrong_reference_source_inconsistent():
    r = baseline_source_consistency(_exact_match_case(), _found_interp(po_ref="PO-999"))
    assert r == "source_inconsistent"


def test_found_po_correct_reference_source_consistent():
    r = baseline_source_consistency(_exact_match_case(), _found_interp(po_ref="PO-100"))
    assert r == "source_consistent"


# --- S4E.4 live-execution wiring ---------------------------------------------------

# 1. Live config validation refuses unsafe configuration before any network.
def test_validate_live_config_missing_api_key():
    cfg = InferenceConfig(api_key="", model_id=DESIGNATED_MODEL, base_url=APPROVED_AMD_ENDPOINT)
    err = validate_live_config(cfg)
    assert err is not None and "API_KEY" in err


def test_validate_live_config_wrong_model():
    cfg = InferenceConfig(api_key="k", model_id="WrongModel", base_url=APPROVED_AMD_ENDPOINT)
    assert "model_id" in validate_live_config(cfg)


def test_validate_live_config_wrong_endpoint():
    cfg = InferenceConfig(api_key="k", model_id=DESIGNATED_MODEL, base_url="https://evil.example.com/v1")
    assert "base_url" in validate_live_config(cfg)


def test_validate_live_config_valid():
    cfg = InferenceConfig(api_key="k", model_id=DESIGNATED_MODEL, base_url=APPROVED_AMD_ENDPOINT)
    assert validate_live_config(cfg) is None


# 2. Live run refuses before network when credentials/endpoint are absent or invalid.
def test_live_run_refused_without_credentials(monkeypatch, capsys):
    for var in ("BOUNDARY_AMD_API_KEY", "BOUNDARY_AMD_MODEL_ID", "BOUNDARY_AMD_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert main(["--live"]) == 1
    assert "refused" in capsys.readouterr().out


def test_live_run_refused_wrong_endpoint(monkeypatch, capsys):
    monkeypatch.setenv("BOUNDARY_AMD_API_KEY", "k")
    monkeypatch.setenv("BOUNDARY_AMD_MODEL_ID", DESIGNATED_MODEL)
    monkeypatch.setenv("BOUNDARY_AMD_BASE_URL", "https://wrong.example.com/v1")
    assert main(["--live"]) == 1
    assert "refused" in capsys.readouterr().out


def test_build_live_experiment_refuses_invalid_config(monkeypatch):
    for var in ("BOUNDARY_AMD_API_KEY", "BOUNDARY_AMD_MODEL_ID", "BOUNDARY_AMD_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LiveConfigError):
        build_live_experiment()


# 3. Provenance distinguishes live vs mock, and confirmed vs unconfirmed vs zero-request.
def test_live_provenance_confirmed_and_zero_request():
    budget = RequestBudget()
    transport = ExperimentTransport(build_mock_transport(), budget, live=True)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    events, route_records = run_experiment(adapter)
    for e in events:
        assert e.provenance == LIVE_PROVENANCE
        assert e.confirmed is True
    rr = {(r["scenario"], r["repetition"]): r for r in route_records}
    for s in ("exact_match", "missing_po"):
        for rep in range(2):
            assert rr[(s, rep)]["provenance"] == LIVE_ZERO_REQUEST_PROVENANCE
    for rep in range(2):
        assert rr[("financial_discrepancy", rep)]["provenance"] is None


def test_live_unconfirmed_provenance():
    budget = RequestBudget()
    transport = ExperimentTransport(build_mock_transport(exc=InferenceTimeout("t")), budget, live=True)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    events, _ = run_experiment(adapter)
    for e in events:
        assert e.provenance == LIVE_UNCONFIRMED_PROVENANCE
        assert e.confirmed is False


def test_mock_provenance_offline():
    adapter, transport, budget = build_preflight()
    events, _ = run_experiment(adapter)
    for e in events:
        assert e.provenance == MOCK_PROVENANCE


# 4. Output isolation: mock artifacts are preserved and distinct from live output.
def test_mock_main_writes_offline_artifacts(tmp_path):
    jp = tmp_path / "mock.json"
    rp = tmp_path / "mock.md"
    assert main(["--json", str(jp), "--report", str(rp)]) == 0
    assert jp.exists() and rp.exists()
    assert "mock preflight" in json.loads(jp.read_text())["provenance"]


def test_live_refusal_writes_no_output(tmp_path, monkeypatch, capsys):
    for var in ("BOUNDARY_AMD_API_KEY", "BOUNDARY_AMD_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    jp = tmp_path / "live.json"
    rp = tmp_path / "live.md"
    assert main(["--live", "--json", str(jp), "--report", str(rp)]) == 1
    assert "refused" in capsys.readouterr().out
    assert not jp.exists() and not rp.exists()


# 5. Accidental live-mode is impossible: default invocation is mock-only.
def test_default_invocation_is_mock(tmp_path):
    jp = tmp_path / "r.json"
    rp = tmp_path / "r.md"
    assert main(["--json", str(jp), "--report", str(rp)]) == 0
    assert json.loads(jp.read_text())["provenance"] == "OBSERVED_OFFLINE (mock preflight)"


# 6. The budget is per-invocation, not a durable cross-process limit.
def test_budget_is_per_instance_not_durable():
    b1 = RequestBudget()
    b2 = RequestBudget()
    assert b1.attempted == 0 and b2.attempted == 0
    b1.acquire()
    assert b1.attempted == 1 and b2.attempted == 0


# --- S4E.4 evidence durability -----------------------------------------------------

# 1. Run identifiers are unique (repeated runs produce distinct output, no overwrite).
def test_new_run_id_unique():
    assert _new_run_id() != _new_run_id()


# 2. Exclusive-create fails on a preexisting file (no silent overwrite).
def test_write_exclusive_fails_on_preexisting(tmp_path):
    p = tmp_path / "r.json"
    _write_exclusive(str(p), {"a": 1})
    with pytest.raises(FileExistsError):
        _write_exclusive(str(p), {"a": 2})


# 3. on_event fires incrementally, once per request event, in order.
def test_run_experiment_on_event_incremental():
    adapter, transport, budget = build_preflight()
    captured = []
    events, _ = run_experiment(adapter, on_event=captured.append)
    assert len(captured) == 8
    assert [e.strategy for e in captured] == [e.strategy for e in events]
    assert sum(1 for e in captured if e.strategy == "A") == 6
    assert sum(1 for e in captured if e.strategy == "B") == 2


# 4. Interruption preserves the events already attempted (incremental, not lost).
def test_incremental_preservation_on_interruption():
    calls = {"n": 0}

    def interrupting_transport(payload):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt()
        return {"model": DESIGNATED_MODEL, "choices": [{"message": {"content": "{}"}}]}

    budget = RequestBudget()
    transport = ExperimentTransport(interrupting_transport, budget)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    captured = []
    with pytest.raises(KeyboardInterrupt):
        run_experiment(adapter, on_event=captured.append)
    assert len(captured) == 3  # the three completed requests' evidence is preserved


# 5. A partial run is marked "partial" and never combined with completed claims.
def test_live_main_partial_status_on_interruption(tmp_path, monkeypatch):
    calls = {"n": 0}

    def interrupting_transport(payload):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt()
        return {"model": DESIGNATED_MODEL, "choices": [{"message": {"content": "{}"}}]}

    budget = RequestBudget()
    transport = ExperimentTransport(interrupting_transport, budget, live=True)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    monkeypatch.setattr("boundary.live_experiment.build_live_experiment", lambda: (adapter, transport, budget))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        _live_main(SimpleNamespace(json=None, report=None))

    partial = list(tmp_path.glob("*_partial.json"))
    assert len(partial) == 1
    data = json.loads(partial[0].read_text())
    assert data["status"] == "partial"
    assert data["attempted"] == 4
    assert len(data["events"]) == 3
    # No completed results file was written for an interrupted run.
    assert list(tmp_path.glob("*_results.json")) == []


# 6. The report is plain Markdown, not an escaped JSON string.
def test_live_report_is_markdown_not_json(tmp_path, monkeypatch):
    adapter, transport, budget = build_preflight()
    monkeypatch.setattr("boundary.live_experiment.build_live_experiment", lambda: (adapter, transport, budget))
    monkeypatch.chdir(tmp_path)

    json_path = tmp_path / "out.json"
    report_path = tmp_path / "out.md"
    assert _live_main(SimpleNamespace(json=str(json_path), report=str(report_path))) == 0

    report = report_path.read_text()
    assert report.startswith("# BOUNDARY")          # Markdown heading
    assert "- " in report                            # Markdown list item
    assert not report.startswith('"')                # not a JSON-escaped string
    assert "\\n" not in report                       # no literal escaped newlines
    data = json.loads(json_path.read_text())         # JSON artifact stays valid JSON
    assert data["status"] == "completed"


# 7. Preexisting output destinations are rejected BEFORE any outbound request.
def test_live_refuses_preexisting_output_before_requests(tmp_path, monkeypatch):
    existing = tmp_path / "out.json"
    existing.write_text("preexisting")

    requests = {"n": 0}

    def counting_transport(payload):
        requests["n"] += 1
        return {"model": DESIGNATED_MODEL, "choices": [{"message": {"content": "{}"}}]}

    budget = RequestBudget()
    transport = ExperimentTransport(counting_transport, budget, live=True)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    monkeypatch.setattr("boundary.live_experiment.build_live_experiment", lambda: (adapter, transport, budget))
    monkeypatch.chdir(tmp_path)

    result = _live_main(SimpleNamespace(json=str(existing), report=str(tmp_path / "out.md")))
    assert result == 1                                # refused
    assert requests["n"] == 0                          # no outbound request consumed
    assert existing.read_text() == "preexisting"      # existing evidence preserved
