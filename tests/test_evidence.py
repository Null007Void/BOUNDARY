"""B6 S3 Evidence & Performance page tests (offline).

Verifies the page presents recorded evidence truthfully: recorded-versus-live
labeling, numeric accuracy, navigation, read-only behavior, authority boundary,
separation of counterfactual from measured data, and absence of secrets or
accidental external calls.
"""

import inspect

from fastapi.testclient import TestClient

from boundary import evidence as evidence_mod
from boundary.app import create_app
from boundary.demo import run_demo


def _client(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    return TestClient(create_app(path))


def test_evidence_page_renders(tmp_path):
    client = _client(tmp_path)
    r = client.get("/evidence")
    assert r.status_code == 200


def test_recorded_live_label_present(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "RECORDED LIVE TEST" in t
    assert "NOT RE-RUN BY THIS PAGE" in t
    # The label appears for both AMD and Evolus sections.
    assert t.count("RECORDED LIVE TEST") >= 2


def test_numeric_accuracy(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    # S4E.5 controlled pilot numbers.
    assert "1314" in t
    assert "473" in t
    assert "66.7%" in t
    assert "64.0%" in t
    assert "DeepSeek-V4-Flash" in t
    assert "self-dploy/DeepSeek-V4-Flash" in t
    assert "unverified_mismatch" in t
    # Offline benchmark numbers.
    assert "18" in t
    assert "2 of 6" in t


def test_evolus_values_present(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    for v in ("INV-2048", "SUP-9001", "PO-77", "USD 1,250.00", "unverified"):
        assert v in t


def test_authority_boundary_prominent(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "No AI authorization" in t
    assert "no payments" in t
    assert "simulated only" in t
    assert "NONE" in t


def test_counterfactual_separated_from_measured(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "COUNTERFACTUAL" in t
    # The counterfactual is explicitly not savings.
    assert "NOT measured token, latency, or cost savings" in t


def test_provenance_data_split_correctly():
    from boundary.evidence import B5_1_COUNTERFACTUAL, B5_1_OBSERVED_OFFLINE
    # OBSERVED_OFFLINE carries the 18 routing observations, NOT the comparison.
    assert B5_1_OBSERVED_OFFLINE["label"] == "OBSERVED_OFFLINE"
    assert B5_1_OBSERVED_OFFLINE["observations"] == 18
    assert B5_1_OBSERVED_OFFLINE["repetitions_per_category"] == 3
    assert "call_opportunities_avoided" not in B5_1_OBSERVED_OFFLINE
    # COUNTERFACTUAL carries the 2-of-6 LLM-first comparison, NOT the observations.
    assert B5_1_COUNTERFACTUAL["label"] == "COUNTERFACTUAL"
    assert B5_1_COUNTERFACTUAL["call_opportunities_avoided"] == 2
    assert B5_1_COUNTERFACTUAL["categories"] == 6
    assert "observations" not in B5_1_COUNTERFACTUAL


def test_provenance_labels_both_present_on_page(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "OBSERVED_OFFLINE" in t
    assert "COUNTERFACTUAL" in t
    assert "Offline Routing Benchmark" in t
    assert "LLM-First Comparison" in t


def test_limitations_clarify_scope_and_authority(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    # Scope is the recorded S4E.15 comparison, not a general-performance claim.
    assert "six synthetic cases per model" in t
    assert "sixteen genuine AMD requests" in t
    assert "not a prediction" in t
    # Simulated, not authenticated human authorization.
    assert "simulated only" in t
    # No end-to-end Evolus-plus-AMD savings claim.
    assert "Evolus-plus-AMD" in t


def test_gemma_limitation_present(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "Gemma-4-31B-IT" in t
    assert "structured-output parsing" in t
    # The integrated Gemma evaluation remains explicitly incomplete and availability-limited.
    assert "not dispatched" in t
    assert "catalog" in t


def test_amd_evidence_s4e5_structure():
    from boundary.evidence import AMD_EVIDENCE
    assert AMD_EVIDENCE["strategy_a"]["requests"] == 6
    assert AMD_EVIDENCE["strategy_a"]["tokens"] == 1314
    assert AMD_EVIDENCE["strategy_b"]["requests"] == 2
    assert AMD_EVIDENCE["strategy_b"]["tokens"] == 473
    assert AMD_EVIDENCE["confirmed_responses"] == 8
    assert AMD_EVIDENCE["request_reduction"] == "66.7%"
    assert AMD_EVIDENCE["token_reduction"] == "64.0%"
    assert AMD_EVIDENCE["endpoint_model_id"] == "self-dploy/DeepSeek-V4-Flash"
    assert AMD_EVIDENCE["model_verified"] == "unverified_mismatch"


def test_no_private_identifiers_or_secrets(tmp_path):
    import re
    client = _client(tmp_path)
    t = client.get("/evidence").text
    for forbidden in ("api.evolus.ai", "X-ApiKey", "X-Application-Id", "sk-"):
        assert forbidden not in t
    # No UUID-like account/workflow identifiers leak onto the page.
    uuid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    assert not uuid_re.search(t)


def test_navigation(tmp_path):
    client = _client(tmp_path)
    assert "/evidence" in client.get("/").text
    assert "/evidence" in client.get("/cases/CASE-1").text
    assert 'href="/"' in client.get("/evidence").text


def test_evidence_read_only(tmp_path):
    client = _client(tmp_path)
    assert client.post("/evidence").status_code == 405


def test_evidence_module_has_no_live_calls():
    src = inspect.getsource(evidence_mod)
    assert "httpx" not in src
    assert "build_amd_transport" not in src
    assert "import requests" not in src
    assert "urlopen" not in src


def test_evidence_status_labels_present(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "status-label--measured" in t
    assert "status-label--demonstrated" in t
    assert "status-label--offline" in t
    assert "status-label--counterfactual" in t
    assert "MEASURED" in t
    assert "DEMONSTRATED" in t
    # The underlying provenance remains, not flattened.
    assert "RECORDED LIVE TEST" in t
    assert "unverified_mismatch" in t


def test_evidence_source_link(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "github.com/Null007Void/BOUNDARY" in t
    assert "MIT licensed" in t
