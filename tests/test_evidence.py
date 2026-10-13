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
    assert "127" in t
    assert "118" in t
    assert "245" in t
    assert "3974" in t
    assert "DeepSeek-V4-Flash" in t
    # Offline benchmark numbers.
    assert "6" in t
    assert "18" in t
    assert "2 of 6" in t
    assert "3" in t  # repetitions per category


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


def test_limitations_clarify_single_inference_response(tmp_path):
    client = _client(tmp_path)
    t = client.get("/evidence").text
    assert "only one genuine AMD inference response" in t


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
