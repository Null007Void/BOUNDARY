"""S4F.16 — production evidence hotfix regressions.

Verifies: (1) the /experiment checksum verifies across LF and CRLF line endings and fails
closed on substantive tampering; (2) the /evidence page shows demonstrated Gemma history with
correct figures and transparency; (3) the integrated Gemma evaluation is explicitly
incomplete; (4) the public app remains offline with no live-inference endpoint.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

import boundary.public_experiment as pe
from boundary.app import create_app
from boundary.serve import build_app


def _canonical_evidence():
    out = {}
    for fn in pe._EXPECTED_SHA256:
        data = (pe._DATA_ROOT / fn).read_bytes()
        out[fn] = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return out


def _offline_text():
    return (pe._DATA_ROOT.parent / "s4e10_offline_summary.json").read_text(encoding="utf-8")


def _stage(tmp_path, ending):
    """Write the three pinned files (LF or CRLF) + the offline summary into tmp_path."""
    data_root = tmp_path / "s4e20"
    data_root.mkdir(parents=True)
    for fn, data in _canonical_evidence().items():
        if ending == "crlf":
            data = data.replace(b"\n", b"\r\n")
        (data_root / fn).write_bytes(data)
    (tmp_path / "s4e10_offline_summary.json").write_text(_offline_text(), encoding="utf-8")
    return data_root


# --- A: /experiment checksum portability + tamper fail-closed ----------------- #

def test_experiment_loads_lf_and_crlf(tmp_path, monkeypatch):
    for ending in ("lf", "crlf"):
        data_root = _stage(tmp_path / ending, ending)
        monkeypatch.setattr(pe, "_DATA_ROOT", data_root)
        d = pe.load_public_experiment()
        assert d["experiment_id"] == "s4e15-live"
        assert len(d["paths"]) == 4


def test_tampered_evidence_fails_closed(tmp_path, monkeypatch):
    data_root = _stage(tmp_path / "tampered", "lf")
    monkeypatch.setattr(pe, "_DATA_ROOT", data_root)
    # Substantively alter a field (not line endings).
    p = data_root / "two_model_evidence.json"
    p.write_text(p.read_text().replace('"s4e15-live"', '"tampered"', 1), encoding="utf-8")
    with pytest.raises(ValueError):
        pe.load_public_experiment()


# --- B: Gemma evidence figures + transparency --------------------------------- #

def test_gemma_demonstration_metrics():
    g = pe.gemma_demonstration()
    assert g["status"] == "DEMONSTRATED — RECORDED LIVE"
    assert g["llm_first"] == {"requests": 6, "tokens": 1614,
                              "source_consistent": 5, "source_inconsistent": 1}
    assert g["boundary"] == {"requests": 2, "tokens": 572,
                             "source_consistent": 2, "source_inconsistent": 0}
    assert g["total_requests"] == 8
    assert g["source_consistent_overall"] == 7
    assert g["authority"] == "NONE"


@pytest.fixture
def client(tmp_path):
    db = str(tmp_path / "b.db")
    os.environ["BOUNDARY_DB_PATH"] = db
    return TestClient(build_app())


def test_evidence_page_shows_demonstrated_gemma(client):
    text = client.get("/evidence").text
    assert "DEMONSTRATED" in text
    assert "INCOMPLETE" not in text
    assert "MALFORMED_RESPONSE" not in text
    assert "1614" in text and "572" in text
    assert "5/6" in text and "2/2" in text
    assert "/experiment" in text


def test_gemma_history_disclosed(client):
    text = client.get("/evidence").text
    # Initial parsing issue and later catalog unavailability are disclosed.
    assert "code-fenced JSON" in text
    assert "normalization" in text
    assert "model catalog" in text


def test_integrated_gemma_incomplete(client):
    text = client.get("/experiment").text
    # The integrated four-path run is partial; Gemma BOUNDARY not dispatched.
    assert "partial" in text.lower()
    assert "Not executed" in text


def test_public_app_remains_offline(client):
    # No live AMD/Evolus inference endpoint exists in the public candidate.
    assert client.post("/live-demo", headers={"x-boundary-operator-token": "x"}).status_code == 404
    # The offline intake endpoint is still present (deterministic, no provider).
    assert client.get("/try").status_code == 200


def test_experiment_page_renders(client):
    resp = client.get("/experiment")
    assert resp.status_code == 200
    assert "Gemma-4-31B-IT" in resp.text


def test_amd_demonstration_metrics():
    a = pe.amd_demonstration()
    assert a["deepseek"] == {"requests": 8, "tokens": 1754}
    assert a["gemma"] == {"requests": 8, "tokens": 2186}
    assert a["combined"] == {"requests": 16, "tokens": 3940}
    assert a["llm_first"] == {"requests": 12, "tokens": 2930}
    assert a["boundary"] == {"requests": 4, "tokens": 1010}
    assert a["request_reduction_percent"] == 66.7
    assert a["token_reduction_percent"] == 65.5
    assert a["authority"] == "NONE"


def test_evidence_shows_s4e15_amd_primary_and_s4e5_pilot(client):
    text = client.get("/evidence").text
    assert "16 requests" in text
    assert "3,940 tokens" in text or "3940" in text
    assert "S4E.15" in text
    assert "S4E.5 earlier pilot" in text
    assert "66.7%" in text
    assert "65.5%" in text


def test_cross_page_numerical_agreement(client):
    # Homepage, Evidence, and Experiment all derive the same S4E.15 reductions.
    for text in (client.get("/").text, client.get("/evidence").text,
                 client.get("/experiment").text):
        assert "66.7%" in text
        assert "65.5%" in text
    assert "16 requests" in client.get("/evidence").text


def test_try_intake_disabled_inference_wording(client):
    # A discrepancy requires inference, but the public app is intentionally offline:
    # it says so plainly and confirms no model was called for the visitor's invoice.
    resp = client.post("/evaluate", data={
        "invoice_id": "INV-701", "supplier_id": "SUP-SYN-100",
        "po_reference": "PO-SYN-100", "amount": "1100.00",
    })
    assert resp.status_code == 200
    assert "LIVE INFERENCE IS INTENTIONALLY DISABLED" in resp.text
    assert "No model was called" in resp.text
    assert "/evidence" in resp.text
    assert "/experiment" in resp.text


def test_batch_uses_simulated_demonstration_label(client):
    # Batch/comparison views use finished-product wording, not development-slice wording,
    # and the empty batch view still renders the corrected label.
    text = client.get("/batch").text
    assert "synthetic batch demonstration" in text.lower()
    assert "offline development slice" not in text.lower()
