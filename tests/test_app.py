import os

import pytest
from fastapi.testclient import TestClient

from boundary.app import create_app
from boundary.demo import demo_cases, run_demo
from boundary.ledger import ActionJournal
from boundary.verifier import VerificationOutcome, verify


def test_run_demo_end_to_end(tmp_path):
    path = str(tmp_path / "demo.db")
    results = run_demo(path)

    assert len(results) == 3

    exact = results[0]
    assert exact["label"] == "valid exact match"
    assert exact["proposal"].exact_match_eligible is True
    assert exact["guard_action"] == "APPROVE"
    assert exact["commit_status"] == "COMMITTED"
    assert exact["receipt"] is not None
    assert exact["verification"].outcome == VerificationOutcome.VERIFIED

    disc = results[1]
    assert disc["commit_status"] == "HELD"
    assert disc["guard_action"] == "HOLD"
    assert disc["receipt"] is None
    assert disc["verification"].outcome == VerificationOutcome.NOT_ELIGIBLE

    unavail = results[2]
    assert unavail["commit_status"] == "HELD"
    assert unavail["guard_action"] == "HOLD"
    assert unavail["receipt"] is None
    assert unavail["verification"].outcome == VerificationOutcome.NOT_ELIGIBLE

    j = ActionJournal(path)
    try:
        assert j.count() == 1  # exactly one durable record
    finally:
        j.close()


def test_demo_idempotent(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    run_demo(path)  # second run must not duplicate the exact-match action
    j = ActionJournal(path)
    try:
        assert j.count() == 1
    finally:
        j.close()


def test_health_ready_after_demo(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    client = TestClient(create_app(path))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["journal"] == "ready"
    assert body["offline"] is True
    assert body["amd_connected"] is False
    assert body["evolus_connected"] is False


def test_health_uninitialized(tmp_path):
    client = TestClient(create_app(str(tmp_path / "demo.db")))  # no DB present
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "uninitialized"
    assert body["journal"] == "uninitialized"


def test_health_unavailable(tmp_path):
    path = str(tmp_path / "demo.db")
    with open(path, "w") as f:
        f.write("not a database")
    client = TestClient(create_app(path))
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["journal"] == "unavailable"


def test_get_does_not_create_database(tmp_path):
    path = str(tmp_path / "demo.db")
    assert not os.path.exists(path)
    client = TestClient(create_app(path))
    r = client.get("/")
    assert r.status_code == 200
    assert "uninitialized" in r.text.lower()
    assert not os.path.exists(path)  # GET must not create the database
    client.get("/cases/CASE-1")
    assert not os.path.exists(path)


def test_case_list_and_detail(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)  # populate the journal first
    client = TestClient(create_app(path))

    r = client.get("/")
    assert r.status_code == 200
    assert "CASE-1" in r.text and "CASE-2" in r.text and "CASE-3" in r.text

    d1 = client.get("/cases/CASE-1")
    assert d1.status_code == 200
    assert "APPROVE" in d1.text
    assert "VERIFIED" in d1.text

    d2 = client.get("/cases/CASE-2")
    assert d2.status_code == 200
    assert "HOLD" in d2.text
    assert "no committed action" in d2.text.lower()

    d3 = client.get("/cases/CASE-3")
    assert d3.status_code == 200
    assert "HOLD" in d3.text
    assert "no committed action" in d3.text.lower()

    assert client.get("/cases/NOPE").status_code == 404


def test_web_is_read_only(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    j = ActionJournal(path)
    before = j.count()
    j.close()

    client = TestClient(create_app(path))
    client.get("/")
    client.get("/cases/CASE-1")
    client.get("/cases/CASE-2")
    client.get("/cases/CASE-3")
    client.get("/health")

    j = ActionJournal(path)
    after = j.count()
    j.close()
    assert after == before  # read-only web requests never commit


def test_no_anonymous_mutation_endpoint(tmp_path):
    client = TestClient(create_app(str(tmp_path / "demo.db")))
    assert client.post("/").status_code == 405
    assert client.post("/cases/CASE-1").status_code == 405


def test_verify_after_reopen(tmp_path):
    path = str(tmp_path / "demo.db")
    run_demo(path)
    j = ActionJournal(path)
    try:
        label, case = demo_cases()[0]  # the exact-match case
        v = verify(j, case)
        assert v.outcome == VerificationOutcome.VERIFIED
    finally:
        j.close()
