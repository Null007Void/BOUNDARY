"""B6 S4A deployment-startup tests (offline).

Verifies the Railway startup path: a clean seed of the synthetic journal into a
temporary directory using Railway-like environment variables, and that the served
routes require no live credentials and expose no mutation endpoints.
"""

import os

from fastapi.testclient import TestClient

from boundary.demo import run_demo
from boundary.ledger import ActionJournal
from boundary.serve import _parse_port, build_app


def test_clean_startup_seeds_and_serves(tmp_path, monkeypatch):
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)
    monkeypatch.setenv("PORT", "8080")  # Railway-like port

    app = build_app()  # mirrors python -m boundary.serve
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/cases/CASE-2").status_code == 200
    assert client.get("/evidence").status_code == 200

    h = client.get("/health").json()
    assert h["journal"] == "ready"
    assert h["amd_connected"] is False
    assert h["evolus_connected"] is False


def test_startup_idempotent(tmp_path, monkeypatch):
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)

    build_app()
    j = ActionJournal(db)
    first = j.count()
    j.close()

    build_app()  # second startup must not duplicate the exact-match commit
    j = ActionJournal(db)
    second = j.count()
    j.close()
    assert second == first == 1


def test_startup_requires_no_credentials(tmp_path, monkeypatch):
    # No AMD/Evolus credential env vars are needed for the offline demo to run.
    for var in ("BOUNDARY_AMD_API_KEY", "BOUNDARY_EVOLUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)

    app = build_app()
    assert TestClient(app).get("/health").status_code == 200


def test_no_mutation_endpoints(tmp_path, monkeypatch):
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)
    client = TestClient(build_app())
    # Only GET is served; no anonymous mutation endpoint exists.
    for path in ("/", "/evidence", "/cases/CASE-1"):
        assert client.post(path).status_code == 405
        assert client.put(path).status_code == 405


def test_parse_port_valid():
    assert _parse_port("8000") == (8000, None)
    assert _parse_port("1") == (1, None)
    assert _parse_port("65535") == (65535, None)


def test_parse_port_invalid():
    for raw in ("abc", "0", "-1", "70000", "", "12.5"):
        port, err = _parse_port(raw)
        assert port is None
        assert err is not None
        assert "1-65535" in err


def test_invalid_port_has_no_journal_side_effect(tmp_path, monkeypatch):
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)
    monkeypatch.setenv("PORT", "70000")
    monkeypatch.setattr("boundary.serve.uvicorn.run", lambda *a, **k: None)

    import boundary.serve as serve

    assert serve.main() == 1
    assert not os.path.exists(db)  # PORT rejected before any journal initialization


def test_valid_port_runs_startup_and_seeds(tmp_path, monkeypatch):
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)
    monkeypatch.setenv("PORT", "8080")
    captured = {}
    monkeypatch.setattr("boundary.serve.uvicorn.run", lambda app, **kw: captured.update(kw))

    import boundary.serve as serve

    assert serve.main() == 0
    assert os.path.exists(db)  # journal seeded
    assert captured["port"] == 8080
