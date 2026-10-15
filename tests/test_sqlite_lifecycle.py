"""S4E.23 — SQLite resource lifecycle regression (Windows file-lock).

The accepted batch/web application opens the demo journal (``boundary.db``) and the
batch journal (``boundary.db.batch.db``) through :class:`ActionJournal`. On Windows an
unclosed SQLite connection keeps the database file locked, so a later ``TemporaryDirectory``
(or pytest ``tmp_path``) cleanup fails with ``PermissionError [WinError 32]``.

These tests prove the journals are fully closed after (a) a double-start + TestClient
walkthrough and (b) a failure during journal setup, so the files can be removed.
"""

import os

import pytest
from fastapi.testclient import TestClient

from boundary.app import batch_paths, create_app, initiate_batch
from boundary.demo import run_demo
from boundary.ledger import ActionJournal

RUN_ID = "s4f14-public-offline-v1"


def _remove_or_fail(path: str) -> None:
    """Remove a SQLite file, failing loudly if a Windows file lock is still held.

    On Windows, ``os.remove`` of a database with an open connection raises
    ``PermissionError`` (WinError 32); a correctly closed journal must be removable.
    Companion ``-wal``/``-shm``/``-journal`` files are also removed.
    """
    os.remove(path)
    for suffix in ("-wal", "-shm", "-journal"):
        companion = path + suffix
        if os.path.exists(companion):
            os.remove(companion)


def test_journals_released_after_double_start_and_testclient(tmp_path, monkeypatch):
    """Double-start (stable run id) + full GET walkthrough must leave no open handles."""
    db = str(tmp_path / "boundary.db")
    monkeypatch.setenv("BOUNDARY_DB_PATH", db)

    run_demo(db)
    initiate_batch(db, run_id=RUN_ID)
    initiate_batch(db, run_id=RUN_ID)  # idempotent second start (no re-dispatch)

    client = TestClient(create_app(db))
    for route in ("/", "/health", "/cases/CASE-1", "/evidence", "/batch", "/comparison"):
        assert client.get(route).status_code == 200
    client.close()

    # Both journals must be fully closed and removable now.
    _remove_or_fail(db)                        # demo journal
    _remove_or_fail(batch_paths(db)["journal"])  # batch journal


def test_actionjournal_does_not_leak_connection_on_setup_failure(tmp_path, monkeypatch):
    """A failure during journal setup must not leak the open SQLite connection."""
    db = str(tmp_path / "boundary.db")

    def _boom(_conn):
        raise RuntimeError("setup failed")

    monkeypatch.setattr("boundary.ledger.reconcile_registry", _boom)

    with pytest.raises(RuntimeError):
        ActionJournal(db)

    # __init__ opened the connection, then closed it on failure: the file is removable.
    _remove_or_fail(db)


def test_actionjournal_context_manager_closes(tmp_path):
    """The context manager closes the connection on normal exit and on exception."""
    db = str(tmp_path / "boundary.db")

    with ActionJournal(db) as journal:
        assert journal.count() == 0
    _remove_or_fail(db)

    with pytest.raises(RuntimeError):
        with ActionJournal(db) as journal:
            raise RuntimeError("boom")
    _remove_or_fail(db)
