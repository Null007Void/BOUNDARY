"""Shared cross-path execution registry (B2.2b-2a).

B1 (exact-match ``ActionJournal``) and B2 (``AuthorizedExecutor``) both commit
simulated ``APPROVE`` actions. To prevent duplicate approvals across the two
paths, both write a single claim row per committed action into this shared
``execution_registry`` table in the *same* database file. The table's unique
constraints establish durable, concurrency-safe cross-path uniqueness:

  - ``case_id`` UNIQUE — no two actions (B1 or B2) under the same case id;
  - ``(supplier_id, invoice_id)`` UNIQUE — no two approvals for one invoice.

Separate databases do not share this table and therefore do not provide global
uniqueness; both paths must open the same canonical database for this guarantee
to hold.
"""

from __future__ import annotations

import sqlite3


class RegistryConflictError(Exception):
    """Historical approval records cannot be reconciled into the registry."""


def create_registry(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_registry (
            case_id TEXT NOT NULL UNIQUE,
            supplier_id TEXT NOT NULL,
            invoice_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_exec_registry_invoice "
        "ON execution_registry(supplier_id, invoice_id)"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def reconcile_registry(conn: sqlite3.Connection) -> None:
    """Create the registry and reconcile it against committed B1/B2 actions.

    Reconciles the registry with the *actual* committed actions, not merely
    against duplicates within the action tables:
      - existing registry entries that match a committed action remain valid;
      - missing historical entries are backfilled;
      - registry entries that disagree with a committed action (mismatched
        case->invoice or invoice->case mapping) fail closed;
      - registry entries with no corresponding committed action (orphans) fail
        closed.

    Inspection, consistency checking, and backfill run under one write
    transaction so concurrent startup or writes cannot invalidate the snapshot.
    Fails closed with ``RegistryConflictError`` leaving all records untouched.
    Idempotent on an already-reconciled database.
    """
    create_registry(conn)

    try:
        conn.execute("BEGIN IMMEDIATE")

        # Inspect committed actions.
        collected = []
        if _table_exists(conn, "actions"):
            collected += list(
                conn.execute("SELECT case_id, supplier_id, invoice_id FROM actions").fetchall()
            )
        if _table_exists(conn, "authorized_actions"):
            collected += list(
                conn.execute("SELECT case_id, supplier_id, invoice_id FROM authorized_actions").fetchall()
            )

        # Internal consistency of the committed actions.
        actions_by_case = {}
        actions_by_invoice = {}
        for case_id, supplier_id, invoice_id in collected:
            if case_id in actions_by_case:
                raise RegistryConflictError(
                    f"case id {case_id!r} approved more than once across paths"
                )
            if (supplier_id, invoice_id) in actions_by_invoice:
                raise RegistryConflictError(
                    f"invoice {supplier_id!r}/{invoice_id!r} approved more than once across paths"
                )
            actions_by_case[case_id] = (supplier_id, invoice_id)
            actions_by_invoice[(supplier_id, invoice_id)] = case_id

        # Inspect the existing registry and verify it matches committed actions.
        registry_rows = list(
            conn.execute("SELECT case_id, supplier_id, invoice_id FROM execution_registry").fetchall()
        )
        registry_case_to_invoice = {
            case_id: (supplier_id, invoice_id)
            for case_id, supplier_id, invoice_id in registry_rows
        }
        registry_invoice_to_case = {
            (supplier_id, invoice_id): case_id
            for case_id, supplier_id, invoice_id in registry_rows
        }

        for case_id, mapping in registry_case_to_invoice.items():
            if case_id not in actions_by_case:
                raise RegistryConflictError(
                    f"registry has an orphan entry for case {case_id!r} with no committed action"
                )
            if actions_by_case[case_id] != mapping:
                raise RegistryConflictError(
                    f"registry mapping for case {case_id!r} disagrees with the committed action"
                )
        for (supplier_id, invoice_id), case_id in registry_invoice_to_case.items():
            if (supplier_id, invoice_id) not in actions_by_invoice:
                raise RegistryConflictError(
                    f"registry has an orphan entry for invoice {supplier_id!r}/{invoice_id!r}"
                )
            if actions_by_invoice[(supplier_id, invoice_id)] != case_id:
                raise RegistryConflictError(
                    f"registry mapping for invoice {supplier_id!r}/{invoice_id!r} "
                    "disagrees with the committed action"
                )

        # Backfill missing historical entries.
        for case_id, (supplier_id, invoice_id) in actions_by_case.items():
            if case_id not in registry_case_to_invoice:
                conn.execute(
                    "INSERT INTO execution_registry (case_id, supplier_id, invoice_id) VALUES (?, ?, ?)",
                    (case_id, supplier_id, invoice_id),
                )

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def registry_insert(conn: sqlite3.Connection, case_id: str, supplier_id: str, invoice_id: str) -> None:
    """Insert a cross-path execution claim; raises ``IntegrityError`` on conflict."""
    conn.execute(
        "INSERT INTO execution_registry (case_id, supplier_id, invoice_id) VALUES (?, ?, ?)",
        (case_id, supplier_id, invoice_id),
    )


def registry_case_conflict(conn: sqlite3.Connection, case_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM execution_registry WHERE case_id = ?", (case_id,)
    ).fetchone()
    return row is not None


def registry_invoice_conflict(conn: sqlite3.Connection, supplier_id: str, invoice_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM execution_registry WHERE supplier_id = ? AND invoice_id = ?",
        (supplier_id, invoice_id),
    ).fetchone()
    return row is not None
