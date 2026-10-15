"""Guarded SQLite action journal for BOUNDARY.

Records committed simulated ``APPROVE`` actions durably, with an input
fingerprint for idempotency and conflict detection, and returns a verifiable
:class:`ActionReceipt` linked to the committed record.

Layering (preserved explicitly):
  - a classifier *proposal* is only an eligibility statement;
  - a *guard decision* is the internal permission check (``APPROVE``/``HOLD``);
  - a *committed action* is a durable journal row plus a receipt.

The journal never trusts an external eligibility flag: :meth:`ActionJournal.approve`
re-derives the classification from the case through the guard before any write.
No real payments, balance transfers, or accounting movements are performed — this
is a simulated approval record only.

Persistence-boundary safeguards:
  - amounts outside SQLite's signed 64-bit range fail closed (``AMOUNT_OVERFLOW``)
    — a storage constraint, not a business threshold;
  - a previously approved invoice identity (``supplier_id``, ``invoice_id``)
    cannot be approved again under another case id (``DUPLICATE_INVOICE``, backed
    by a durable unique index);
  - a reused case id with changed input yields ``CONFLICT`` regardless of the new
    input's eligibility;
  - concurrent requests are reconciled via the unique constraints plus
    ``IntegrityError`` recovery: one commit, the rest idempotent/conflict.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .contracts import CaseInput
from .guard import ActionDecision, GuardDecision, evaluate
from .registry import (
    reconcile_registry,
    registry_case_conflict,
    registry_insert,
    registry_invoice_conflict,
)

# SQLite signed 64-bit integer range (a storage constraint, not business policy).
_SQLITE_MIN_INT = -(2 ** 63)
_SQLITE_MAX_INT = (2 ** 63) - 1


def _readonly_uri(path: str) -> str:
    """Build a SQLite read-only URI for an existing database file."""
    return "file:" + path.replace("\\", "/") + "?mode=ro"


def case_fingerprint(case: CaseInput) -> str:
    """Deterministic SHA-256 fingerprint of the full case input.

    Uses ``model_dump()`` with ``sort_keys`` and an exact ``str`` fallback for
    ``Decimal``/enum values, so identical content always hashes identically and
    any change to the input changes the fingerprint.
    """
    payload = json.dumps(case.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ActionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: int
    case_id: str
    action: str
    invoice_id: str
    po_id: str
    supplier_id: str
    amount_cents: int
    input_fingerprint: str
    policy_version: str
    committed_at: str


class CommitStatus(str, Enum):
    COMMITTED = "COMMITTED"          # new approval written
    IDEMPOTENT = "IDEMPOTENT"        # identical retry, existing record returned
    CONFLICT = "CONFLICT"            # same case_id, different input; no overwrite
    HELD = "HELD"                    # not an exact match; no write
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"  # invoice already approved elsewhere
    AMOUNT_OVERFLOW = "AMOUNT_OVERFLOW"      # amount outside SQLite int64 range


class CommitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CommitStatus
    case_id: str
    decision: GuardDecision
    receipt: Optional[ActionReceipt] = None


class ActionJournal:
    """Minimal SQLite journal of committed simulated approvals."""

    def __init__(self, path: str, read_only: bool = False):
        if read_only:
            # Genuinely read-only access: the file must already exist; no
            # initialization, migration, or repair. Raises sqlite3.OperationalError
            # when the journal is missing or not a valid database.
            self._conn = sqlite3.connect(_readonly_uri(path), uri=True)
            self._conn.execute("PRAGMA query_only = ON")
            return
        # Autocommit: each statement is durably committed; single INSERTs are
        # atomic, and concurrency is reconciled via unique constraints.
        self._conn = sqlite3.connect(path, isolation_level=None)
        try:
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    invoice_id TEXT NOT NULL,
                    po_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    committed_at TEXT NOT NULL
                )
                """
            )
            # Durable invoice identity: a given (supplier_id, invoice_id) may only be
            # approved once, even across different case ids.
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_identity "
                "ON actions(supplier_id, invoice_id)"
            )
            # Shared cross-path registry: case-id and invoice uniqueness across both
            # the B1 exact-match path and the B2 authorized-exception path. Backfill
            # any historical approvals that predate the registry.
            reconcile_registry(self._conn)
        except Exception:
            # A failure during schema/registry setup must not leak the open SQLite
            # connection: on Windows an unclosed connection keeps the database file
            # locked, so ``TemporaryDirectory``/test cleanup then fails with
            # ``PermissionError [WinError 32]``.
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ActionJournal":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get(self, case_id: str) -> Optional[ActionReceipt]:
        """Return the committed receipt for a case, or ``None`` if absent."""
        return self._fetch(case_id)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]

    def approve(self, case: CaseInput) -> CommitResult:
        """Evaluate the guard internally and commit an APPROVE if permitted."""
        fingerprint = case_fingerprint(case)
        decision = evaluate(case)  # internal classification; no external flag trusted

        # 1. Reused case id: conflict/idempotency takes precedence over eligibility,
        #    so a changed, now-ineligible input is still reported as a conflict.
        existing = self._fetch(case.case_id)
        if existing is not None:
            if existing.input_fingerprint == fingerprint:
                return CommitResult(status=CommitStatus.IDEMPOTENT, case_id=case.case_id,
                                    decision=decision, receipt=existing)
            return CommitResult(status=CommitStatus.CONFLICT, case_id=case.case_id,
                                decision=decision)

        # 2. Eligibility (guard).
        if decision.action != ActionDecision.APPROVE:
            return CommitResult(status=CommitStatus.HELD, case_id=case.case_id, decision=decision)

        # 3. Storage range (fail closed, no write).
        amount = case.invoice.computed_total_cents
        if amount > _SQLITE_MAX_INT or amount < _SQLITE_MIN_INT:
            return CommitResult(status=CommitStatus.AMOUNT_OVERFLOW, case_id=case.case_id,
                                decision=decision)

        # 4. Invoice identity: no double approval of the same invoice under a
        #    *different* case id. Same-case-id handling (idempotent/conflict,
        #    above) takes precedence and must never be misclassified as a
        #    duplicate, even under a concurrent commit between the two checks.
        if self._invoice_exists(case.invoice.supplier_id, case.invoice.invoice_id, case.case_id):
            return CommitResult(status=CommitStatus.DUPLICATE_INVOICE, case_id=case.case_id,
                                decision=decision)

        po = case.po_lookup.purchase_order  # APPROVE implies an exact match -> FOUND PO
        committed_at = datetime.now(timezone.utc).isoformat()

        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cur = self._conn.execute(
                """
                INSERT INTO actions
                    (case_id, action, invoice_id, po_id, supplier_id, amount_cents,
                     input_fingerprint, policy_version, committed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.case_id, "APPROVE", case.invoice.invoice_id, po.po_id,
                    case.invoice.supplier_id, amount, fingerprint,
                    case.policy_version, committed_at,
                ),
            )
            action_id = cur.lastrowid
            registry_insert(
                self._conn, case.case_id, case.invoice.supplier_id, case.invoice.invoice_id
            )
            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            # A concurrent writer (B1 or B2) committed first; reconcile without
            # overwriting, and never leave a partial action.
            self._conn.execute("ROLLBACK")
            existing = self._fetch(case.case_id)
            if existing is not None:
                if existing.input_fingerprint == fingerprint:
                    return CommitResult(status=CommitStatus.IDEMPOTENT, case_id=case.case_id,
                                        decision=decision, receipt=existing)
                return CommitResult(status=CommitStatus.CONFLICT, case_id=case.case_id,
                                    decision=decision)
            if registry_case_conflict(self._conn, case.case_id):
                return CommitResult(status=CommitStatus.CONFLICT, case_id=case.case_id,
                                    decision=decision)
            if registry_invoice_conflict(self._conn, case.invoice.supplier_id, case.invoice.invoice_id):
                return CommitResult(status=CommitStatus.DUPLICATE_INVOICE, case_id=case.case_id,
                                    decision=decision)
            if self._invoice_exists(case.invoice.supplier_id, case.invoice.invoice_id, case.case_id):
                return CommitResult(status=CommitStatus.DUPLICATE_INVOICE, case_id=case.case_id,
                                    decision=decision)
            raise

        receipt = ActionReceipt(
            action_id=action_id,
            case_id=case.case_id,
            action="APPROVE",
            invoice_id=case.invoice.invoice_id,
            po_id=po.po_id,
            supplier_id=case.invoice.supplier_id,
            amount_cents=amount,
            input_fingerprint=fingerprint,
            policy_version=case.policy_version,
            committed_at=committed_at,
        )
        return CommitResult(status=CommitStatus.COMMITTED, case_id=case.case_id,
                            decision=decision, receipt=receipt)

    def _invoice_exists(self, supplier_id: str, invoice_id: str, case_id: str) -> bool:
        """True if this invoice identity is already approved under a *different* case.

        Excluding the caller's own case id enforces result precedence: a reused
        case id with the same invoice is handled as idempotent/conflict (via the
        case-id lookup and its UNIQUE constraint), never as a duplicate. Only a
        genuinely different case id with a previously approved invoice identity
        yields ``DUPLICATE_INVOICE``.
        """
        row = self._conn.execute(
            "SELECT 1 FROM actions WHERE supplier_id = ? AND invoice_id = ? AND case_id != ?",
            (supplier_id, invoice_id, case_id),
        ).fetchone()
        return row is not None

    def _fetch(self, case_id: str) -> Optional[ActionReceipt]:
        row = self._conn.execute(
            "SELECT id, case_id, action, invoice_id, po_id, supplier_id, amount_cents, "
            "input_fingerprint, policy_version, committed_at FROM actions WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        return ActionReceipt(
            action_id=row[0], case_id=row[1], action=row[2], invoice_id=row[3],
            po_id=row[4], supplier_id=row[5], amount_cents=row[6],
            input_fingerprint=row[7], policy_version=row[8], committed_at=row[9],
        )
