"""Durable human-review requests for BOUNDARY (B2.1 foundation).

A review request is a durable, non-executing record that a simulated invoice
exception requires human authority before any simulated action may execute. It
never grants authority, never authenticates evidence, and never asserts that an
exception is commercially valid or executable.

The ``$500`` authority boundary applies to the *positive discrepancy* (invoice
total minus PO total), not the invoice total. A supported positive discrepancy
exceeding ``$500`` requires human review.

Structural validity is enforced: a case with an unresolved PO, a supplier
mismatch, a missing/mismatched PO association, inconsistent arithmetic, or
duplicate line identifiers is ``NOT_REVIEWABLE`` (never ``NOT_REQUIRED``, which
is reserved for valid cases that simply do not exceed the boundary).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CaseInput, SupplierMatchState
from .ledger import case_fingerprint

# The $500 authority boundary, in integer USD cents. Applies to the discrepancy.
AUTHORITY_BOUNDARY_CENTS = 50_000  # $500.00

# SQLite signed 64-bit integer range (a storage constraint, not business policy).
_SQLITE_MIN_INT = -(2 ** 63)
_SQLITE_MAX_INT = (2 ** 63) - 1


class ReviewDecision(str, Enum):
    REQUIRES_AUTHORITY = "REQUIRES_AUTHORITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_REQUIRED = "NOT_REQUIRED"
    NOT_REVIEWABLE = "NOT_REVIEWABLE"


class RequestStatus(str, Enum):
    CREATED = "CREATED"
    IDEMPOTENT = "IDEMPOTENT"
    CONFLICT = "CONFLICT"
    NOT_REQUIRED = "NOT_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_REVIEWABLE = "NOT_REVIEWABLE"
    STORAGE_OVERFLOW = "STORAGE_OVERFLOW"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: int
    case_id: str
    case_fingerprint: str
    invoice_id: str
    po_id: str
    supplier_id: str
    proposed_action: str
    amount_cents: int
    discrepancy_cents: int
    policy_version: str
    state: str
    reason: str
    created_at: str


class RequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RequestStatus
    case_id: str
    decision: Optional[ReviewDecision] = None
    reasons: list[str] = Field(default_factory=list)
    request: Optional[ReviewRequest] = None


def _duplicate_line_numbers(items) -> bool:
    seen: set[int] = set()
    for item in items:
        if item.line_number in seen:
            return True
        seen.add(item.line_number)
    return False


def _structural_reasons(case: CaseInput) -> list[str]:
    """Reasons the case is structurally invalid/unresolved for review (empty if valid)."""
    reasons: list[str] = []

    if case.po_lookup.state == "UNAVAILABLE":
        reasons.append("po_unavailable")
    elif case.po_lookup.state == "NOT_FOUND":
        reasons.append("po_not_found")

    if case.supplier_lookup.state == SupplierMatchState.UNRESOLVED:
        reasons.append("supplier_unresolved")
    elif case.supplier_lookup.state == SupplierMatchState.MISMATCH:
        reasons.append("supplier_mismatch")

    if case.po_lookup.state == "FOUND":
        po = case.po_lookup.purchase_order
        if case.invoice.supplier_id != po.supplier_id:
            reasons.append("supplier_id_inconsistent")
        if case.invoice.po_reference is None:
            reasons.append("po_association_missing")
        elif case.invoice.po_reference != po.po_id:
            reasons.append("po_association_mismatch")
        if case.invoice.stated_total_cents != case.invoice.computed_total_cents:
            reasons.append("invoice_arithmetic_mismatch")
        if po.stated_total_cents != po.computed_total_cents:
            reasons.append("po_arithmetic_mismatch")
        if _duplicate_line_numbers(case.invoice.line_items) or _duplicate_line_numbers(po.line_items):
            reasons.append("duplicate_line_identifiers")

    return reasons


def _has_meaningful_evidence(case: CaseInput) -> bool:
    """True if the case carries at least one non-blank evidence identifier.

    Blank/whitespace-only identifiers are not meaningful evidence for review
    intake. Presence is never authentication and never proof of business support.
    Nonblank identifiers are left untouched (never stripped or interpreted).
    """
    return any(e.strip() for e in case.evidence)


def review_decision(case: CaseInput) -> ReviewDecision:
    """Classify whether a case requires human review.

    Structurally invalid or unresolved cases are ``NOT_REVIEWABLE``. A
    structurally valid case with a positive discrepancy at or below ``$500`` is
    ``NOT_REQUIRED``. A valid discrepancy above ``$500`` requires evidence
    (``REQUIRES_AUTHORITY``), otherwise ``INSUFFICIENT_EVIDENCE``.
    """
    if _structural_reasons(case):
        return ReviewDecision.NOT_REVIEWABLE
    po = case.po_lookup.purchase_order
    d = case.invoice.computed_total_cents - po.computed_total_cents
    if d <= AUTHORITY_BOUNDARY_CENTS:
        return ReviewDecision.NOT_REQUIRED
    if not _has_meaningful_evidence(case):
        return ReviewDecision.INSUFFICIENT_EVIDENCE
    return ReviewDecision.REQUIRES_AUTHORITY


class ReviewStore:
    """SQLite store of durable human-review requests."""

    def __init__(self, path: str, conn=None):
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(path, isolation_level=None)
            self._owns_conn = True
        # Stable, defensible store identity so grants can bind to a specific
        # review store rather than an ambiguous numeric request id. INSERT OR
        # IGNORE makes first-time initialization concurrency-safe.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO store_meta (key, value) VALUES ('store_id', ?)",
            (uuid.uuid4().hex,),
        )
        row = self._conn.execute(
            "SELECT value FROM store_meta WHERE key = 'store_id'"
        ).fetchone()
        self._store_id = row[0]
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL UNIQUE,
                case_fingerprint TEXT NOT NULL,
                invoice_id TEXT NOT NULL,
                po_id TEXT NOT NULL,
                supplier_id TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                discrepancy_cents INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                state TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Cross-case invoice-identity dedup: the same invoice may only be
        # reviewed once, even under different case ids.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_review_invoice_identity "
            "ON review_requests(supplier_id, invoice_id)"
        )

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    @property
    def store_id(self) -> str:
        return self._store_id

    def get(self, case_id: str) -> Optional[ReviewRequest]:
        return self._fetch(case_id)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM review_requests").fetchone()[0]

    def _evaluate(self, case: CaseInput) -> ReviewDecision:
        """Test seam: evaluate eligibility (overridable to force interleaving)."""
        return review_decision(case)

    def _invoice_reviewed(self, supplier_id: str, invoice_id: str, case_id: str) -> bool:
        """True if this invoice identity is already reviewed under a different case."""
        row = self._conn.execute(
            "SELECT 1 FROM review_requests WHERE supplier_id = ? AND invoice_id = ? AND case_id != ?",
            (supplier_id, invoice_id, case_id),
        ).fetchone()
        return row is not None

    def request(self, case: CaseInput) -> RequestResult:
        """Create (or reconcile) a durable review request for a case."""
        fingerprint = case_fingerprint(case)

        # 1. Existing case-id precedence (checked before eligibility): a reused
        #    case id must never silently reuse or re-evaluate a prior request.
        existing = self._fetch(case.case_id)
        if existing is not None:
            if existing.case_fingerprint == fingerprint:
                return RequestResult(status=RequestStatus.IDEMPOTENT, case_id=case.case_id,
                                     request=existing)
            return RequestResult(status=RequestStatus.CONFLICT, case_id=case.case_id)

        # 2. Eligibility.
        decision = self._evaluate(case)
        if decision != ReviewDecision.REQUIRES_AUTHORITY:
            # Re-check: a concurrent writer may have committed this case id after
            # the initial lookup but before eligibility, which takes precedence.
            existing = self._fetch(case.case_id)
            if existing is not None:
                if existing.case_fingerprint == fingerprint:
                    return RequestResult(status=RequestStatus.IDEMPOTENT, case_id=case.case_id,
                                         request=existing)
                return RequestResult(status=RequestStatus.CONFLICT, case_id=case.case_id)
            if decision == ReviewDecision.NOT_REVIEWABLE:
                return RequestResult(status=RequestStatus.NOT_REVIEWABLE, case_id=case.case_id,
                                     decision=decision, reasons=_structural_reasons(case))
            if decision == ReviewDecision.NOT_REQUIRED:
                return RequestResult(status=RequestStatus.NOT_REQUIRED, case_id=case.case_id,
                                     decision=decision)
            return RequestResult(status=RequestStatus.INSUFFICIENT_EVIDENCE, case_id=case.case_id,
                                 decision=decision, reasons=["no evidence provided"])

        # 3. Storage range (both amount and discrepancy must fit signed 64-bit).
        po = case.po_lookup.purchase_order
        amount = case.invoice.computed_total_cents
        discrepancy = amount - po.computed_total_cents
        if not (_SQLITE_MIN_INT <= amount <= _SQLITE_MAX_INT
                and _SQLITE_MIN_INT <= discrepancy <= _SQLITE_MAX_INT):
            return RequestResult(status=RequestStatus.STORAGE_OVERFLOW, case_id=case.case_id,
                                 decision=decision,
                                 reasons=["amount or discrepancy exceeds SQLite integer range"])

        # 3b. Cross-case duplicate invoice identity.
        if self._invoice_reviewed(case.invoice.supplier_id, case.invoice.invoice_id, case.case_id):
            return RequestResult(status=RequestStatus.DUPLICATE_INVOICE, case_id=case.case_id,
                                 decision=decision,
                                 reasons=["invoice already reviewed under another case"])

        # 4. INSERT with concurrency reconciliation.
        created_at = datetime.now(timezone.utc).isoformat()
        reason = (
            f"positive discrepancy {discrepancy} cents exceeds the "
            f"${AUTHORITY_BOUNDARY_CENTS // 100} authority boundary; human review required"
        )

        try:
            cur = self._conn.execute(
                """
                INSERT INTO review_requests
                    (case_id, case_fingerprint, invoice_id, po_id, supplier_id,
                     proposed_action, amount_cents, discrepancy_cents, policy_version,
                     state, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.case_id, fingerprint, case.invoice.invoice_id, po.po_id,
                    case.invoice.supplier_id, "APPROVE", amount,
                    discrepancy, case.policy_version, "WAITING_AUTHORITY", reason, created_at,
                ),
            )
            request_id = cur.lastrowid
        except sqlite3.IntegrityError:
            # A concurrent writer committed first; reconcile without overwriting.
            existing = self._fetch(case.case_id)
            if existing is not None:
                if existing.case_fingerprint == fingerprint:
                    return RequestResult(status=RequestStatus.IDEMPOTENT, case_id=case.case_id,
                                         request=existing)
                return RequestResult(status=RequestStatus.CONFLICT, case_id=case.case_id)
            if self._invoice_reviewed(case.invoice.supplier_id, case.invoice.invoice_id, case.case_id):
                return RequestResult(status=RequestStatus.DUPLICATE_INVOICE, case_id=case.case_id,
                                     decision=decision,
                                     reasons=["invoice already reviewed under another case"])
            raise

        request = ReviewRequest(
            request_id=request_id, case_id=case.case_id, case_fingerprint=fingerprint,
            invoice_id=case.invoice.invoice_id, po_id=po.po_id,
            supplier_id=case.invoice.supplier_id, proposed_action="APPROVE",
            amount_cents=amount, discrepancy_cents=discrepancy,
            policy_version=case.policy_version, state="WAITING_AUTHORITY", reason=reason,
            created_at=created_at,
        )
        return RequestResult(status=RequestStatus.CREATED, case_id=case.case_id,
                             decision=decision, request=request)

    def _fetch(self, case_id: str) -> Optional[ReviewRequest]:
        row = self._conn.execute(
            "SELECT id, case_id, case_fingerprint, invoice_id, po_id, supplier_id, "
            "proposed_action, amount_cents, discrepancy_cents, policy_version, state, "
            "reason, created_at FROM review_requests WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        return ReviewRequest(
            request_id=row[0], case_id=row[1], case_fingerprint=row[2], invoice_id=row[3],
            po_id=row[4], supplier_id=row[5], proposed_action=row[6], amount_cents=row[7],
            discrepancy_cents=row[8], policy_version=row[9], state=row[10], reason=row[11],
            created_at=row[12],
        )
