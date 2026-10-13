"""Scoped human authorization grants for BOUNDARY (B2.2a).

A grant is a durable, action-specific record that a reviewer has authorized a
*specific* pending review request for a *specific* case and action. It is
distinct from a ``WAITING_AUTHORITY`` review request and from actual execution:
a grant authorizes nothing by itself, and nothing is consumed or executed in
B2.2a.

Issuance integrity:
- Simulated issuance requires an explicit trusted test-mode opt-in on the grant
  store, not merely a caller-controlled reviewer-source value.
- A reviewer name is never treated as proof of human identity.
- The grant scope is derived independently from the source case and the stored
  review is validated against it; corrupted or inconsistent reviews fail closed.
- A ``WAITING_AUTHORITY`` row is not proof that valid review intake occurred:
  eligibility is revalidated from the case before issuance.
- Grants bind to the review store's identity (not an ambiguous numeric id).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CaseInput
from .ledger import case_fingerprint
from .review import ReviewDecision, ReviewRequest, ReviewStore, review_decision

# Default grant lifetime (24 hours). Expiry is recorded as metadata here; it is
# not enforced because consumption/execution is out of scope for B2.2a.
GRANT_TTL_SECONDS = 24 * 3600

# The only permitted proposed action in this scope.
_PERMITTED_ACTION = "APPROVE"


class GrantSchemaError(Exception):
    """A legacy grant database with existing records cannot be safely migrated."""


class ReviewerSource(str, Enum):
    # Explicit simulated test context; not a claim of real human identity.
    SIMULATED = "SIMULATED"
    # A bare name with no verification source; must be rejected.
    CLAIMED_HUMAN = "CLAIMED_HUMAN"


class Reviewer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: ReviewerSource


class GrantState(str, Enum):
    ISSUED = "ISSUED"
    CLAIMED = "CLAIMED"


class Grant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_id: str
    review_store_id: str
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
    reviewer: str
    reviewer_source: str
    issued_at: str
    expires_at: str
    state: str


class GrantStatus(str, Enum):
    ISSUED = "ISSUED"
    ALREADY_ISSUED = "ALREADY_ISSUED"
    NO_REVIEW_REQUEST = "NO_REVIEW_REQUEST"
    REVIEW_NOT_PENDING = "REVIEW_NOT_PENDING"
    CASE_CHANGED = "CASE_CHANGED"
    REVIEW_INCONSISTENT = "REVIEW_INCONSISTENT"
    CASE_NOT_ELIGIBLE = "CASE_NOT_ELIGIBLE"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    REVIEWER_REJECTED = "REVIEWER_REJECTED"
    SIMULATION_NOT_ENABLED = "SIMULATION_NOT_ENABLED"


class GrantResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GrantStatus
    grant: Optional[Grant] = None
    reasons: list[str] = Field(default_factory=list)


class ClaimStatus(str, Enum):
    CLAIMED = "CLAIMED"
    ALREADY_CLAIMED = "ALREADY_CLAIMED"
    GRANT_NOT_FOUND = "GRANT_NOT_FOUND"
    GRANT_EXPIRED = "GRANT_EXPIRED"
    GRANT_INVALID_METADATA = "GRANT_INVALID_METADATA"
    GRANT_STORE_MISMATCH = "GRANT_STORE_MISMATCH"
    CASE_CHANGED = "CASE_CHANGED"
    GRANT_INCONSISTENT = "GRANT_INCONSISTENT"
    REVIEW_INCONSISTENT = "REVIEW_INCONSISTENT"
    REVIEW_NOT_PENDING = "REVIEW_NOT_PENDING"
    REVIEWER_MISMATCH = "REVIEWER_MISMATCH"
    CASE_NOT_ELIGIBLE = "CASE_NOT_ELIGIBLE"
    NO_REVIEW_REQUEST = "NO_REVIEW_REQUEST"
    REVIEWER_REJECTED = "REVIEWER_REJECTED"
    SIMULATION_NOT_ENABLED = "SIMULATION_NOT_ENABLED"


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_id: str
    claimed_at: str
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


class ClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ClaimStatus
    claim: Optional[ClaimRecord] = None
    reasons: list[str] = Field(default_factory=list)


def _derive_scope(case: CaseInput) -> dict:
    """Independently derive the grant scope from the source case.

    The only permitted action is ``APPROVE``. Values come from the case, never
    from a stored review row.
    """
    po = case.po_lookup.purchase_order
    amount = case.invoice.computed_total_cents
    discrepancy = amount - po.computed_total_cents
    return {
        "case_id": case.case_id,
        "fingerprint": case_fingerprint(case),
        "supplier_id": case.invoice.supplier_id,
        "invoice_id": case.invoice.invoice_id,
        "po_id": po.po_id,
        "proposed_action": _PERMITTED_ACTION,
        "amount_cents": amount,
        "discrepancy_cents": discrepancy,
        "policy_version": case.policy_version,
    }


def _review_consistent(request: ReviewRequest, scope: dict) -> bool:
    """True if the stored review's security-relevant fields match the source case.

    Fingerprint is checked separately (it maps to the ``CASE_CHANGED`` outcome).
    """
    return (
        request.case_id == scope["case_id"]
        and request.supplier_id == scope["supplier_id"]
        and request.invoice_id == scope["invoice_id"]
        and request.po_id == scope["po_id"]
        and request.proposed_action == scope["proposed_action"]
        and request.amount_cents == scope["amount_cents"]
        and request.discrepancy_cents == scope["discrepancy_cents"]
        and request.policy_version == scope["policy_version"]
    )


def _grant_matches_scope(grant: Grant, scope: dict) -> bool:
    """True if the grant's security-relevant fields match the derived case scope.

    Fingerprint is checked separately (it maps to the ``CASE_CHANGED`` outcome).
    """
    return (
        grant.case_id == scope["case_id"]
        and grant.supplier_id == scope["supplier_id"]
        and grant.invoice_id == scope["invoice_id"]
        and grant.po_id == scope["po_id"]
        and grant.proposed_action == scope["proposed_action"]
        and grant.amount_cents == scope["amount_cents"]
        and grant.discrepancy_cents == scope["discrepancy_cents"]
        and grant.policy_version == scope["policy_version"]
    )


def _utcnow() -> datetime:
    """Default clock: current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def _validate_times(issued_at: str, expires_at: str, now: datetime) -> tuple[bool, str]:
    """Validate grant authorization metadata against the clock.

    Returns ``(ok, reason)``. Fails closed on malformed, naive, future-issued,
    invalid-window, or expired metadata.
    """
    try:
        issued = datetime.fromisoformat(issued_at)
        expires = datetime.fromisoformat(expires_at)
    except (ValueError, TypeError):
        return False, "malformed_timestamp"
    if issued.tzinfo is None or expires.tzinfo is None:
        return False, "naive_timestamp"
    if issued > now:
        return False, "future_issued"
    if expires - issued != timedelta(seconds=GRANT_TTL_SECONDS):
        return False, "invalid_window"
    if now >= expires:
        return False, "expired"
    return True, "ok"


class GrantStore:
    """SQLite store of durable, action-specific authorization grants.

    A grant never authorizes anything outside its exact case/action scope, and
    carries no execution field: it records *authorization*, not *execution*.
    """

    def __init__(self, path: str, test_mode: bool = False, clock=None, conn=None):
        self._test_mode = test_mode
        self._clock = clock if clock is not None else _utcnow
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(path, isolation_level=None)
            self._owns_conn = True
        self._init_schema()
        self._create_claims_table()

    def _init_schema(self) -> None:
        """Create the current schema, or migrate an empty legacy schema safely.

        A legacy B2.2a grants table (no ``review_store_id``) is migrated only if
        empty; a populated legacy table fails closed with ``GrantSchemaError`` so
        durable records are never discarded or given an invented identity.
        """
        columns = self._existing_columns()
        if columns is None:
            self._create_new_schema()
            return
        if "review_store_id" in columns:
            self._ensure_indexes()
            return
        # Legacy schema without review_store_id.
        count = self._conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0]
        if count == 0:
            self._conn.execute("DROP TABLE grants")
            self._create_new_schema()
        else:
            raise GrantSchemaError(
                "legacy grant schema detected with existing records; cannot migrate safely"
            )

    def _existing_columns(self):
        """Column names of the grants table, or None if it does not exist."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'grants'"
        ).fetchone()
        if row is None:
            return None
        return {r[1] for r in self._conn.execute("PRAGMA table_info(grants)").fetchall()}

    def _create_new_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grant_id TEXT NOT NULL UNIQUE,
                review_store_id TEXT NOT NULL,
                request_id INTEGER NOT NULL,
                case_id TEXT NOT NULL,
                case_fingerprint TEXT NOT NULL,
                invoice_id TEXT NOT NULL,
                po_id TEXT NOT NULL,
                supplier_id TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                discrepancy_cents INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                reviewer_source TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                state TEXT NOT NULL
            )
            """
        )
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # One grant per (review store, request); invoice dedup across cases.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_grant_request "
            "ON grants(review_store_id, request_id)"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_grant_invoice_identity "
            "ON grants(supplier_id, invoice_id)"
        )

    def _create_claims_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grant_id TEXT NOT NULL UNIQUE,
                claimed_at TEXT NOT NULL,
                case_id TEXT NOT NULL,
                case_fingerprint TEXT NOT NULL,
                invoice_id TEXT NOT NULL,
                po_id TEXT NOT NULL,
                supplier_id TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                discrepancy_cents INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                state TEXT NOT NULL
            )
            """
        )

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0]

    def get(self, grant_id: str) -> Optional[Grant]:
        row = self._conn.execute(
            "SELECT grant_id, review_store_id, request_id, case_id, case_fingerprint, "
            "invoice_id, po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, "
            "policy_version, reviewer, reviewer_source, issued_at, expires_at, state "
            "FROM grants WHERE grant_id = ?",
            (grant_id,),
        ).fetchone()
        return self._row_to_grant(row)

    def get_by_request(self, review_store_id: str, request_id: int) -> Optional[Grant]:
        row = self._conn.execute(
            "SELECT grant_id, review_store_id, request_id, case_id, case_fingerprint, "
            "invoice_id, po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, "
            "policy_version, reviewer, reviewer_source, issued_at, expires_at, state "
            "FROM grants WHERE review_store_id = ? AND request_id = ?",
            (review_store_id, request_id),
        ).fetchone()
        return self._row_to_grant(row)

    def _row_to_grant(self, row) -> Optional[Grant]:
        if row is None:
            return None
        return Grant(
            grant_id=row[0], review_store_id=row[1], request_id=row[2], case_id=row[3],
            case_fingerprint=row[4], invoice_id=row[5], po_id=row[6], supplier_id=row[7],
            proposed_action=row[8], amount_cents=row[9], discrepancy_cents=row[10],
            policy_version=row[11], reviewer=row[12], reviewer_source=row[13],
            issued_at=row[14], expires_at=row[15], state=row[16],
        )

    def issue(
        self, review_store: ReviewStore, case: CaseInput, reviewer: Reviewer
    ) -> GrantResult:
        """Issue an action-specific grant for a pending review request.

        Fails closed on: missing test-mode opt-in, unverified/blank reviewer,
        ineligible case, missing/non-pending request, changed case content,
        inconsistent stored review, duplicate invoice, or double issuance.
        Never consumes or executes the grant.
        """
        # 1. Simulated issuance requires an explicit trusted test-mode opt-in.
        if not self._test_mode:
            return GrantResult(
                status=GrantStatus.SIMULATION_NOT_ENABLED,
                reasons=["simulated issuance requires an explicit test-mode opt-in"],
            )

        # 2. Reviewer source must be explicitly simulated, with a non-blank name.
        if reviewer.source != ReviewerSource.SIMULATED:
            return GrantResult(
                status=GrantStatus.REVIEWER_REJECTED,
                reasons=["reviewer identity is not verified; only simulated issuance is available"],
            )
        if not reviewer.name.strip():
            return GrantResult(
                status=GrantStatus.REVIEWER_REJECTED,
                reasons=["reviewer name is blank"],
            )

        # 3. Independently revalidate eligibility: a WAITING_AUTHORITY row is not
        #    proof that valid review intake occurred.
        if review_decision(case) != ReviewDecision.REQUIRES_AUTHORITY:
            return GrantResult(
                status=GrantStatus.CASE_NOT_ELIGIBLE,
                reasons=["underlying review eligibility is not satisfied"],
            )

        # 4. The review request must exist and be pending.
        request = review_store.get(case.case_id)
        if request is None:
            return GrantResult(
                status=GrantStatus.NO_REVIEW_REQUEST,
                reasons=["no review request for this case"],
            )
        if request.state != "WAITING_AUTHORITY":
            return GrantResult(
                status=GrantStatus.REVIEW_NOT_PENDING,
                reasons=["review request is not pending"],
            )

        # 5. Independently derive the grant scope from the source case.
        scope = _derive_scope(case)

        # 6. Validate the stored review against the derived facts (fail closed).
        if request.case_fingerprint != scope["fingerprint"]:
            return GrantResult(
                status=GrantStatus.CASE_CHANGED,
                reasons=["case content changed since review request"],
            )
        if not _review_consistent(request, scope):
            return GrantResult(
                status=GrantStatus.REVIEW_INCONSISTENT,
                reasons=["stored review is inconsistent with the source case"],
            )

        # 7. One grant per review request, bound to the review-store identity.
        store_id = review_store.store_id
        if self.get_by_request(store_id, request.request_id) is not None:
            return GrantResult(
                status=GrantStatus.ALREADY_ISSUED,
                reasons=["a grant already exists for this review request"],
            )

        # 8. Cross-case invoice dedup.
        if self._invoice_granted(scope["supplier_id"], scope["invoice_id"]):
            return GrantResult(
                status=GrantStatus.DUPLICATE_INVOICE,
                reasons=["invoice already granted under another case"],
            )

        # 9. Issue from the derived scope (never unverified stored values).
        now = self._clock()
        issued_at = now.isoformat()
        expires_at = (now + timedelta(seconds=GRANT_TTL_SECONDS)).isoformat()

        grant = Grant(
            grant_id=uuid.uuid4().hex,
            review_store_id=store_id,
            request_id=request.request_id,
            case_id=scope["case_id"],
            case_fingerprint=scope["fingerprint"],
            invoice_id=scope["invoice_id"],
            po_id=scope["po_id"],
            supplier_id=scope["supplier_id"],
            proposed_action=scope["proposed_action"],
            amount_cents=scope["amount_cents"],
            discrepancy_cents=scope["discrepancy_cents"],
            policy_version=scope["policy_version"],
            reviewer=reviewer.name,
            reviewer_source=reviewer.source.value,
            issued_at=issued_at,
            expires_at=expires_at,
            state=GrantState.ISSUED.value,
        )

        try:
            self._conn.execute(
                """
                INSERT INTO grants
                    (grant_id, review_store_id, request_id, case_id, case_fingerprint,
                     invoice_id, po_id, supplier_id, proposed_action, amount_cents,
                     discrepancy_cents, policy_version, reviewer, reviewer_source,
                     issued_at, expires_at, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grant.grant_id, grant.review_store_id, grant.request_id, grant.case_id,
                    grant.case_fingerprint, grant.invoice_id, grant.po_id, grant.supplier_id,
                    grant.proposed_action, grant.amount_cents, grant.discrepancy_cents,
                    grant.policy_version, grant.reviewer, grant.reviewer_source,
                    grant.issued_at, grant.expires_at, grant.state,
                ),
            )
        except sqlite3.IntegrityError:
            # A concurrent writer issued first; reconcile without overwriting.
            existing = self.get_by_request(store_id, request.request_id)
            if existing is not None:
                return GrantResult(
                    status=GrantStatus.ALREADY_ISSUED, grant=existing,
                    reasons=["a grant already exists for this review request"],
                )
            if self._invoice_granted(scope["supplier_id"], scope["invoice_id"]):
                return GrantResult(
                    status=GrantStatus.DUPLICATE_INVOICE,
                    reasons=["invoice already granted under another case"],
                )
            raise

        return GrantResult(status=GrantStatus.ISSUED, grant=grant)

    def _invoice_granted(self, supplier_id: str, invoice_id: str) -> bool:
        """True if any grant exists for this invoice identity, across all cases/stores."""
        row = self._conn.execute(
            "SELECT 1 FROM grants WHERE supplier_id = ? AND invoice_id = ?",
            (supplier_id, invoice_id),
        ).fetchone()
        return row is not None

    def get_claim(self, grant_id: str) -> Optional[ClaimRecord]:
        row = self._conn.execute(
            "SELECT grant_id, claimed_at, case_id, case_fingerprint, invoice_id, po_id, "
            "supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, state "
            "FROM claims WHERE grant_id = ?",
            (grant_id,),
        ).fetchone()
        if row is None:
            return None
        return ClaimRecord(
            grant_id=row[0], claimed_at=row[1], case_id=row[2], case_fingerprint=row[3],
            invoice_id=row[4], po_id=row[5], supplier_id=row[6], proposed_action=row[7],
            amount_cents=row[8], discrepancy_cents=row[9], policy_version=row[10], state=row[11],
        )

    def _revalidate_binding(self, grant, request, reviewer, review_store, scope, now):
        """Return a ClaimResult on any binding/scope/expiration failure, else None.

        Validates the complete authorization binding of a freshly loaded grant
        and review: state, review-store identity, request identity, simulated
        provenance, reviewer identity consistency, review validity, business
        scope, and expiration.
        """
        if grant.state != GrantState.ISSUED.value:
            return ClaimResult(status=ClaimStatus.ALREADY_CLAIMED,
                               reasons=["grant is not in ISSUED state"])
        if grant.review_store_id != review_store.store_id:
            return ClaimResult(status=ClaimStatus.GRANT_STORE_MISMATCH,
                               reasons=["grant was issued by a different review store"])
        if grant.request_id != request.request_id:
            return ClaimResult(status=ClaimStatus.GRANT_STORE_MISMATCH,
                               reasons=["grant is not bound to this review request"])
        if grant.reviewer_source != ReviewerSource.SIMULATED.value:
            return ClaimResult(status=ClaimStatus.REVIEWER_REJECTED,
                               reasons=["grant reviewer provenance is not simulated"])
        if grant.reviewer != reviewer.name:
            return ClaimResult(status=ClaimStatus.REVIEWER_MISMATCH,
                               reasons=["claiming reviewer does not match the issuing reviewer"])
        if request.state != "WAITING_AUTHORITY":
            return ClaimResult(status=ClaimStatus.REVIEW_NOT_PENDING,
                               reasons=["review request is no longer pending"])
        if request.case_fingerprint != scope["fingerprint"]:
            return ClaimResult(status=ClaimStatus.CASE_CHANGED,
                               reasons=["case content changed since review request"])
        if not _review_consistent(request, scope):
            return ClaimResult(status=ClaimStatus.REVIEW_INCONSISTENT,
                               reasons=["stored review is inconsistent with the source case"])
        if grant.case_fingerprint != scope["fingerprint"]:
            return ClaimResult(status=ClaimStatus.CASE_CHANGED,
                               reasons=["case content changed since grant issuance"])
        if not _grant_matches_scope(grant, scope):
            return ClaimResult(status=ClaimStatus.GRANT_INCONSISTENT,
                               reasons=["grant is inconsistent with the source case"])
        ok, why = _validate_times(grant.issued_at, grant.expires_at, now)
        if not ok:
            if why == "expired":
                return ClaimResult(status=ClaimStatus.GRANT_EXPIRED,
                                   reasons=["grant has expired"])
            return ClaimResult(status=ClaimStatus.GRANT_INVALID_METADATA, reasons=[why])
        return None

    def claim(
        self, grant_id: str, review_store: ReviewStore, case: CaseInput, reviewer: Reviewer
    ) -> ClaimResult:
        """Atomically claim a grant exactly once, enforcing expiration and scope.

        Fails closed on any invalid metadata, scope mismatch, binding mismatch,
        or replay. Never executes the granted action: a claim is distinct from
        execution.

        Cross-database limitation: the review lives in the review store and the
        grant/claim in this store. The review is re-read and re-validated under
        this store's write lock, but the review re-read and the claim commit
        cannot be a single atomic transaction across two databases. A review
        change in the microscopic window between that re-read and the commit is
        not caught here; the later claim-to-execution integration must treat a
        claimed grant as still subject to a final review check before any action.
        """
        # 1. Simulation mode.
        if not self._test_mode:
            return ClaimResult(
                status=ClaimStatus.SIMULATION_NOT_ENABLED,
                reasons=["claiming requires an explicit test-mode opt-in"],
            )

        # 2. Reviewer provenance (a SIMULATED reviewer is not authenticated).
        if reviewer.source != ReviewerSource.SIMULATED:
            return ClaimResult(
                status=ClaimStatus.REVIEWER_REJECTED,
                reasons=["reviewer identity is not verified; only simulated claiming is available"],
            )
        if not reviewer.name.strip():
            return ClaimResult(
                status=ClaimStatus.REVIEWER_REJECTED,
                reasons=["reviewer name is blank"],
            )

        # 3. Load the grant.
        grant = self.get(grant_id)
        if grant is None:
            return ClaimResult(
                status=ClaimStatus.GRANT_NOT_FOUND,
                reasons=["no such grant"],
            )
        if grant.state != GrantState.ISSUED.value:
            return ClaimResult(
                status=ClaimStatus.ALREADY_CLAIMED,
                reasons=[f"grant is in state {grant.state}, not ISSUED"],
            )

        # 4. Independently revalidate eligibility.
        if review_decision(case) != ReviewDecision.REQUIRES_AUTHORITY:
            return ClaimResult(
                status=ClaimStatus.CASE_NOT_ELIGIBLE,
                reasons=["underlying review eligibility is not satisfied"],
            )

        # 5. Revalidate review-store and review-request binding.
        request = review_store.get(case.case_id)
        if request is None:
            return ClaimResult(
                status=ClaimStatus.NO_REVIEW_REQUEST,
                reasons=["no review request for this case"],
            )
        if grant.review_store_id != review_store.store_id:
            return ClaimResult(
                status=ClaimStatus.GRANT_STORE_MISMATCH,
                reasons=["grant was issued by a different review store"],
            )
        if grant.request_id != request.request_id:
            return ClaimResult(
                status=ClaimStatus.GRANT_STORE_MISMATCH,
                reasons=["grant is not bound to this review request"],
            )

        # 5b. Revalidate the stored review: it must remain pending and consistent
        #     with the source case (fail closed on any post-issuance change).
        scope = _derive_scope(case)
        if request.state != "WAITING_AUTHORITY":
            return ClaimResult(
                status=ClaimStatus.REVIEW_NOT_PENDING,
                reasons=["review request is no longer pending"],
            )
        if request.case_fingerprint != scope["fingerprint"]:
            return ClaimResult(
                status=ClaimStatus.CASE_CHANGED,
                reasons=["case content changed since review request"],
            )
        if not _review_consistent(request, scope):
            return ClaimResult(
                status=ClaimStatus.REVIEW_INCONSISTENT,
                reasons=["stored review is inconsistent with the source case"],
            )

        # 6. Revalidate the grant scope from the source case (fail closed).
        if grant.case_fingerprint != scope["fingerprint"]:
            return ClaimResult(
                status=ClaimStatus.CASE_CHANGED,
                reasons=["case content changed since grant issuance"],
            )
        if not _grant_matches_scope(grant, scope):
            return ClaimResult(
                status=ClaimStatus.GRANT_INCONSISTENT,
                reasons=["grant is inconsistent with the source case"],
            )

        # 7. Expected reviewer provenance: simulated-only, and identity-consistent
        #    with the issuing reviewer (identity consistency, not authentication).
        if grant.reviewer_source != ReviewerSource.SIMULATED.value:
            return ClaimResult(
                status=ClaimStatus.REVIEWER_REJECTED,
                reasons=["grant reviewer provenance is not simulated"],
            )
        if grant.reviewer != reviewer.name:
            return ClaimResult(
                status=ClaimStatus.REVIEWER_MISMATCH,
                reasons=["claiming reviewer does not match the issuing reviewer"],
            )

        # 8. Expiration / metadata validation against the internal clock (early
        #    fail-closed; re-checked authoritatively under the write lock).
        now = self._clock()
        ok, why = _validate_times(grant.issued_at, grant.expires_at, now)
        if not ok:
            if why == "expired":
                return ClaimResult(
                    status=ClaimStatus.GRANT_EXPIRED,
                    reasons=["grant has expired"],
                )
            return ClaimResult(
                status=ClaimStatus.GRANT_INVALID_METADATA,
                reasons=[why],
            )

        # 9. Atomic, single-use claim. Re-validate state, scope, and expiration
        #    under the write lock using a fresh clock reading, so a grant that
        #    expires (or is claimed) during lock contention cannot commit.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            fresh = self.get(grant_id)
            if fresh is None:
                self._conn.execute("ROLLBACK")
                return ClaimResult(
                    status=ClaimStatus.GRANT_NOT_FOUND,
                    reasons=["no such grant"],
                )
            fresh_request = review_store.get(case.case_id)
            if fresh_request is None:
                self._conn.execute("ROLLBACK")
                return ClaimResult(
                    status=ClaimStatus.NO_REVIEW_REQUEST,
                    reasons=["no review request for this case"],
                )
            fresh_now = self._clock()
            err = self._revalidate_binding(fresh, fresh_request, reviewer, review_store, scope, fresh_now)
            if err is not None:
                self._conn.execute("ROLLBACK")
                return err

            claimed_at = fresh_now.isoformat()
            claim = ClaimRecord(
                grant_id=grant_id,
                claimed_at=claimed_at,
                case_id=scope["case_id"],
                case_fingerprint=scope["fingerprint"],
                invoice_id=scope["invoice_id"],
                po_id=scope["po_id"],
                supplier_id=scope["supplier_id"],
                proposed_action=scope["proposed_action"],
                amount_cents=scope["amount_cents"],
                discrepancy_cents=scope["discrepancy_cents"],
                policy_version=scope["policy_version"],
                state=GrantState.CLAIMED.value,
            )

            cur = self._conn.execute(
                "UPDATE grants SET state = ? WHERE grant_id = ? AND state = ?",
                (GrantState.CLAIMED.value, grant_id, GrantState.ISSUED.value),
            )
            if cur.rowcount == 0:
                self._conn.execute("ROLLBACK")
                return ClaimResult(
                    status=ClaimStatus.ALREADY_CLAIMED,
                    reasons=["grant was already claimed"],
                )
            self._conn.execute(
                "INSERT INTO claims (grant_id, claimed_at, case_id, case_fingerprint, "
                "invoice_id, po_id, supplier_id, proposed_action, amount_cents, "
                "discrepancy_cents, policy_version, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.grant_id, claim.claimed_at, claim.case_id, claim.case_fingerprint,
                    claim.invoice_id, claim.po_id, claim.supplier_id, claim.proposed_action,
                    claim.amount_cents, claim.discrepancy_cents, claim.policy_version, claim.state,
                ),
            )
            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            self._conn.execute("ROLLBACK")
            return ClaimResult(
                status=ClaimStatus.ALREADY_CLAIMED,
                reasons=["grant was already claimed"],
            )
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        return ClaimResult(status=ClaimStatus.CLAIMED, claim=claim)
