"""Atomic simulated human-authorized exception execution (B2.2b-2a).

The review, grant/claim, and authorized-action journal share ONE SQLite database
file and ONE connection, so ``execute`` performs the authorization decision, the
single-use grant transition, the claim insert, and the action insert in a single
transaction. A grant claim alone never permits a separate, uncoordinated
accounting write.

This is a separate, explicitly human-authorized exception path for structurally
valid cases requiring authority under the positive-discrepancy policy. It does
not modify the accepted B1 exact-match ledger, guard, classifier, or verifier,
and does not make ordinary discrepancies eligible.
"""

from __future__ import annotations

import sqlite3
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CaseInput
from .grant import (
    GrantState,
    GrantStore,
    Reviewer,
    ReviewerSource,
    _derive_scope,
)
from .registry import (
    reconcile_registry,
    registry_case_conflict,
    registry_insert,
    registry_invoice_conflict,
)
from .review import ReviewDecision, ReviewStore, review_decision


class ExecutionStatus(str, Enum):
    COMMITTED = "COMMITTED"
    ALREADY_EXECUTED = "ALREADY_EXECUTED"
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
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    CONFLICT = "CONFLICT"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    SIMULATION_NOT_ENABLED = "SIMULATION_NOT_ENABLED"


class AuthorizedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    grant_id: str
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
    executed_at: str
    state: str


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ExecutionStatus
    action: Optional[AuthorizedAction] = None
    reasons: list[str] = Field(default_factory=list)


class AuthorizedExecutor:
    """Atomic simulated human-authorized exception execution.

    Owns a single SQLite connection across the review, grant/claim, and
    authorized-action tables, so the authorization decision, single-use
    transition, claim insert, and action insert happen in one transaction.
    Separate-database configurations are not supported: atomicity cannot be
    proved across databases, so the executor is single-database by construction.
    """

    def __init__(self, path: str, test_mode: bool = False, clock=None):
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._review = ReviewStore(path, conn=self._conn)
        self._grant = GrantStore(path, test_mode=test_mode, clock=clock, conn=self._conn)
        self._clock = self._grant._clock
        self._create_action_table()

    def close(self) -> None:
        self._conn.close()

    def _create_action_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS authorized_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL UNIQUE,
                grant_id TEXT NOT NULL UNIQUE,
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
                executed_at TEXT NOT NULL,
                state TEXT NOT NULL
            )
            """
        )
        # Cross-case invoice dedup: an invoice may be approved once.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_action_invoice_identity "
            "ON authorized_actions(supplier_id, invoice_id)"
        )
        # Shared cross-path registry: case-id and invoice uniqueness across B1/B2.
        # Backfill any historical approvals that predate the registry.
        reconcile_registry(self._conn)

    def _get_action_by_grant(self, grant_id: str) -> Optional[AuthorizedAction]:
        row = self._conn.execute(
            "SELECT action_id, grant_id, case_id, case_fingerprint, invoice_id, po_id, "
            "supplier_id, proposed_action, amount_cents, discrepancy_cents, policy_version, "
            "reviewer, executed_at, state FROM authorized_actions WHERE grant_id = ?",
            (grant_id,),
        ).fetchone()
        if row is None:
            return None
        return AuthorizedAction(
            action_id=row[0], grant_id=row[1], case_id=row[2], case_fingerprint=row[3],
            invoice_id=row[4], po_id=row[5], supplier_id=row[6], proposed_action=row[7],
            amount_cents=row[8], discrepancy_cents=row[9], policy_version=row[10],
            reviewer=row[11], executed_at=row[12], state=row[13],
        )

    def _invoice_executed(self, supplier_id: str, invoice_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM authorized_actions WHERE supplier_id = ? AND invoice_id = ?",
            (supplier_id, invoice_id),
        ).fetchone()
        return row is not None

    def _action_matches(self, action: AuthorizedAction, scope: dict, reviewer_name: str) -> bool:
        """True if a stored action matches the current case scope and reviewer.

        Used to distinguish a legitimate idempotent replay from a mismatched
        retry (changed case contents or a different reviewer).
        """
        return (
            action.case_fingerprint == scope["fingerprint"]
            and action.invoice_id == scope["invoice_id"]
            and action.po_id == scope["po_id"]
            and action.supplier_id == scope["supplier_id"]
            and action.proposed_action == scope["proposed_action"]
            and action.amount_cents == scope["amount_cents"]
            and action.discrepancy_cents == scope["discrepancy_cents"]
            and action.policy_version == scope["policy_version"]
            and action.reviewer == reviewer_name
        )

    def _insert_action(self, action: AuthorizedAction) -> None:
        """Test seam: insert the durable action row (overridable to force a crash)."""
        self._conn.execute(
            "INSERT INTO authorized_actions (action_id, grant_id, case_id, case_fingerprint, "
            "invoice_id, po_id, supplier_id, proposed_action, amount_cents, discrepancy_cents, "
            "policy_version, reviewer, executed_at, state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id, action.grant_id, action.case_id, action.case_fingerprint,
                action.invoice_id, action.po_id, action.supplier_id, action.proposed_action,
                action.amount_cents, action.discrepancy_cents, action.policy_version,
                action.reviewer, action.executed_at, action.state,
            ),
        )

    def execute(self, grant_id: str, case: CaseInput, reviewer: Reviewer) -> ExecutionResult:
        """Atomically claim and execute a grant for the exact case/action scope.

        A grant is executed only if, inside one transaction on one connection,
        the full binding validates and the single-use transition succeeds. The
        grant transition, claim insert, and action insert commit together; any
        failure rolls back all three, leaving no partial claim or action.
        """
        # 1. Simulation mode.
        if not self._grant._test_mode:
            return ExecutionResult(
                status=ExecutionStatus.SIMULATION_NOT_ENABLED,
                reasons=["execution requires an explicit test-mode opt-in"],
            )

        # 2. Reviewer provenance (identity consistency, not authentication).
        if reviewer.source != ReviewerSource.SIMULATED:
            return ExecutionResult(
                status=ExecutionStatus.REVIEWER_REJECTED,
                reasons=["reviewer identity is not verified; only simulated execution is available"],
            )
        if not reviewer.name.strip():
            return ExecutionResult(
                status=ExecutionStatus.REVIEWER_REJECTED,
                reasons=["reviewer name is blank"],
            )

        # 3. Eligibility: only the human-authorized exception path.
        if review_decision(case) != ReviewDecision.REQUIRES_AUTHORITY:
            return ExecutionResult(
                status=ExecutionStatus.CASE_NOT_ELIGIBLE,
                reasons=["underlying review eligibility is not satisfied"],
            )

        scope = _derive_scope(case)

        # 4. Atomic decision + transition + claim + action, on one connection.
        try:
            self._conn.execute("BEGIN IMMEDIATE")

            grant = self._grant.get(grant_id)
            if grant is None:
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.GRANT_NOT_FOUND,
                    reasons=["no such grant"],
                )
            request = self._review.get(case.case_id)
            if request is None:
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.NO_REVIEW_REQUEST,
                    reasons=["no review request for this case"],
                )

            # Idempotent recovery: already executed. Only return the original
            # receipt if the replay exactly matches the committed execution.
            existing = self._get_action_by_grant(grant_id)
            if existing is not None:
                self._conn.execute("ROLLBACK")
                if self._action_matches(existing, scope, reviewer.name):
                    return ExecutionResult(
                        status=ExecutionStatus.ALREADY_EXECUTED, action=existing,
                        reasons=["grant was already executed"],
                    )
                return ExecutionResult(
                    status=ExecutionStatus.REPLAY_MISMATCH,
                    reasons=["replay does not match the committed execution"],
                )

            # Full revalidation (state, lineage, provenance, review, scope, expiry).
            now = self._clock()
            err = self._grant._revalidate_binding(grant, request, reviewer, self._review, scope, now)
            if err is not None:
                self._conn.execute("ROLLBACK")
                return ExecutionResult(status=ExecutionStatus(err.status.value), reasons=err.reasons)

            # Cross-case invoice dedup.
            if self._invoice_executed(scope["supplier_id"], scope["invoice_id"]):
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.DUPLICATE_INVOICE,
                    reasons=["invoice already approved under another case"],
                )

            # Cross-path registry: case-id and invoice uniqueness across B1/B2.
            if registry_case_conflict(self._conn, scope["case_id"]):
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.CONFLICT,
                    reasons=["case already approved via another path"],
                )
            if registry_invoice_conflict(self._conn, scope["supplier_id"], scope["invoice_id"]):
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.DUPLICATE_INVOICE,
                    reasons=["invoice already approved via another path"],
                )

            # Single-use transition (conditional update gate).
            cur = self._conn.execute(
                "UPDATE grants SET state = ? WHERE grant_id = ? AND state = ?",
                (GrantState.CLAIMED.value, grant_id, GrantState.ISSUED.value),
            )
            if cur.rowcount == 0:
                self._conn.execute("ROLLBACK")
                return ExecutionResult(
                    status=ExecutionStatus.ALREADY_CLAIMED,
                    reasons=["grant was already claimed"],
                )

            # Durable claim record (claim is distinct from execution).
            claimed_at = now.isoformat()
            self._conn.execute(
                "INSERT INTO claims (grant_id, claimed_at, case_id, case_fingerprint, "
                "invoice_id, po_id, supplier_id, proposed_action, amount_cents, "
                "discrepancy_cents, policy_version, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    grant_id, claimed_at, scope["case_id"], scope["fingerprint"],
                    scope["invoice_id"], scope["po_id"], scope["supplier_id"],
                    scope["proposed_action"], scope["amount_cents"], scope["discrepancy_cents"],
                    scope["policy_version"], GrantState.CLAIMED.value,
                ),
            )

            # Durable simulated APPROVE action (the receipt).
            action = AuthorizedAction(
                action_id=uuid.uuid4().hex,
                grant_id=grant_id,
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
                executed_at=now.isoformat(),
                state="COMMITTED",
            )
            self._insert_action(action)
            registry_insert(self._conn, scope["case_id"], scope["supplier_id"], scope["invoice_id"])

            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            self._conn.execute("ROLLBACK")
            if registry_case_conflict(self._conn, scope["case_id"]):
                return ExecutionResult(
                    status=ExecutionStatus.CONFLICT,
                    reasons=["case already approved via another path"],
                )
            if registry_invoice_conflict(self._conn, scope["supplier_id"], scope["invoice_id"]):
                return ExecutionResult(
                    status=ExecutionStatus.DUPLICATE_INVOICE,
                    reasons=["invoice already approved via another path"],
                )
            return ExecutionResult(
                status=ExecutionStatus.ALREADY_CLAIMED,
                reasons=["grant was already claimed or executed"],
            )
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        return ExecutionResult(status=ExecutionStatus.COMMITTED, action=action)
