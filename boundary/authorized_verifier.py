"""Read-only independent verifier for the B2 authorized-exception path (B2.2b-2b).

Establishes whether a committed simulated human-authorized action actually exists
and is consistent with the source case, its recorded grant/claim/review lineage,
and the shared execution registry.

The verifier never infers execution from a grant or claim alone, never calls the
executor or claim path, and never repairs the database: a corrupted or
inconsistent database is observed and reported, not silently fixed.

A verified result means consistency with the recorded *simulated* authorization
and source case. It does not establish real reviewer authentication or
independently authenticate opaque business evidence.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CaseInput
from .execution import AuthorizedAction
from .grant import GRANT_TTL_SECONDS, _derive_scope
from .review import ReviewDecision, review_decision


class AuthorizedVerificationOutcome(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"          # case does not require authority
    NOT_FOUND = "NOT_FOUND"                # no execution evidence for this case
    INCOMPLETE = "INCOMPLETE"              # grant/claim present but no committed action
    RECORD_MISMATCH = "RECORD_MISMATCH"    # committed action disagrees with source case
    UNAUTHORIZED = "UNAUTHORIZED"          # authority lineage/registry inconsistent
    RECEIPT_MISMATCH = "RECEIPT_MISMATCH"  # supplied receipt disagrees with the action


class AuthorizedVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: AuthorizedVerificationOutcome
    case_id: str
    reasons: list[str] = Field(default_factory=list)
    action: Optional[AuthorizedAction] = None


def _open_readonly(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _row_to_action(row) -> Optional[AuthorizedAction]:
    if row is None:
        return None
    return AuthorizedAction(
        action_id=row["action_id"], grant_id=row["grant_id"], case_id=row["case_id"],
        case_fingerprint=row["case_fingerprint"], invoice_id=row["invoice_id"],
        po_id=row["po_id"], supplier_id=row["supplier_id"], proposed_action=row["proposed_action"],
        amount_cents=row["amount_cents"], discrepancy_cents=row["discrepancy_cents"],
        policy_version=row["policy_version"], reviewer=row["reviewer"],
        executed_at=row["executed_at"], state=row["state"],
    )


def _scope_mismatches(row, scope: dict, label: str) -> list[str]:
    msgs = []
    if row["case_id"] != scope["case_id"]:
        msgs.append(f"{label} case id disagrees with the source case")
    if row["case_fingerprint"] != scope["fingerprint"]:
        msgs.append(f"{label} case fingerprint disagrees with the source case")
    if row["invoice_id"] != scope["invoice_id"]:
        msgs.append(f"{label} invoice id disagrees with the source case")
    if row["po_id"] != scope["po_id"]:
        msgs.append(f"{label} PO id disagrees with the source case")
    if row["supplier_id"] != scope["supplier_id"]:
        msgs.append(f"{label} supplier id disagrees with the source case")
    if row["proposed_action"] != scope["proposed_action"]:
        msgs.append(f"{label} proposed action disagrees with the source case")
    if row["amount_cents"] != scope["amount_cents"]:
        msgs.append(f"{label} amount disagrees with the source case")
    if row["discrepancy_cents"] != scope["discrepancy_cents"]:
        msgs.append(f"{label} discrepancy disagrees with the source case")
    if row["policy_version"] != scope["policy_version"]:
        msgs.append(f"{label} policy version disagrees with the source case")
    return msgs


def _chronology_msgs(issued_at, expires_at, claimed_at, executed_at) -> list[str]:
    """Validate the execution chronology: issued <= claimed <= executed < expires."""
    try:
        issued = datetime.fromisoformat(issued_at)
        expires = datetime.fromisoformat(expires_at)
        claimed = datetime.fromisoformat(claimed_at)
        executed = datetime.fromisoformat(executed_at)
    except (ValueError, TypeError):
        return ["a timestamp is malformed"]
    for name, dt in (("issued", issued), ("expires", expires), ("claimed", claimed), ("executed", executed)):
        if dt.tzinfo is None:
            return [f"{name} timestamp is naive (missing timezone)"]
    msgs = []
    if expires - issued != timedelta(seconds=GRANT_TTL_SECONDS):
        msgs.append("grant lifetime is not exactly 24 hours")
    if issued > claimed:
        msgs.append("claim precedes grant issuance")
    if claimed > executed:
        msgs.append("execution precedes claim")
    if executed >= expires:
        msgs.append("execution occurs at or after grant expiration")
    return msgs


def _store_id(conn) -> Optional[str]:
    if not _table_exists(conn, "store_meta"):
        return None
    row = conn.execute("SELECT value FROM store_meta WHERE key = 'store_id'").fetchone()
    return row["value"] if row else None


def verify_authorized(
    path: str,
    case: CaseInput,
    grant_id: Optional[str] = None,
    receipt: Optional[AuthorizedAction] = None,
) -> AuthorizedVerificationResult:
    """Independently verify a committed B2 authorized execution against its source case."""
    conn = _open_readonly(path)
    try:
        scope = _derive_scope(case)
        decision = review_decision(case)

        if not _table_exists(conn, "authorized_actions"):
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.NOT_FOUND,
                case_id=case.case_id,
                reasons=["no authorized-action table in this database"],
            )

        # Look up the committed action for this case.
        action = conn.execute(
            "SELECT * FROM authorized_actions WHERE case_id = ?", (case.case_id,)
        ).fetchone()
        if grant_id is not None and action is not None and action["grant_id"] != grant_id:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.RECORD_MISMATCH,
                case_id=case.case_id,
                reasons=["supplied grant id does not match the committed action"],
                action=_row_to_action(action),
            )

        if action is None:
            # No committed action. Distinguish partial from absent evidence.
            has_grant = conn.execute(
                "SELECT 1 FROM grants WHERE case_id = ?", (case.case_id,)
            ).fetchone() is not None
            has_claim = conn.execute(
                "SELECT 1 FROM claims WHERE case_id = ?", (case.case_id,)
            ).fetchone() is not None
            if has_grant or has_claim:
                return AuthorizedVerificationResult(
                    outcome=AuthorizedVerificationOutcome.INCOMPLETE,
                    case_id=case.case_id,
                    reasons=["grant or claim present but no committed action"],
                )
            if decision != ReviewDecision.REQUIRES_AUTHORITY:
                return AuthorizedVerificationResult(
                    outcome=AuthorizedVerificationOutcome.NOT_ELIGIBLE,
                    case_id=case.case_id,
                    reasons=["case does not require authority"],
                )
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.NOT_FOUND,
                case_id=case.case_id,
                reasons=["no committed action for this case"],
            )

        action_obj = _row_to_action(action)

        # A committed action exists for a case that does not require authority.
        if decision != ReviewDecision.REQUIRES_AUTHORITY:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["committed action exists for a case not requiring authority"],
                action=action_obj,
            )

        # The action must agree with the independently derived source-case scope.
        scope_msgs = _scope_mismatches(action, scope, "action")
        if scope_msgs:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.RECORD_MISMATCH,
                case_id=case.case_id,
                reasons=scope_msgs,
                action=action_obj,
            )
        if action["state"] != "COMMITTED":
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["committed action is not in COMMITTED state"],
                action=action_obj,
            )

        # No other B2 action may share this case id or invoice identity.
        dup_count = conn.execute(
            "SELECT COUNT(*) FROM authorized_actions "
            "WHERE (case_id = ? OR (supplier_id = ? AND invoice_id = ?)) AND grant_id != ?",
            (case.case_id, scope["supplier_id"], scope["invoice_id"], action["grant_id"]),
        ).fetchone()[0]
        if dup_count > 0:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["contradictory B2 approval evidence exists for this case/invoice"],
                action=action_obj,
            )

        # Grant lineage.
        grant = conn.execute(
            "SELECT * FROM grants WHERE grant_id = ?", (action["grant_id"],)
        ).fetchone()
        if grant is None:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["committed action has no matching grant"],
                action=action_obj,
            )
        grant_msgs = _scope_mismatches(grant, scope, "grant")
        if grant["state"] != "CLAIMED":
            grant_msgs.append("grant is not in CLAIMED state")
        if grant["reviewer"] != action["reviewer"]:
            grant_msgs.append("grant reviewer disagrees with the committed action reviewer")
        if grant["reviewer_source"] != "SIMULATED":
            grant_msgs.append("grant reviewer provenance is not simulated")
        if grant["review_store_id"] != _store_id(conn):
            grant_msgs.append("grant was issued by a different review store")
        if grant_msgs:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=grant_msgs,
                action=action_obj,
            )

        # Original review request.
        review = conn.execute(
            "SELECT * FROM review_requests WHERE case_id = ?", (case.case_id,)
        ).fetchone()
        if review is None:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["committed action has no matching review request"],
                action=action_obj,
            )
        review_msgs = _scope_mismatches(review, scope, "review")
        if review["state"] != "WAITING_AUTHORITY":
            review_msgs.append("review request is not pending")
        if review["id"] != grant["request_id"]:
            review_msgs.append("review request id disagrees with the grant")
        if review_msgs:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=review_msgs,
                action=action_obj,
            )

        # Claim record (single-use semantics).
        claim = conn.execute(
            "SELECT * FROM claims WHERE grant_id = ?", (action["grant_id"],)
        ).fetchone()
        if claim is None:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["committed action has no matching claim record"],
                action=action_obj,
            )
        claim_msgs = _scope_mismatches(claim, scope, "claim")
        if claim["state"] != "CLAIMED":
            claim_msgs.append("claim record is not in CLAIMED state")
        if claim_msgs:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=claim_msgs,
                action=action_obj,
            )

        # Execution chronology: issued <= claimed <= executed < expires.
        chrono_msgs = _chronology_msgs(
            grant["issued_at"], grant["expires_at"], claim["claimed_at"], action["executed_at"]
        )
        if chrono_msgs:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=chrono_msgs,
                action=action_obj,
            )

        # Execution registry must match, with no contradictory B1 approval.
        registry = conn.execute(
            "SELECT * FROM execution_registry WHERE case_id = ?", (case.case_id,)
        ).fetchone()
        if registry is None:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["no execution-registry entry for this action"],
                action=action_obj,
            )
        if registry["supplier_id"] != scope["supplier_id"] or registry["invoice_id"] != scope["invoice_id"]:
            return AuthorizedVerificationResult(
                outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                case_id=case.case_id,
                reasons=["execution-registry entry disagrees with the committed action"],
                action=action_obj,
            )
        if _table_exists(conn, "actions"):
            b1_conflict = conn.execute(
                "SELECT 1 FROM actions WHERE case_id = ? OR (supplier_id = ? AND invoice_id = ?)",
                (case.case_id, scope["supplier_id"], scope["invoice_id"]),
            ).fetchone()
            if b1_conflict is not None:
                return AuthorizedVerificationResult(
                    outcome=AuthorizedVerificationOutcome.UNAUTHORIZED,
                    case_id=case.case_id,
                    reasons=["contradictory B1 approval evidence exists for this case/invoice"],
                    action=action_obj,
                )

        # A supplied receipt must agree field-by-field with the persisted action.
        if receipt is not None:
            receipt_msgs = [
                f for f in (
                    "action_id", "grant_id", "case_id", "case_fingerprint", "invoice_id",
                    "po_id", "supplier_id", "proposed_action", "amount_cents",
                    "discrepancy_cents", "policy_version", "reviewer", "executed_at", "state",
                ) if getattr(receipt, f) != action[f]
            ]
            if receipt_msgs:
                return AuthorizedVerificationResult(
                    outcome=AuthorizedVerificationOutcome.RECEIPT_MISMATCH,
                    case_id=case.case_id,
                    reasons=[f"supplied receipt {f} disagrees with the persisted action" for f in receipt_msgs],
                    action=action_obj,
                )

        return AuthorizedVerificationResult(
            outcome=AuthorizedVerificationOutcome.VERIFIED,
            case_id=case.case_id,
            action=action_obj,
        )
    finally:
        conn.close()
