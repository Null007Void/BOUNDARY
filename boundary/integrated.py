"""Canonical integrated BOUNDARY execution entry point (B2.2b-2a).

Wires the B1 exact-match path and the B2 authorized-exception path to a single
canonical database file, so cross-path invoice and case-id uniqueness hold.

Standalone ``ActionJournal`` / ``AuthorizedExecutor`` instances remain usable
for isolated tests, but they do not provide cross-database uniqueness. For
integrated operation, use :class:`Boundary` with one database path — it cannot
silently give the two paths different persistence files.
"""

from __future__ import annotations

from .contracts import CaseInput
from .execution import AuthorizedExecutor, ExecutionResult
from .grant import Reviewer
from .ledger import ActionJournal, CommitResult


class Boundary:
    """Canonical single-database entry point for both B1 and B2 paths."""

    def __init__(self, path: str, test_mode: bool = False, clock=None):
        self._journal = ActionJournal(path)
        self._executor = AuthorizedExecutor(path, test_mode=test_mode, clock=clock)

    def approve_exact(self, case: CaseInput) -> CommitResult:
        """B1 exact-match approval path (same database)."""
        return self._journal.approve(case)

    def execute(self, grant_id: str, case: CaseInput, reviewer: Reviewer) -> ExecutionResult:
        """B2 human-authorized exception path (same database)."""
        return self._executor.execute(grant_id, case, reviewer)

    def close(self) -> None:
        self._journal.close()
        self._executor.close()
