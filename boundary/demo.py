"""Offline demonstration orchestration for BOUNDARY.

Defines three clearly synthetic development cases and a bounded local-only
``run_demo`` function that executes them through the actual classifier, guard,
SQLite journal, and verifier, returning the full decision/commit/verification
chain for each. No real transactions, payments, or bank integrations.
"""

from __future__ import annotations

from .contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupUnavailable,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from .ledger import ActionJournal
from .verifier import verify


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def demo_cases() -> list[tuple[str, CaseInput]]:
    """Three synthetic development cases (not held-out evaluation data)."""
    # 1. Valid exact match -> eligible -> APPROVE -> committed -> VERIFIED.
    inv1 = Invoice(invoice_id="INV-100", supplier_id="SUP-100", po_reference="PO-100",
                   issued_date="2026-10-12", line_items=[_line(1, "2", "25.00")],
                   stated_total="50.00")
    po1 = PurchaseOrder(po_id="PO-100", supplier_id="SUP-100", issued_date="2026-10-12",
                        line_items=[_line(1, "2", "25.00")], stated_total="50.00")
    case1 = CaseInput(case_id="CASE-1", invoice=inv1,
                      po_lookup=POLookupFound(purchase_order=po1),
                      supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                      policy_version="policy-v1", input_identity="demo")

    # 2. Nonzero invoice/PO discrepancy -> held (no commit).
    inv2 = Invoice(invoice_id="INV-200", supplier_id="SUP-200", po_reference="PO-200",
                   issued_date="2026-10-12", line_items=[_line(1, "1", "100.00")],
                   stated_total="100.00")
    po2 = PurchaseOrder(po_id="PO-200", supplier_id="SUP-200", issued_date="2026-10-12",
                        line_items=[_line(1, "1", "95.00")], stated_total="95.00")
    case2 = CaseInput(case_id="CASE-2", invoice=inv2,
                      po_lookup=POLookupFound(purchase_order=po2),
                      supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                      policy_version="policy-v1", input_identity="demo")

    # 3. Unavailable PO -> held (no commit).
    inv3 = Invoice(invoice_id="INV-300", supplier_id="SUP-300", po_reference="PO-300",
                   issued_date="2026-10-12", line_items=[_line(1, "1", "10.00")],
                   stated_total="10.00")
    case3 = CaseInput(case_id="CASE-3", invoice=inv3,
                      po_lookup=POLookupUnavailable(),
                      supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
                      policy_version="policy-v1", input_identity="demo")

    return [
        ("valid exact match", case1),
        ("nonzero discrepancy (hold)", case2),
        ("unavailable PO (hold)", case3),
    ]


def run_demo(db_path: str) -> list[dict]:
    """Execute the demo cases through the real pipeline and return summaries.

    Each summary distinguishes the decision proposal, the guard decision, the
    committed action (or hold), and the independent verification outcome.
    """
    journal = ActionJournal(db_path)
    results: list[dict] = []
    try:
        for label, case in demo_cases():
            commit = journal.approve(case)  # commits exact match, holds the rest
            verification = verify(journal, case, receipt=commit.receipt)
            results.append({
                "label": label,
                "case": case,
                "proposal": commit.decision.proposal,
                "guard_action": commit.decision.action.value,
                "commit_status": commit.status.value,
                "receipt": commit.receipt,
                "verification": verification,
            })
    finally:
        journal.close()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the BOUNDARY offline demo")
    parser.add_argument("--db", default="boundary.db", help="SQLite journal path")
    args = parser.parse_args()

    for r in run_demo(args.db):
        print(f"{r['label']}: guard={r['guard_action']} commit={r['commit_status']} "
              f"verify={r['verification'].outcome.value}")
