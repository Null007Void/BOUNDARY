"""S4E.17 Part A — invoice-only synthetic document.

A separately versioned document representation containing ONLY the invoice's own declared
fields: identity, supplier, declared PO reference, currency, line items, and stated amount
(where present). Trusted PO records, lookup status, comparison conclusions, route labels, and
expected answers are deliberately excluded. Those trusted facts are joined into the common
comparison packet later, after extraction validation, through the existing mechanism
(``batch._build_model_evidence``).

This module is prospective: ``INVOICE_DOCUMENT_VERSION`` records the format version so a
future format change is distinguishable from a content change, and ``document_byte_hash``
records the SHA-256 of the submitted bytes. The frozen workload-v2 and the historical
``build_evidence`` are untouched.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

from .contracts import Invoice
from .money import to_cents

INVOICE_DOCUMENT_VERSION = "invoice-document-v1"


def _money(m: Decimal) -> str:
    """Format a monetary Decimal as ``$X.XX`` using exact cents (no float arithmetic)."""
    cents = to_cents(m)
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"${sign}{c // 100}.{c % 100:02d}"


def _quantity(q: Decimal) -> str:
    """Faithful decimal rendering of a quantity (no trailing-digit invention)."""
    return format(q, "f")


def build_invoice_document(invoice: Invoice) -> str:
    """Build a faithful, deterministic text document from ONLY the invoice's declared fields.

    Deliberately excludes the trusted purchase order, lookup state, comparison conclusions,
    route labels, and expected answers.
    """
    ref = invoice.po_reference or "(none)"
    lines = [
        f"INVOICE {invoice.invoice_id}",
        f"Supplier: {invoice.supplier_id}",
        f"Purchase order reference: {ref}",
        f"Currency: {invoice.currency}",
        f"Issued: {invoice.issued_date}",
        "Line items:",
    ]
    for item in invoice.line_items:
        lines.append(
            f"  {item.line_number}. {item.description} qty={_quantity(item.quantity)} "
            f"@ {_money(item.unit_price)}"
        )
    lines.append(f"Stated total: {_money(invoice.stated_total)}")
    return "\n".join(lines)


def document_byte_hash(document: str) -> str:
    """SHA-256 of the document's UTF-8 bytes."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()
