"""S4E.17: translation bridge from the existing Evolus HTTP client result into the batch's
extraction claims and validated comparison packet.

This exercises the *real* ``EvolusHttpClient`` code path (submission, execution-ID handling,
bounded polling, output validation) but is offline-testable via an injected ``httpx.Client``
(or transport). The entire rehearsal is labelled offline/mock — no Evolus platform is contacted.

Design notes
------------
- Field translation is explicit: ``supplierId -> supplier_id``, ``invoiceId -> invoice_id``,
  ``purchaseOrderReference``/``purchaseOrderId -> po_reference``, ``statedTotal -> amount``.
- The monetary value (``statedTotal``) is passed through *exactly* (raw string preserved); the
  canonical integer-cent value is derived separately via ``money.parse_usd_cents`` during packet
  validation — never rounded, never float arithmetic.
- Confidence, extractor identity, and platform usage are NOT invented: the Evolus extraction
  contract carries no confidence and no usage counters, so each translated claim omits
  confidence and usage stays unknown.
- Extracted claims, trusted lookups, and Evolus advisory analysis remain distinct: advisory
  analysis is preserved as provenance and is never folded into trusted ground truth or
  execution authority.

Invoice-specific validation
---------------------------
The generic ``EvolusHttpClient`` remains capable of other document types. This bridge is
invoice intake: it rejects (never silently re-processes) an extraction whose ``documentType``
is not exactly ``INVOICE`` or whose ``currency`` is not exactly ``USD``, including missing or
unknown metadata. No fixture default is substituted for an absent extraction fact.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Tuple

from .evolus import EvolusOutcome
from .evolus_http import EvolusExecutionResult, EvolusHttpClient


def evolus_extraction_to_claims(extraction: Dict) -> List[dict]:
    """Translate the Evolus HTTP extraction (12-field contract) into the batch's advisory claims.

    Returns a list of ``{field, value}`` dicts (no confidence — not invented). Only fields that
    are present (not ``None``) are mapped. The monetary ``statedTotal`` is preserved raw.
    """
    claims: List[dict] = []

    def _add(src_field: str, dst_field: str) -> None:
        value = extraction.get(src_field)
        if value is not None:
            claims.append({"field": dst_field, "value": str(value)})

    _add("supplierId", "supplier_id")
    _add("invoiceId", "invoice_id")
    po_ref = extraction.get("purchaseOrderReference")
    if po_ref is None:
        po_ref = extraction.get("purchaseOrderId")
    if po_ref is not None:
        claims.append({"field": "po_reference", "value": str(po_ref)})
    _add("statedTotal", "amount")
    return claims


def execution_to_invoice_intake(execution: EvolusExecutionResult) -> Tuple[str, List[dict], Dict]:
    """Translate an HTTP execution result into invoice-intake ``(outcome, claims, provenance)``.

    Validates the extraction's ``documentType`` (must be ``INVOICE``) and ``currency`` (must be
    ``USD``) before translating claims; a PO-identifying response, a conflicting currency, or
    missing metadata yields a non-success outcome and empty claims. ``provenance`` carries the
    workflow execution identity, status, authority, evidence status, advisory analysis, the
    submission mode (text vs file), the submitted-content hash, and separate submission/poll
    call accounting — kept distinct from trusted facts and from downstream model requests.
    """
    submission_mode = "file" if execution.original_file_name else "text"
    if submission_mode == "file":
        submitted_content_sha256 = execution.original_file_sha256
    else:
        submitted_content_sha256 = (
            hashlib.sha256(execution.original_evidence.encode("utf-8")).hexdigest()
            if execution.original_evidence else None
        )
    provenance = {
        "execution_id": execution.execution_id,
        "status": execution.status,
        "authority": execution.authority,
        "evidence_status": execution.evidence_status,
        "advisory_analysis": execution.advisory_analysis,
        "document_type": (execution.extraction or {}).get("documentType"),
        "currency": (execution.extraction or {}).get("currency"),
        "submission_mode": submission_mode,
        "submitted_content_sha256": submitted_content_sha256,
        # Workflow submissions (POST) and status polls (GET) are accounted separately; polls do
        # not establish model-call counts, and platform usage stays unknown.
        "submission_count": 1 if execution.call_count >= 1 else 0,
        "poll_count": max(execution.call_count - 1, 0),
        "call_count": execution.call_count,
        "elapsed_ms": execution.elapsed_ms,
        "reasons": list(execution.reasons),
    }
    if execution.outcome != EvolusOutcome.SUCCESS:
        return execution.outcome.value, [], provenance
    extraction = execution.extraction or {}
    if extraction.get("documentType") != "INVOICE":
        provenance["reasons"].append(f"documentType {extraction.get('documentType')!r} is not INVOICE")
        return "INCOMPLETE_RESPONSE", [], provenance
    if extraction.get("currency") != "USD":
        provenance["reasons"].append(f"currency {extraction.get('currency')!r} is not USD")
        return "INCOMPLETE_RESPONSE", [], provenance
    return "SUCCESS", evolus_extraction_to_claims(extraction), provenance


def evolus_http_claims_fn(client: EvolusHttpClient, *, submission_mode: str = "text") -> Callable:
    """Return a ``claims_fn`` for ``batch.run_intake`` that submits each invoice's document via
    the real HTTP client (offline via the injected transport) and translates the execution
    result.

    ``submission_mode`` selects text vs file submission. The current bridge default (and the
    exercised path) is ``text``; ``file`` reuses the established attachment contract and must be
    proven offline separately before it is treated as verified.

    ``claims_fn(inv) -> (outcome, claims, extractor_id, provenance)``.
    """

    def claims_fn(inv) -> Tuple[str, List[dict], str, Dict]:
        if submission_mode == "file":
            result = client.run(filename=f"{inv.invoice_id}.txt",
                                file_bytes=inv.document.encode("utf-8"),
                                content_type="text/plain")
        else:
            result = client.run(text=inv.document)
        outcome, claims, provenance = execution_to_invoice_intake(result)
        extractor_id = f"evolus-http:{provenance.get('execution_id') or 'unsubmitted'}"
        return outcome, claims, extractor_id, provenance

    return claims_fn
