"""S4E.16 — batch-capable invoice workflow with a four-path experimental view.

A practical invoice application slice: one batch of the six approved development cases flows
through Evolus document intake (offline doubles) into an immutable comparison input, then runs the
four model/strategy paths (DeepSeek/Gemma × LLM-first/BOUNDARY) with injected offline transports,
records a correlated durable event stream, and exposes an operational outcome (DeepSeek+BOUNDARY)
distinct from three shadow evaluation paths.

Offline only: no real AMD inference, no real Evolus execution, no network. All model and Evolus
transports are injected offline doubles.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, List, Optional

from .classifier import classify
from .contracts import CaseInput, POState, SupplierMatchState
from .evolus import (
    DocumentExtractionRequest,
    DocumentExtractionResult,
    EvolusAdapter,
    EvolusConfig,
    ExtractionClaim,
)
from .experiment_engine import (
    DESIGNATED_MODEL_DEEPSEEK,
    DESIGNATED_MODEL_GEMMA,
    EXPECTED_ROUTE,
    LIVE_UNCONFIRMED,
    LIVE_ZERO_REQUEST,
    STRATEGY_BOUNDARY,
    STRATEGY_LLM_FIRST,
    ObservationState,
    ExperimentRecord,
    _classify_model_identity,
    _interpretation_dict,
    _mock_interpretation,
    build_workload_v2,
    evidence_id,
    expected_interpretation,
    interpretation_is_correct,
    verify_records,
)
from .guard import ActionDecision, evaluate
from .inference import InferenceAdapter, InferenceConfig
from .invoice_document import INVOICE_DOCUMENT_VERSION, build_invoice_document, document_byte_hash
from .ledger import ActionJournal
from .live_experiment import build_evidence
from .money import parse_usd_cents
from .preflight import select_dev_sample
from .router import Route, route
from .verifier import verify

BATCH_SIZE_LIMIT = 6
MODELS = (DESIGNATED_MODEL_DEEPSEEK, DESIGNATED_MODEL_GEMMA)
STRATEGIES = (STRATEGY_LLM_FIRST, STRATEGY_BOUNDARY)

# The operational path (selected before processing starts): DeepSeek + BOUNDARY.
OPERATIONAL_MODEL = DESIGNATED_MODEL_DEEPSEEK
OPERATIONAL_STRATEGY = STRATEGY_BOUNDARY

REQUIRED_FIELDS = ["supplier_id", "invoice_id", "po_reference", "amount"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sum_tokens(values) -> Optional[int]:
    """Sum token values; returns ``None`` (unknown/incomplete usage) when any value is missing."""
    vals = list(values)
    if any(v is None for v in vals):
        return None
    return sum(vals)


# --- Intake --------------------------------------------------------------------------

@dataclass
class InvoiceRecord:
    invoice_id: str
    case_id: str
    family: str
    content_hash: str
    case: CaseInput
    document: str
    declared_po_ref: Optional[str]
    trusted_po_state: str
    trusted_supplier_state: str
    document_version: str = INVOICE_DOCUMENT_VERSION  # invoice-only document format version
    document_hash: str = ""  # SHA-256 of the submitted document bytes
    status: str = "INTAKE"


@dataclass
class BatchManifest:
    batch_id: str
    created_at: str
    batch_size_limit: int
    invoices: List[InvoiceRecord]

    @property
    def invoice_count(self) -> int:
        return len(self.invoices)


@dataclass
class ComparisonPacket:
    """A single validated comparison input built from document extraction (advisory) plus
    trusted supporting facts. Every consumer (all four paths and operational processing) must
    use this packet and its ``packet_hash`` — never silently revert to the original fixture."""
    invoice_id: str
    case_id: str
    family: str
    case: CaseInput                       # trusted facts (deterministic routing/guard)
    evidence_text: str                    # serialized extracted claims — what a model sees
    extracted_claims: List[dict]          # advisory extracted claims (not trusted)
    extraction_outcome: str              # Evolus extraction outcome
    extractor_id: Optional[str]          # provenance of the extraction double
    trusted_po_state: str                # trusted lookup result (FOUND/NOT_FOUND/UNAVAILABLE)
    trusted_po_id: Optional[str]
    trusted_supplier_state: str
    amount_cents: int
    packet_hash: str                     # immutable identity of the validated packet
    valid: bool                          # False when extraction failed or claims contradict
    block_reason: Optional[str]          # why this invoice is blocked (None when valid)
    status: str = "VALIDATED"            # VALIDATED or BLOCKED


@dataclass
class PathResult:
    model_id: str
    strategy: str
    records: List[ExperimentRecord]
    made_calls: int
    bypassed_calls: int
    blocked: int                     # invoices blocked by invalid intake (no observation)
    total_tokens: Optional[int]
    interpretation_checked: int     # model-interpretation checks (applicable denominator)
    interpretation_correct: int     # of those checks, independently evaluated correct
    no_call: int                    # bypassed invoices (no model call, no interpretation)
    route_correct: int              # independently evaluated routing (BOUNDARY only)
    route_total: int                # records carrying a router decision
    authority_none: int
    packet_hash_by_case: Dict[str, str] = field(default_factory=dict)  # case_id -> packet_hash


# --- Event stream --------------------------------------------------------------------

@dataclass
class BatchEvent:
    event_id: str
    batch_id: str
    run_id: str
    invoice_id: str
    path: str            # "INTAKE" or "DeepSeek/LLM_FIRST" etc.
    phase: str           # INTAKE / EXTRACT / VALIDATE / LOOKUP / REQUEST / BYPASS / GUARD / OUTCOME
    timestamp: str
    payload: dict

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "invoice_id": self.invoice_id,
            "path": self.path,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EventStore:
    """Append-only durable event log (JSON Lines) for the batch, plus in-memory index."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.events: List[BatchEvent] = []
        self.committed: List[BatchEvent] = []  # events durably persisted (fsync succeeded)

    def append(self, event: BatchEvent) -> None:
        self.events.append(event)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            # Mark committed only after the durable write succeeded; an in-memory-only store
            # (path=None) has no durable storage and cannot satisfy operational durability.
            self.committed.append(event)

    def for_invoice(self, invoice_id: str) -> List[BatchEvent]:
        return [e for e in self.events if e.invoice_id == invoice_id]

    def for_path(self, path: str) -> List[BatchEvent]:
        return [e for e in self.events if e.path == path]


# --- Batch construction --------------------------------------------------------------

def build_batch(batch_id: str = "batch-001", selected_invoice_id: Optional[str] = None) -> BatchManifest:
    """Build the six-invoice batch from the frozen workload-v2 development sample.

    ``selected_invoice_id`` optionally bounds the batch to a single development invoice (the
    explicit S4E.17 Part B integrated-trial selection). When omitted, the full six-invoice
    batch is built; a selection must be explicit, never accidental.
    """
    manifest = build_workload_v2()
    sample = select_dev_sample(manifest)
    if selected_invoice_id is not None:
        sample = [wc for wc in sample if wc.case.invoice.invoice_id == selected_invoice_id]
        if not sample:
            raise ValueError(f"no development invoice matches {selected_invoice_id!r}")
    invoices: List[InvoiceRecord] = []
    for wc in sample:
        case = wc.case
        doc = build_invoice_document(case.invoice)
        invoices.append(InvoiceRecord(
            invoice_id=case.invoice.invoice_id,
            case_id=case.case_id,
            family=wc.family,
            content_hash=wc.content_hash,
            case=case,
            document=doc,
            declared_po_ref=case.invoice.po_reference,
            trusted_po_state=case.po_lookup.state,
            trusted_supplier_state=case.supplier_lookup.state,
            document_version=INVOICE_DOCUMENT_VERSION,
            document_hash=document_byte_hash(doc),
        ))
    return BatchManifest(
        batch_id=batch_id,
        created_at=_utcnow(),
        batch_size_limit=BATCH_SIZE_LIMIT,
        invoices=invoices,
    )


# --- Evolus offline extraction double ------------------------------------------------

def _evolus_extract_double(record: InvoiceRecord) -> Callable[[dict], dict]:
    """An offline Evolus extraction double: returns the case's declared fields as advisory claims."""
    def transport(payload: dict) -> dict:
        case = record.case
        inv = case.invoice
        amount = f"${inv.stated_total_cents / 100:.2f}"
        claims = [
            {"field": "supplier_id", "value": inv.supplier_id, "confidence": 1.0},
            {"field": "invoice_id", "value": inv.invoice_id, "confidence": 1.0},
            {"field": "po_reference", "value": inv.po_reference or "", "confidence": 1.0},
            {"field": "amount", "value": amount, "confidence": 1.0},
        ]
        return {
            "document_id": payload["document_id"],
            "extractor_id": "evolus-extraction-double",
            "claims": claims,
            "usage": {"extraction_calls": 1},
        }
    return transport


def _parse_amount_cents(amount_str: str) -> Optional[int]:
    """Parse a USD amount string to integer cents via the exact-money primitives;
    ``None`` when unsupported/unparseable (never rounded)."""
    try:
        return parse_usd_cents(amount_str)
    except (ValueError, InvalidOperation):
        return None


def _build_model_evidence(extracted_claims: List[dict], case: CaseInput,
                          trusted_po_state: str, trusted_supplier_state: str) -> str:
    """Model-facing evidence: validated extracted claims (advisory) plus the trusted supporting
    facts a model needs to interpret the invoice (PO lookup state and retrieved amount), with
    the two sources kept distinct. A model receiving only this JSON can compute its own
    comparison; the expected answer is never inserted."""
    extracted = {c["field"]: c["value"] for c in extracted_claims}
    po = case.po_lookup.purchase_order if case.po_lookup.state == POState.FOUND else None
    trusted = {
        "supplier_state": trusted_supplier_state,
        "po_state": trusted_po_state,
        "po_id": po.po_id if po else None,
        "po_amount_cents": po.computed_total_cents if po else None,
    }
    return json.dumps({"extracted": extracted, "trusted": trusted}, sort_keys=True)


def _interpretation_from_evidence(evidence_text: str) -> str:
    """Derive a source-consistent interpretation from the actual model-facing evidence, not a
    hidden fixture. Mirrors ``_mock_interpretation`` but reads only the outgoing payload."""
    try:
        ev = json.loads(evidence_text)
    except Exception:
        return "{}"
    extracted = ev.get("extracted", {})
    trusted = ev.get("trusted", {})
    inv_amount = _parse_amount_cents(extracted.get("amount", ""))
    po_state = trusted.get("po_state")
    po_amount = trusted.get("po_amount_cents")
    if po_state != "FOUND" or po_amount is None:
        return json.dumps({
            "supplier_id": extracted.get("supplier_id"),
            "invoice_id": extracted.get("invoice_id"),
            "po_reference": None,
            "amount_cents": inv_amount,
            "discrepancy_cents": 0,  # absent-PO convention (an unavailable comparison, not a verified zero)
            "confidence": 0.5,
            "notes": "no trusted purchase order",
        })
    disc = (inv_amount - po_amount) if inv_amount is not None else None
    return json.dumps({
        "supplier_id": extracted.get("supplier_id"),
        "invoice_id": extracted.get("invoice_id"),
        "po_reference": extracted.get("po_reference"),
        "amount_cents": inv_amount,
        "discrepancy_cents": disc,
        "confidence": 0.9,
        "notes": "",
    })


def _validate_packet(extraction_outcome: str, claims: List[dict], case: CaseInput) -> tuple[bool, Optional[str]]:
    """Validate extracted claims against trusted facts; return ``(valid, block_reason)``.

    A failed extraction, a missing required field, or a claim that contradicts a trusted fact
    (including the invoice identity) blocks the invoice.
    """
    if extraction_outcome != "SUCCESS":
        return False, f"extraction failed ({extraction_outcome})"
    claim_map = {c["field"]: c["value"] for c in claims}
    inv = case.invoice
    # Invoice identity: the extracted invoice_id must match the document's own invoice.
    if claim_map.get("invoice_id") != inv.invoice_id:
        return False, "contradictory claims (invoice identity)"
    # Supplier association.
    if case.supplier_lookup.state == SupplierMatchState.MATCH:
        if claim_map.get("supplier_id") != inv.supplier_id:
            return False, "contradictory claims (supplier)"
    # Declared PO reference fidelity: the extracted reference must match the invoice's declared
    # reference, even when no trusted PO was retrieved (declared vs retrieved stay distinct).
    declared_ref = inv.po_reference or ""
    if claim_map.get("po_reference") != declared_ref:
        return False, "contradictory claims (declared purchase order reference)"
    # When a trusted PO was retrieved, the declared reference must also match it.
    if case.po_lookup.state == POState.FOUND:
        po = case.po_lookup.purchase_order
        if claim_map.get("po_reference") != po.po_id:
            return False, "contradictory claims (purchase order)"
    # Amount must be present and match the trusted invoice total.
    amount_str = claim_map.get("amount", "")
    if not amount_str:
        return False, "contradictory claims (missing amount)"
    amount_cents = _parse_amount_cents(amount_str)
    if amount_cents is None:
        return False, "contradictory claims (malformed amount)"
    if amount_cents != inv.computed_total_cents:
        return False, "contradictory claims (amount)"
    return True, None


def run_intake(
    batch: BatchManifest,
    store: EventStore,
    extract_fn: Optional[Callable[[InvoiceRecord], Callable[[dict], dict]]] = None,
    claims_fn: Optional[Callable] = None,
    stop_flag: Optional[Dict[str, bool]] = None,
    run_id: str = "s4e16-batch",
) -> Dict[str, ComparisonPacket]:
    """Run Evolus intake (offline double) once per invoice; produce validated comparison packets.

    ``extract_fn`` lets tests inject an altered or failing extraction transport to prove the
    packet (not the original fixture) is what downstream consumers use. ``claims_fn`` (when
    provided) replaces the transport path entirely: it is called as
    ``claims_fn(inv) -> (outcome, claims, extractor_id, provenance)`` and is the S4E.17 bridge
    entry point for the real Evolus HTTP client (offline via an injected HTTP transport).
    ``stop_flag`` requests cancellation before each next extraction.
    """
    results: Dict[str, ComparisonPacket] = {}
    adapter = EvolusAdapter(EvolusConfig())
    for inv in batch.invoices:
        if stop_flag is not None and stop_flag.get("stop"):
            break
        store.append(BatchEvent(
            event_id=_sha256(f"{run_id}|{inv.invoice_id}|intake"),
            batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path="INTAKE",
            phase="INTAKE", timestamp=_utcnow(),
            payload={"document": inv.document, "document_sha256": _sha256(inv.document),
                     "document_version": inv.document_version},
        ))
        request = DocumentExtractionRequest(
            document_id=inv.invoice_id, document_content=inv.document, fields=REQUIRED_FIELDS,
        )
        if claims_fn is not None:
            outcome, claims, extractor_id, provenance = claims_fn(inv)
        else:
            adapter._extract_transport = (extract_fn(inv) if extract_fn is not None
                                          else _evolus_extract_double(inv))
            call = adapter.extract_document(request)
            claims = [c.model_dump() for c in (call.result.claims if call.result else [])]
            outcome = call.outcome.value
            extractor_id = (call.result.extractor_id if call.result else None)
            provenance = None
        store.append(BatchEvent(
            event_id=_sha256(f"{run_id}|{inv.invoice_id}|extract"),
            batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path="INTAKE",
            phase="EXTRACT", timestamp=_utcnow(),
            payload={"outcome": outcome,
                     "extractor_id": extractor_id,
                     "claims": claims,
                     "provenance": provenance},
        ))
        case = inv.case
        trusted_po_id = (case.po_lookup.purchase_order.po_id
                         if case.po_lookup.state == POState.FOUND else None)
        evidence_text = _build_model_evidence(claims, case, inv.trusted_po_state,
                                              inv.trusted_supplier_state)
        valid, block_reason = _validate_packet(outcome, claims, case)
        # Immutable packet identity: extracted (advisory) claims + trusted lookups, kept distinct.
        snapshot = {
            "invoice_id": inv.invoice_id,
            "case_id": inv.case_id,
            "declared_po_reference": inv.declared_po_ref,
            "extracted_claims": claims,
            "evidence_text": evidence_text,
            "trusted_po_state": inv.trusted_po_state,
            "trusted_po_id": trusted_po_id,
            "trusted_supplier_state": inv.trusted_supplier_state,
            "amount_cents": case.invoice.computed_total_cents,
        }
        packet_hash = _sha256(json.dumps(snapshot, sort_keys=True))
        store.append(BatchEvent(
            event_id=_sha256(f"{run_id}|{inv.invoice_id}|lookup"),
            batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path="INTAKE",
            phase="LOOKUP", timestamp=_utcnow(),
            payload={"trusted_po_state": inv.trusted_po_state,
                     "trusted_po_id": trusted_po_id,
                     "trusted_supplier_state": inv.trusted_supplier_state,
                     "packet_hash": packet_hash, "valid": valid,
                     "block_reason": block_reason},
        ))
        results[inv.invoice_id] = ComparisonPacket(
            invoice_id=inv.invoice_id,
            case_id=inv.case_id,
            family=inv.family,
            case=case,
            evidence_text=evidence_text,
            extracted_claims=claims,
            extraction_outcome=outcome,
            extractor_id=extractor_id,
            trusted_po_state=inv.trusted_po_state,
            trusted_po_id=trusted_po_id,
            trusted_supplier_state=inv.trusted_supplier_state,
            amount_cents=case.invoice.computed_total_cents,
            packet_hash=packet_hash,
            valid=valid,
            block_reason=block_reason,
            status=("VALIDATED" if valid else "BLOCKED"),
        )
    return results


# --- Model offline transports (injected doubles) -------------------------------------

def _model_double(model_id: str) -> Callable[[dict], dict]:
    """A source-consistent offline model double that answers from the actual outgoing payload,
    never from a hidden fixture. Gemma fences its JSON (normalized)."""
    is_gemma = model_id == DESIGNATED_MODEL_GEMMA
    endpoint = f"self-dploy/{model_id}"

    def transport(payload: dict) -> dict:
        user_msg = payload["messages"][1]["content"]
        body = _interpretation_from_evidence(user_msg)
        if is_gemma:
            body = f"```json\n{body}\n```"
        usage = ({"prompt_tokens": 152, "completion_tokens": 101, "total_tokens": 253}
                 if is_gemma else
                 {"prompt_tokens": 120, "completion_tokens": 99, "total_tokens": 219})
        return {"model": endpoint, "choices": [{"message": {"content": body}}], "usage": usage}

    return transport


# --- Four-path execution --------------------------------------------------------------

def _record_to_dict(r: ExperimentRecord) -> dict:
    """Serialize one observation to metadata-only JSON (interpretation, usage, model identity,
    verification, provenance) so summaries can be reconstructed without raw response text."""
    return {
        "run_id": r.run_id,
        "model_id": r.model_id,
        "strategy": r.strategy,
        "repetition": r.repetition,
        "case_id": r.case_id,
        "family": r.family,
        "split": r.split,
        "state": r.state.value if hasattr(r.state, "value") else str(r.state),
        "evidence_id": r.evidence_id,
        "route": r.route,
        "route_correct": r.route_correct,
        "model_calls": r.model_calls,
        "advisory_outcome": r.advisory_outcome,
        "advisory_consistent": r.advisory_consistent,
        "interpretation": r.interpretation,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
        "elapsed_ms": r.elapsed_ms,
        "endpoint_model_id": r.endpoint_model_id,
        "model_verified": r.model_verified,
        "normalized": r.normalized,
        "execution_authority": r.execution_authority,
        "provenance": r.provenance,
        "diagnostics": r.diagnostics,
    }


def _observation_dict(r: ExperimentRecord, batch_id: str, invoice_id: str,
                      packet_hash: Optional[str]) -> dict:
    """The complete metadata-only observation, including batch/invoice identity and the canonical
    comparison-packet fingerprint, as it is durably persisted and reported in the result."""
    d = _record_to_dict(r)
    d["batch_id"] = batch_id
    d["invoice_id"] = invoice_id
    d["packet_hash"] = packet_hash
    return d


def _commit_observation(store: EventStore, batch: BatchManifest, run_id: str, path: str,
                        inv: "InvoiceRecord", rec: ExperimentRecord,
                        packet_hash: Optional[str]) -> None:
    """Durably persist one complete observation *before* it is marked available downstream.

    The write is append-only and flushed/fsync'd (see ``EventStore.append``); a failure raises
    rather than being treated as a successful completion, so an uncommitted observation can never
    fall through into an operational approval or an advanced progress count.
    """
    store.append(BatchEvent(
        event_id=_sha256(f"{run_id}|{inv.invoice_id}|{path}|observation|{rec.repetition}"),
        batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path=path,
        phase="OBSERVATION", timestamp=_utcnow(),
        payload=_observation_dict(rec, batch.batch_id, inv.invoice_id, packet_hash),
    ))


def read_durable_observations(event_path: Optional[str], run_id: Optional[str] = None) -> List[dict]:
    """Read the complete observations persisted durably to the event log (read-only).

    Returns the full metadata-only observation dicts in append order. When ``run_id`` is given,
    only observations for that run are returned, so a later run never inherits earlier-run
    records. This is the durable source of reported results: it survives an interruption even
    when no final result file was written. Missing, absent, or unreadable files yield an empty
    list (never fabricated observations).
    """
    observations: List[dict] = []
    if not event_path or not os.path.exists(event_path):
        return observations
    try:
        with open(event_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("phase") != "OBSERVATION":
                    continue
                if run_id is not None and e.get("run_id") != run_id:
                    continue
                observations.append(e.get("payload", {}))
    except Exception:
        return observations
    return observations


def read_durable_observation_events(event_path: Optional[str], run_id: Optional[str] = None):
    """Read the OBSERVATION events (full wrapper) for the selected run, plus a corruption flag.

    Returns ``(events, corrupt)``. ``corrupt`` is True when any line could not be parsed, so a
    caller can distinguish an absent/empty log (not-started) from an unreadable one. Readable
    observations are still returned alongside the flag (a valid prefix is preserved).
    """
    events: List[dict] = []
    corrupt = False
    if not event_path or not os.path.exists(event_path):
        return events, False
    try:
        with open(event_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    corrupt = True
                    continue
                if e.get("phase") != "OBSERVATION":
                    continue
                if run_id is not None and e.get("run_id") != run_id:
                    continue
                events.append(e)
    except Exception:
        corrupt = True
    return events, corrupt


def _validate_observation_set(events: List[dict], batch: BatchManifest, run_id: str,
                              expected_packet_hashes: Dict[str, Optional[str]]):
    """Validate the selected run's observation set; return ``(valid_payloads, issues)``.

    Validates identity (run wrapper == payload), membership (model/strategy/case), evidence
    identity, packet association, and duplicate detection. Foreign identities, conflicting or
    duplicate evidence ids, and mismatched evidence/packet identities are reported as issues and
    excluded from the valid set — they must not yield complete accounting.
    """
    valid: List[dict] = []
    issues: List[str] = []
    seen = set()
    expected_cases = {i.case_id for i in batch.invoices}
    case_to_invoice = {i.case_id: i.invoice_id for i in batch.invoices}
    expected_repetition = 0  # the accepted plan's repetition set for this slice
    for e in events:
        payload = e.get("payload", {})
        case = payload.get("case_id")
        tag = f"{case or '?'}:"
        if e.get("run_id") != run_id or payload.get("run_id") != run_id:
            issues.append(f"{tag} run identity mismatch")
            continue
        model = payload.get("model_id")
        strategy = payload.get("strategy")
        if model not in MODELS:
            issues.append(f"{tag} foreign model {model!r}")
            continue
        if strategy not in STRATEGIES:
            issues.append(f"{tag} foreign strategy {strategy!r}")
            continue
        if case not in expected_cases:
            issues.append(f"{tag} foreign case")
            continue
        # Wrapper and payload batch_id must both equal the manifest's ID.
        if e.get("batch_id") != batch.batch_id or payload.get("batch_id") != batch.batch_id:
            issues.append(f"{tag} batch association mismatch "
                          f"(wrapper={e.get('batch_id')!r}, payload={payload.get('batch_id')!r})")
            continue
        # Wrapper and payload invoice_id must both equal the case's invoice in the manifest.
        if e.get("invoice_id") != case_to_invoice.get(case) or payload.get("invoice_id") != case_to_invoice.get(case):
            issues.append(f"{tag} invoice association mismatch "
                          f"(wrapper={e.get('invoice_id')!r}, payload={payload.get('invoice_id')!r})")
            continue
        repetition = payload.get("repetition")
        if repetition != expected_repetition:
            issues.append(f"{tag} unplanned repetition {repetition!r}")
            continue
        expected_eid = evidence_id(run_id, model, strategy, expected_repetition, case)
        if payload.get("evidence_id") != expected_eid:
            issues.append(f"{tag} evidence id mismatch")
            continue
        if case in expected_packet_hashes and payload.get("packet_hash") != expected_packet_hashes[case]:
            issues.append(f"{tag} packet association mismatch")
            continue
        eid = payload.get("evidence_id")
        if eid in seen:
            issues.append(f"{tag} duplicate evidence id")
            continue
        seen.add(eid)
        valid.append(payload)
    return valid, issues


def _compute_path_metrics(records: List[ExperimentRecord]) -> dict:
    """Interpretation checks use the applicable denominator (model calls only); no-call
    (bypassed) invoices, blocked invoices, and router decisions are reported separately."""
    interpretation_checked = sum(1 for r in records if (r.model_calls or 0) > 0)
    interpretation_correct = sum(
        1 for r in records if (r.model_calls or 0) > 0 and r.advisory_consistent is True)
    blocked = sum(1 for r in records if r.state == ObservationState.FAILED)
    no_call = sum(
        1 for r in records
        if (not r.model_calls or r.model_calls == 0) and r.state != ObservationState.FAILED)
    route_total = sum(1 for r in records if r.route is not None)
    route_correct = sum(1 for r in records if r.route is not None and r.route_correct is True)
    return {
        "interpretation_checked": interpretation_checked,
        "interpretation_correct": interpretation_correct,
        "no_call": no_call,
        "blocked": blocked,
        "route_correct": route_correct,
        "route_total": route_total,
    }


def run_paths(
    batch: BatchManifest,
    store: EventStore,
    packets: Optional[Dict[str, ComparisonPacket]] = None,
    run_id: str = "s4e16-batch",
    stop_flag: Optional[Dict[str, bool]] = None,
    model_transport_fn: Optional[Callable] = None,
    stop_on_error: bool = False,
    provenance_label: str = "OBSERVED_OFFLINE",
) -> List[PathResult]:
    """Run the four model/strategy paths over validated packets, each with isolated state.

    Every path consumes the same validated :class:`ComparisonPacket` (extracted evidence text +
    trusted facts) — never the original fixture. ``packets`` defaults to a fresh intake run.
    ``model_transport_fn`` (when provided) is the injected live-provider (AMD) transport factory.
    ``stop_on_error`` + ``provenance_label`` are the S4E.17 trial-mode controls.
    """
    if packets is None:
        packets = run_intake(batch, store)
    results: List[PathResult] = []
    for model_id in MODELS:
        for strategy in STRATEGIES:
            if stop_flag is not None and stop_flag.get("stop"):
                raise StopRequested("batch cancelled before dispatch")
            results.append(_run_single_path(batch, packets, store, model_id, strategy,
                                            run_id=run_id, stop_flag=stop_flag,
                                            model_transport_fn=model_transport_fn,
                                            stop_on_error=stop_on_error,
                                            provenance_label=provenance_label))
    return results


# --- Operational path (DeepSeek + BOUNDARY) -------------------------------------------

def _reason_label(reason_codes) -> str:
    """Map guard reason codes to a concise human-readable cause (missing evidence vs
    unsupported discrepancy vs structural). Empty for an approvable (exact-match) case."""
    codes = {getattr(rc, "value", str(rc)) for rc in (reason_codes or [])}
    if "PO_NOT_FOUND" in codes or "PO_UNAVAILABLE" in codes:
        return "missing evidence (no trusted purchase order)"
    if "NONZERO_DISCREPANCY" in codes:
        return "unsupported discrepancy"
    if "EVIDENCE_PENDING_REVIEW" in codes:
        return "evidence pending review"
    if "SUPPLIER_MISMATCH" in codes or "SUPPLIER_UNRESOLVED" in codes or "SUPPLIER_ID_INCONSISTENT" in codes:
        return "supplier unresolved"
    if "LINE_ITEMS_NONCONFORMING" in codes:
        return "line items non-conforming"
    if codes:
        return "not deterministically approvable"
    return ""


def _recompute_packet_hash(packet: ComparisonPacket) -> str:
    """Recompute the canonical packet hash from the packet's *current* contents.

    This is distinct from reading the stored ``packet_hash``: if a consumer mutates
    ``evidence_text`` (or extracted claims / trusted facts) after intake without updating the
    stored hash, the recomputed hash diverges and the mutation is detected. Two stored hash
    strings matching does not prove the contents still match the processed snapshot.
    """
    snapshot = {
        "invoice_id": packet.invoice_id,
        "case_id": packet.case_id,
        "declared_po_reference": packet.case.invoice.po_reference,
        "extracted_claims": packet.extracted_claims,
        "evidence_text": packet.evidence_text,
        "trusted_po_state": packet.trusted_po_state,
        "trusted_po_id": packet.trusted_po_id,
        "trusted_supplier_state": packet.trusted_supplier_state,
        "amount_cents": packet.amount_cents,
    }
    return _sha256(json.dumps(snapshot, sort_keys=True))


def run_operational(
    batch: BatchManifest,
    packets: Dict[str, ComparisonPacket],
    store: EventStore,
    journal_path: str,
    operational_path: Optional[PathResult] = None,
    run_id: str = "s4e16-batch",
    stop_flag: Optional[Dict[str, bool]] = None,
) -> List[dict]:
    """Run the selected operational path (DeepSeek+BOUNDARY) through the deterministic guard +
    journal over the validated packets, and independently verify each receipt before approval.

    A valid selected-path observation (matching model, strategy, run, and case) is REQUIRED
    before any operational write; a missing, failed, or mismatched processing record is
    rejected (``NO_OBSERVATION``) rather than bypassed because the fixture is an exact match.
    A valid approval is displayed only when the commit succeeded AND the receipt independently
    verifies. Shadow paths never write the journal.
    """
    journal: Optional[ActionJournal] = None
    # Index the selected path's COMPLETED observations by case, matching model/strategy/run,
    # plus the packet fingerprint each observation was bound to.
    record_by_case: Dict[str, ExperimentRecord] = {}
    packet_hash_by_case: Dict[str, str] = {}
    if operational_path is not None:
        packet_hash_by_case = operational_path.packet_hash_by_case
        committed_eids = {
            e.payload.get("evidence_id")
            for e in store.committed
            if e.phase == "OBSERVATION"
            and e.run_id == run_id
            and e.path == f"{OPERATIONAL_MODEL}/{OPERATIONAL_STRATEGY}"
        }
        for r in operational_path.records:
            if (r.model_id == OPERATIONAL_MODEL and r.strategy == OPERATIONAL_STRATEGY
                    and r.run_id == run_id and r.state == ObservationState.COMPLETED):
                expected_eid = evidence_id(run_id, OPERATIONAL_MODEL, OPERATIONAL_STRATEGY,
                                          r.repetition, r.case_id)
                if r.evidence_id != expected_eid:
                    # Corrupted / foreign / stale evidence identity: exclude from the valid index,
                    # so the case is rejected below as NO_OBSERVATION (never an operational write).
                    continue
                if r.evidence_id not in committed_eids:
                    # Not durably committed: an observation-persistence failure must never fall
                    # through into an operational approval.
                    continue
                record_by_case[r.case_id] = r
    outcomes: List[dict] = []
    try:
        journal = ActionJournal(journal_path)
        for inv in batch.invoices:
            if stop_flag is not None and stop_flag.get("stop"):
                break
            packet = packets[inv.invoice_id]
            if not packet.valid:
                outcomes.append({
                    "invoice_id": inv.invoice_id, "case_id": inv.case_id,
                    "status": "BLOCKED", "block_reason": packet.block_reason,
                    "guard_action": None, "commit_status": None,
                    "verification_outcome": None, "verified": False,
                    "reason": packet.block_reason, "reason_codes": [],
                    "packet_hash": packet.packet_hash, "path_evidence_id": None,
                    "receipt": None,
                })
                store.append(BatchEvent(
                    event_id=_sha256(f"{run_id}|{inv.invoice_id}|operational|blocked"),
                    batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id,
                    path=f"{OPERATIONAL_MODEL}/{OPERATIONAL_STRATEGY}",
                    phase="BLOCKED", timestamp=_utcnow(),
                    payload={"block_reason": packet.block_reason,
                             "packet_hash": packet.packet_hash},
                ))
                continue
            record = record_by_case.get(inv.case_id)
            if record is not None and (
                    _recompute_packet_hash(packet) != packet.packet_hash
                    or packet_hash_by_case.get(inv.case_id) != packet.packet_hash):
                # Stale/altered/mismatched packet fingerprint: reject, do NOT write the journal.
                outcomes.append({
                    "invoice_id": inv.invoice_id, "case_id": inv.case_id,
                    "status": "MISMATCHED",
                    "block_reason": "selected-path packet fingerprint does not match the comparison packet",
                    "guard_action": None, "commit_status": None,
                    "verification_outcome": None, "verified": False,
                    "reason": "packet fingerprint mismatch", "reason_codes": [],
                    "packet_hash": packet.packet_hash, "path_evidence_id": record.evidence_id,
                    "receipt": None,
                })
                store.append(BatchEvent(
                    event_id=_sha256(f"{run_id}|{inv.invoice_id}|operational|mismatched"),
                    batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id,
                    path=f"{OPERATIONAL_MODEL}/{OPERATIONAL_STRATEGY}",
                    phase="MISMATCHED", timestamp=_utcnow(),
                    payload={"packet_hash": packet.packet_hash,
                             "bound_hash": packet_hash_by_case.get(inv.case_id)},
                ))
                continue
            if record is None:
                # Missing/mismatched selected-path observation: do NOT write the journal.
                outcomes.append({
                    "invoice_id": inv.invoice_id, "case_id": inv.case_id,
                    "status": "NO_OBSERVATION",
                    "block_reason": "no valid selected-path observation",
                    "guard_action": None, "commit_status": None,
                    "verification_outcome": None, "verified": False,
                    "reason": "no valid selected-path observation", "reason_codes": [],
                    "packet_hash": packet.packet_hash, "path_evidence_id": None,
                    "receipt": None,
                })
                store.append(BatchEvent(
                    event_id=_sha256(f"{run_id}|{inv.invoice_id}|operational|no_observation"),
                    batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id,
                    path=f"{OPERATIONAL_MODEL}/{OPERATIONAL_STRATEGY}",
                    phase="NO_OBSERVATION", timestamp=_utcnow(),
                    payload={"packet_hash": packet.packet_hash},
                ))
                continue
            case = packet.case
            guard = evaluate(case)
            commit = journal.approve(case)
            verification = verify(journal, case, receipt=commit.receipt)
            reason_codes = [rc.value for rc in guard.proposal.reason_codes]
            approved = commit.status.value in ("COMMITTED", "IDEMPOTENT")
            verified = verification.outcome.value == "VERIFIED"
            if approved and verified:
                status = "APPROVED"
            elif approved:
                status = "VERIFICATION_FAILED"
            else:
                status = "HELD"
            outcomes.append({
                "invoice_id": inv.invoice_id,
                "case_id": inv.case_id,
                "guard_action": guard.action.value,
                "commit_status": commit.status.value,
                "verification_outcome": verification.outcome.value,
                "verified": verified,
                "status": status,
                "reason": _reason_label(guard.proposal.reason_codes),
                "reason_codes": reason_codes,
                "packet_hash": packet.packet_hash,
                "path_evidence_id": record.evidence_id,
                "receipt": commit.receipt.model_dump() if commit.receipt else None,
            })
            store.append(BatchEvent(
                event_id=_sha256(f"{run_id}|{inv.invoice_id}|operational|guard"),
                batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id,
                path=f"{OPERATIONAL_MODEL}/{OPERATIONAL_STRATEGY}",
                phase="GUARD", timestamp=_utcnow(),
                payload={"guard_action": guard.action.value,
                         "commit_status": commit.status.value,
                         "verification_outcome": verification.outcome.value,
                         "packet_hash": packet.packet_hash},
            ))
    finally:
        if journal is not None:
            journal.close()
    return outcomes


# --- Measurements ---------------------------------------------------------------------

def measure(batch: BatchManifest, paths: List[PathResult], intake_results: Dict[str, ComparisonPacket],
            operational_outcomes: List[dict]) -> dict:
    """Measurement accounting with distinct boundaries (shared intake / operational / shadow / whole)."""
    operational = next(p for p in paths
                       if p.model_id == OPERATIONAL_MODEL and p.strategy == OPERATIONAL_STRATEGY)
    shadow = [p for p in paths if not (p.model_id == OPERATIONAL_MODEL
                                       and p.strategy == OPERATIONAL_STRATEGY)]

    def path_metrics(p: PathResult) -> dict:
        return {
            "model_id": p.model_id, "strategy": p.strategy,
            "made_calls": p.made_calls, "bypassed_calls": p.bypassed_calls,
            "blocked": p.blocked,
            "total_tokens": p.total_tokens,
            "interpretation_checked": p.interpretation_checked,
            "interpretation_correct": p.interpretation_correct,
            "no_call": p.no_call,
            "route_correct": p.route_correct, "route_total": p.route_total,
            "authority_none": p.authority_none,
        }

    whole_made = sum(p.made_calls for p in paths)
    whole_tokens = _sum_tokens(p.total_tokens for p in paths)

    return {
        "shared_intake": {
            "extraction_calls": batch.invoice_count,  # extracted once per invoice
            "invoices": batch.invoice_count,
        },
        "operational_path": path_metrics(operational),
        "shadow_evaluation": [path_metrics(p) for p in shadow],
        "whole_batch": {
            "made_calls": whole_made,
            "bypassed_calls": sum(p.bypassed_calls for p in paths),
            "blocked": sum(p.blocked for p in paths),
            "total_tokens": whole_tokens,
            # Approvals, holds, blocked, and verification failures are distinct.
            "operational_approved": sum(1 for o in operational_outcomes if o["status"] == "APPROVED"),
            "operational_held": sum(1 for o in operational_outcomes if o["status"] == "HELD"),
            "operational_blocked": sum(1 for o in operational_outcomes if o["status"] == "BLOCKED"),
            "operational_verification_failed": sum(
                1 for o in operational_outcomes if o["status"] == "VERIFICATION_FAILED"),
        },
    }


def _reconcile_measurements(measurements: dict, observations: List[dict]) -> dict:
    """Reconcile reported measurements against the authoritative committed observations.

    Every per-path summary (usage AND correctness counters) is derived from the same validated
    observations, so a mutable in-memory cache can never contradict saved evidence. Shared-intake
    activity and operational receipt outcomes retain their separate sources.
    """
    if not observations:
        return measurements
    op = measurements["operational_path"]
    measurements["operational_path"] = _summarize_observations(op["model_id"], op["strategy"], observations)
    for i, m in enumerate(measurements.get("shadow_evaluation", [])):
        measurements["shadow_evaluation"][i] = _summarize_observations(m["model_id"], m["strategy"], observations)
    whole = measurements["whole_batch"]
    model_calls = [o for o in observations if (o.get("model_calls") or 0) > 0]
    no_call = [o for o in observations if (o.get("model_calls") or 0) == 0 and o.get("state") != "FAILED"]
    whole["made_calls"] = len(model_calls)
    whole["bypassed_calls"] = len(no_call)
    whole["blocked"] = sum(1 for o in observations if o.get("state") == "FAILED")
    whole["total_tokens"] = _sum_tokens(o.get("total_tokens") for o in model_calls)
    # operational_* receipt outcomes and shared_intake are NOT overridden (separate sources).
    return measurements


def _summarize_observations(model_id: str, strategy: str, observations: List[dict]) -> dict:
    """The complete per-path summary (usage AND correctness counters) from committed observations.

    This is the single source of truth for ``result.paths``, ``measurements.operational_path``,
    and each ``measurements.shadow_evaluation`` entry. A mutable in-memory ``PathResult`` cache
    can never contradict the saved evidence.
    """
    obs = [o for o in observations if o.get("model_id") == model_id and o.get("strategy") == strategy]
    model_calls = [o for o in obs if (o.get("model_calls") or 0) > 0]
    no_call = [o for o in obs if (o.get("model_calls") or 0) == 0 and o.get("state") != "FAILED"]
    return {
        "model_id": model_id,
        "strategy": strategy,
        "made_calls": len(model_calls),
        "bypassed_calls": len(no_call),
        "blocked": sum(1 for o in obs if o.get("state") == "FAILED"),
        "total_tokens": _sum_tokens(o.get("total_tokens") for o in model_calls),
        "interpretation_checked": len(model_calls),
        "interpretation_correct": sum(1 for o in model_calls if o.get("advisory_consistent") is True),
        "no_call": len(no_call),
        "route_correct": sum(1 for o in obs if o.get("route") is not None and o.get("route_correct") is True),
        "route_total": sum(1 for o in obs if o.get("route") is not None),
        "authority_none": sum(1 for o in obs if o.get("execution_authority") == "NONE"),
    }


# --- Orchestration: progress + stop/cancel + duplicate protection ---------------------

class StopRequested(Exception):
    """The batch was cancelled; further dispatch is prevented and existing results preserved."""


@dataclass
class BatchProgress:
    batch_id: str
    total_invoices: int
    invoices_completed: int = 0
    invoice_status: Dict[str, str] = field(default_factory=dict)
    stopped: bool = False

    def mark(self, invoice_id: str, status: str) -> None:
        self.invoice_status[invoice_id] = status

    def complete(self, invoice_id: str) -> None:
        self.mark(invoice_id, "COMPLETED")
        self.invoices_completed += 1


def _batch_content_hash(batch: BatchManifest) -> str:
    canon = json.dumps(
        sorted((i.case_id, i.content_hash) for i in batch.invoices), sort_keys=True)
    return _sha256(canon)


def _new_run_id(batch_id: str) -> str:
    return f"{batch_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"


def _write_progress(progress_path: str, state: dict) -> None:
    """Atomically persist the current progress so a concurrent reader can observe it mid-run."""
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, progress_path)


def run_batch(
    batch_id: str = "batch-001",
    run_id: Optional[str] = None,
    journal_path: Optional[str] = None,
    event_path: Optional[str] = None,
    result_path: Optional[str] = None,
    progress_path: Optional[str] = None,
    stop_flag: Optional[Dict[str, bool]] = None,
    extract_fn: Optional[Callable[[InvoiceRecord], Callable[[dict], dict]]] = None,
    claims_fn: Optional[Callable] = None,
    selected_invoice_id: Optional[str] = None,
    model_transport_fn: Optional[Callable] = None,
    stop_on_error: bool = False,
    provenance_label: str = "OBSERVED_OFFLINE",
) -> dict:
    """Orchestrate the full bounded batch: intake → four paths → operational → measurement.

    Uses a run-scoped identity (``run_id``); every downstream evidence id is keyed by it, so a
    duplicate resume of the same run is distinguishable from an intentional new repetition.
    ``stop_flag`` requests cancellation before each next extraction/model/operational dispatch;
    completed evidence is preserved in the durable event log and journal. Progress is persisted
    to ``progress_path`` after each stage so the run can be observed queued/partial/finished.
    ``claims_fn`` (when provided) is the injected intake provider (S4E.17 Evolus HTTP bridge);
    otherwise the offline double is used. ``selected_invoice_id`` bounds the batch to a single
    invoice (explicit S4E.17 Part B trial selection).
    """
    run_id = run_id or _new_run_id(batch_id)
    batch = build_batch(batch_id, selected_invoice_id=selected_invoice_id)
    if event_path is None:
        # A durable observation store is always required: default to a unique temp file so the
        # operational path has persisted evidence even in non-durable (simulation) invocations.
        fd, event_path = tempfile.mkstemp(prefix="boundary-batch-events-", suffix=".jsonl")
        os.close(fd)
    store = EventStore(event_path)
    progress = BatchProgress(batch_id=batch_id, total_invoices=batch.invoice_count)
    paths: List[PathResult] = []
    progress_snapshots: List[dict] = []

    def _snapshot(label: str) -> None:
        snap = {
            "label": label,
            "run_id": run_id,
            "batch_id": batch_id,
            "invoices_completed": progress.invoices_completed,
            "total": progress.total_invoices,
            "paths_completed": len(paths),
            "invoice_status": dict(progress.invoice_status),
            "stopped": progress.stopped,
        }
        progress_snapshots.append(snap)
        if progress_path is not None:
            _write_progress(progress_path, snap)

    def _stopped() -> bool:
        return stop_flag is not None and bool(stop_flag.get("stop"))

    _snapshot("queued")

    intake = run_intake(batch, store, extract_fn=extract_fn, claims_fn=claims_fn,
                        stop_flag=stop_flag, run_id=run_id)
    for inv in batch.invoices:
        pkt = intake.get(inv.invoice_id)
        if pkt is None:
            progress.mark(inv.invoice_id, "INTAKE")  # not yet extracted (cancelled)
        else:
            progress.mark(inv.invoice_id, "VALIDATED" if pkt.valid else "BLOCKED")
    _snapshot("intake_complete")

    for model_id in MODELS:
        for strategy in STRATEGIES:
            if _stopped():
                progress.stopped = True
                break
            paths.append(_run_single_path(batch, intake, store, model_id, strategy,
                                          run_id=run_id, stop_flag=stop_flag,
                                          model_transport_fn=model_transport_fn,
                                          stop_on_error=stop_on_error,
                                          provenance_label=provenance_label))
            if len(paths) == 2:
                _snapshot("partial")
        if progress.stopped:
            break

    # Reconcile a cancellation that occurred mid-path or during the final dispatch: the stop
    # flag must propagate to the overall terminal state even when the break happened inside a
    # path's loop rather than at a boundary check.
    if _stopped():
        progress.stopped = True

    # Operational path (DeepSeek+BOUNDARY) through the deterministic guard + journal.
    operational: List[dict] = []
    if not progress.stopped:
        journal_path = journal_path or "boundary.db"
        operational_path = next((p for p in paths
                                 if p.model_id == OPERATIONAL_MODEL
                                 and p.strategy == OPERATIONAL_STRATEGY), None)
        operational = run_operational(batch, intake, store, journal_path,
                                       operational_path=operational_path, run_id=run_id,
                                       stop_flag=stop_flag)
        # Reconcile a cancellation that occurred during operational processing.
        if _stopped():
            progress.stopped = True

    outcome_by_invoice = {o["invoice_id"]: o for o in operational}
    for inv in batch.invoices:
        if inv.invoice_id in outcome_by_invoice:
            progress.mark(inv.invoice_id, outcome_by_invoice[inv.invoice_id]["status"])
            progress.invoices_completed += 1

    # Completion requires every path to have produced a full observation set AND a full
    # operational outcome; a four-element path list or a partial operational outcome is not
    # itself proof that the batch finished.
    complete = (not progress.stopped
                and len(paths) == len(MODELS) * len(STRATEGIES)
                and all((p.made_calls + p.bypassed_calls + p.blocked) == batch.invoice_count
                        for p in paths)
                and len(operational) == batch.invoice_count)

    # Read + validate the committed observations before finalizing the report.
    expected_packet_hashes = {
        i.case_id: (intake[i.invoice_id].packet_hash if i.invoice_id in intake else None)
        for i in batch.invoices
    }
    observation_events, corrupt = (read_durable_observation_events(event_path, run_id=run_id)
                                   if event_path else ([], False))
    durable_observations, issues = _validate_observation_set(
        observation_events, batch, run_id, expected_packet_hashes)

    # Reconcile completion against the durable evidence: a completed run must have its full,
    # valid observation set durably committed. Corrupt, unreadable, missing, foreign, or
    # conflicting observations cannot silently become a trustworthy completed report.
    expected_cells = {(m, s, c) for m in MODELS for s in STRATEGIES
                      for c in {i.case_id for i in batch.invoices}}
    observed_cells = {(o["model_id"], o["strategy"], o["case_id"]) for o in durable_observations}
    missing_cells = sorted(expected_cells - observed_cells)
    if complete and (corrupt or issues or missing_cells):
        complete = False

    _snapshot("completed" if complete else ("cancelled" if progress.stopped else "incomplete"))

    measurements = measure(batch, paths, intake, operational) if complete else None
    if measurements is not None:
        measurements = _reconcile_measurements(measurements, durable_observations)

    # The reported per-path summaries and per-case records come from the committed observations,
    # not the mutable in-memory path caches/records, so they cannot contradict saved evidence.
    projected_paths = [_summarize_observations(p.model_id, p.strategy, durable_observations)
                       for p in paths]
    projected_records = [
        {"model_id": p.model_id, "strategy": p.strategy,
         "observations": [o for o in durable_observations
                          if o.get("model_id") == p.model_id and o.get("strategy") == p.strategy]}
        for p in paths
    ]
    return {
        "batch_id": batch.batch_id,
        "run_id": run_id,
        "content_hash": _batch_content_hash(batch),
        "invoices": [
            {
                "invoice_id": i.invoice_id, "case_id": i.case_id, "family": i.family,
                "content_hash": i.content_hash, "declared_po_ref": i.declared_po_ref,
                "document_version": i.document_version, "document_hash": i.document_hash,
                "trusted_po_state": i.trusted_po_state,
                "trusted_supplier_state": i.trusted_supplier_state,
                "status": progress.invoice_status.get(i.invoice_id, "INTAKE"),
                "blocked": intake.get(i.invoice_id) is not None and not intake[i.invoice_id].valid,
                "block_reason": (intake[i.invoice_id].block_reason
                                 if intake.get(i.invoice_id) is not None else "cancelled before intake"),
                "packet_hash": (intake[i.invoice_id].packet_hash
                                if intake.get(i.invoice_id) is not None else None),
                "reason": (outcome_by_invoice.get(i.invoice_id) or {}).get("reason", ""),
                "guard_action": (outcome_by_invoice.get(i.invoice_id) or {}).get("guard_action"),
                "commit_status": (outcome_by_invoice.get(i.invoice_id) or {}).get("commit_status"),
                "verification_outcome": (outcome_by_invoice.get(i.invoice_id) or {}).get(
                    "verification_outcome"),
            }
            for i in batch.invoices
        ],
        "progress": {
            "total": progress.total_invoices,
            "completed": progress.invoices_completed,
            "stopped": progress.stopped,
        },
        "paths": projected_paths,
        "operational": operational,
        "measurements": measurements,
        "progress_snapshots": progress_snapshots,
        "records": projected_records,
        "durable_observation_count": len(durable_observations),
        "durable_observations": durable_observations,
        "observation_integrity": {
            "corrupt": corrupt,
            "issues": issues,
            "missing_cells": [f"{m}/{s}/{c}" for m, s, c in missing_cells],
            "valid_count": len(durable_observations),
        },
        "event_count": len(store.events),
        "events": [e.to_dict() for e in store.events],
    }


def _events_mention_run(event_path: str, run_id: str) -> bool:
    """True if the durable event log contains any event for the given run id."""
    try:
        with open(event_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("run_id") == run_id:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


def run_batch_durable(
    batch_id: str = "batch-001",
    run_id: Optional[str] = None,
    journal_path: Optional[str] = None,
    event_path: Optional[str] = None,
    result_path: Optional[str] = None,
    progress_path: Optional[str] = None,
    stop_flag: Optional[Dict[str, bool]] = None,
    extract_fn: Optional[Callable[[InvoiceRecord], Callable[[dict], dict]]] = None,
    claims_fn: Optional[Callable] = None,
    selected_invoice_id: Optional[str] = None,
    model_transport_fn: Optional[Callable] = None,
    stop_on_error: bool = False,
    provenance_label: str = "OBSERVED_OFFLINE",
) -> dict:
    """Run the batch and persist the result JSON to ``result_path`` (when provided).

    The result is written atomically (temp file → fsync → rename) so a subsequent reader
    sees either the previous durable result or the new complete one, never a partial file.
    A completed same-run invocation (same explicit ``run_id``) is NOT re-dispatched: the
    persisted result is returned as-is.
    """
    # The durable entry point requires a durable observation store: an absent event path means
    # complete per-response observations can never be persisted, so dispatch must be refused.
    if event_path is None:
        raise ValueError("durable batch requires a durable observation store (event_path)")

    # Run deduplication: any existing durable state for the same run (completed OR interrupted)
    # is returned as a fail-closed refusal; completed or uncertain work is never re-dispatched.
    if run_id is not None:
        if result_path is not None and os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = None
            if existing is not None and existing.get("run_id") == run_id:
                return existing
        # Incomplete same-run state (progress or events persisted, but no completed result) is a
        # fail-closed refusal: never silently re-start a partially completed run from scratch.
        if progress_path is not None and os.path.exists(progress_path):
            try:
                with open(progress_path, encoding="utf-8") as f:
                    prog = json.load(f)
            except Exception:
                prog = None
            if prog is not None and prog.get("run_id") == run_id:
                raise RuntimeError(
                    f"run '{run_id}' has durable progress but no completed result; "
                    "reconcile before re-dispatch")
        if event_path is not None and os.path.exists(event_path):
            if _events_mention_run(event_path, run_id):
                raise RuntimeError(
                    f"run '{run_id}' has durable events but no completed result; "
                    "reconcile before re-dispatch")
    result = run_batch(batch_id=batch_id, run_id=run_id, journal_path=journal_path,
                        event_path=event_path, progress_path=progress_path,
                        stop_flag=stop_flag, extract_fn=extract_fn, claims_fn=claims_fn,
                        selected_invoice_id=selected_invoice_id,
                        model_transport_fn=model_transport_fn,
                        stop_on_error=stop_on_error, provenance_label=provenance_label)
    if result_path is not None:
        tmp = result_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, result_path)
    return result


def _receipt_provenance(label: str, receipt: Optional[str]) -> str:
    """Derive the observation provenance from the transport's receipt fact (a first-class field on
    the result/evidence), not from the semantic outcome or the global mode flag: a live response-less
    connection loss is ``LIVE_UNCONFIRMED``, distinct from a received HTTP error. Offline rehearsals
    stay offline."""
    if label != "OBSERVED_LIVE":
        return label
    if receipt == "unconfirmed":
        return LIVE_UNCONFIRMED
    return label


def _no_call_provenance(label: str) -> str:
    """A live-run blocked/bypassed observation made no model request: label it zero-request, not
    ``OBSERVED_LIVE`` (which would claim live inference occurred). Offline rehearsals stay offline."""
    if label == "OBSERVED_LIVE":
        return LIVE_ZERO_REQUEST
    return label


def _run_single_path(
    batch: BatchManifest,
    packets: Dict[str, ComparisonPacket],
    store: EventStore,
    model_id: str,
    strategy: str,
    run_id: str = "s4e16-batch",
    stop_flag: Optional[Dict[str, bool]] = None,
    model_transport_fn: Optional[Callable] = None,
    stop_on_error: bool = False,
    provenance_label: str = "OBSERVED_OFFLINE",
) -> PathResult:
    """Run one model/strategy path over the validated packets (isolated state).

    ``model_transport_fn(model_id, config) -> transport`` (when provided) replaces the offline
    model double with an injected transport — the S4E.17 Part B live-provider (AMD) wiring
    entry point. The default (``None``) is the offline ``_model_double``. ``stop_on_error`` (the
    S4E.17 trial mode) sets ``stop_flag`` after the first failed/unconfirmed model call so the
    remaining paths are not dispatched. ``provenance_label`` distinguishes an offline rehearsal
    (default ``OBSERVED_OFFLINE``) from an authorized live run (``OBSERVED_LIVE``).
    """
    path = f"{model_id}/{strategy}"
    config = InferenceConfig(
        model_id=model_id,
        code_fence_normalization=(model_id == DESIGNATED_MODEL_GEMMA),
        timeout_seconds=30.0,
    )
    if model_transport_fn is not None:
        transport = model_transport_fn(model_id, config)
    else:
        transport = _model_double(model_id)
    adapter = InferenceAdapter(config, transport)
    records: List[ExperimentRecord] = []
    made = 0
    bypassed = 0
    blocked = 0
    packet_hash_by_case: Dict[str, str] = {}
    for inv in batch.invoices:
        packet = packets[inv.invoice_id]
        request_id = evidence_id(run_id, model_id, strategy, 0, inv.case_id)
        if not packet.valid:
            # Blocked invoice: no model call, no routing, no observation of interpretation.
            blocked += 1
            rec = ExperimentRecord(
                run_id=run_id, model_id=model_id, strategy=strategy, repetition=0,
                case_id=inv.case_id, family=inv.family, split="development",
                state=ObservationState.FAILED, evidence_id=request_id,
                route=None, route_correct=None, model_calls=0,
                execution_authority="NONE", provenance=_no_call_provenance(provenance_label),
            )
            _commit_observation(store, batch, run_id, path, inv, rec, packet.packet_hash)
            records.append(rec)
            store.append(BatchEvent(
                event_id=_sha256(f"{run_id}|{inv.invoice_id}|{path}|blocked"),
                batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path=path,
                phase="BLOCKED", timestamp=_utcnow(),
                payload={"block_reason": packet.block_reason, "packet_hash": packet.packet_hash},
            ))
            continue
        packet_hash_by_case[inv.case_id] = packet.packet_hash
        if stop_flag is not None and stop_flag.get("stop"):
            break
        case = packet.case
        evidence = packet.evidence_text
        if strategy == STRATEGY_BOUNDARY:
            pre = route(case, evidence_text=evidence, adapter=None)
            needs_model = (pre.route == Route.HELD
                           and pre.reason == "inference adapter not configured")
            if not needs_model:
                bypassed += 1
                rec = ExperimentRecord(
                    run_id=run_id, model_id=model_id, strategy=strategy, repetition=0,
                    case_id=inv.case_id, family=inv.family, split="development",
                    state=ObservationState.COMPLETED, evidence_id=request_id,
                    route=pre.route.value,
                    route_correct=(pre.route.value == EXPECTED_ROUTE[inv.family]),
                    model_calls=0, execution_authority="NONE",
                    provenance=_no_call_provenance(provenance_label),
                )
                _commit_observation(store, batch, run_id, path, inv, rec, packet.packet_hash)
                records.append(rec)
                store.append(BatchEvent(
                    event_id=_sha256(f"{run_id}|{inv.invoice_id}|{path}|bypass"),
                    batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path=path,
                    phase="BYPASS", timestamp=_utcnow(),
                    payload={"route": pre.route.value, "model_calls": 0,
                             "packet_hash": packet.packet_hash},
                ))
                continue
            made += 1
            decision = route(case, evidence_text=evidence, adapter=adapter)
            ev = decision.evidence
            rec = ExperimentRecord(
                run_id=run_id, model_id=model_id, strategy=strategy, repetition=0,
                case_id=inv.case_id, family=inv.family, split="development",
                state=ObservationState.COMPLETED, evidence_id=request_id,
                route=decision.route.value,
                route_correct=(decision.route.value == EXPECTED_ROUTE[inv.family]),
                model_calls=ev.model_calls,
                advisory_outcome=(ev.advisory_outcome.value if ev.advisory_outcome else None),
                advisory_consistent=decision.advisory_consistent,
                interpretation=_interpretation_dict(decision.interpretation),
                prompt_tokens=ev.prompt_tokens, completion_tokens=ev.completion_tokens,
                total_tokens=ev.total_tokens, elapsed_ms=ev.elapsed_ms,
                endpoint_model_id=ev.endpoint_model_id,
                model_verified=_classify_model_identity(ev.endpoint_model_id, model_id),
                normalized=ev.normalized,
                execution_authority=decision.execution_authority,
                provenance=_receipt_provenance(provenance_label, ev.receipt),
                diagnostics=ev.diagnostics,
            )
            _commit_observation(store, batch, run_id, path, inv, rec, packet.packet_hash)
            records.append(rec)
            store.append(BatchEvent(
                event_id=_sha256(f"{run_id}|{inv.invoice_id}|{path}|request"),
                batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path=path,
                phase="REQUEST", timestamp=_utcnow(),
                payload={"route": decision.route.value, "model_calls": ev.model_calls,
                         "packet_hash": packet.packet_hash,
                         "outcome": (ev.advisory_outcome.value if ev.advisory_outcome else None)},
            ))
            if stop_on_error and (ev.advisory_outcome is None or ev.advisory_outcome.value != "SUCCESS"
                                  or decision.advisory_consistent is False):
                if stop_flag is not None:
                    stop_flag["stop"] = True
        else:
            made += 1
            result = adapter.interpret(evidence)
            interp = _interpretation_dict(result.interpretation)
            rec = ExperimentRecord(
                run_id=run_id, model_id=model_id, strategy=strategy, repetition=0,
                case_id=inv.case_id, family=inv.family, split="development",
                state=ObservationState.COMPLETED, evidence_id=request_id,
                route=None, route_correct=None, model_calls=1,
                advisory_outcome=result.outcome.value,
                advisory_consistent=(interpretation_is_correct(interp, case)
                                     if interp is not None else None),
                interpretation=interp,
                prompt_tokens=result.telemetry.prompt_tokens,
                completion_tokens=result.telemetry.completion_tokens,
                total_tokens=result.telemetry.total_tokens,
                elapsed_ms=result.telemetry.elapsed_ms,
                endpoint_model_id=result.endpoint_model_id,
                model_verified=_classify_model_identity(result.endpoint_model_id, model_id),
                normalized=result.normalized,
                execution_authority="NONE",
                provenance=_receipt_provenance(provenance_label, result.receipt),
                diagnostics=result.telemetry.diagnostics,
            )
            _commit_observation(store, batch, run_id, path, inv, rec, packet.packet_hash)
            records.append(rec)
            store.append(BatchEvent(
                event_id=_sha256(f"{run_id}|{inv.invoice_id}|{path}|request"),
                batch_id=batch.batch_id, run_id=run_id, invoice_id=inv.invoice_id, path=path,
                phase="REQUEST", timestamp=_utcnow(),
                payload={"model_calls": 1, "packet_hash": packet.packet_hash,
                         "outcome": result.outcome.value},
            ))
            if stop_on_error and (result.outcome.value != "SUCCESS"
                                  or (interp is not None and not interpretation_is_correct(interp, case))):
                if stop_flag is not None:
                    stop_flag["stop"] = True

    verdicts, vsummary = verify_records(build_workload_v2(), records)
    total_tokens = _sum_tokens(r.total_tokens for r in records if (r.model_calls or 0) > 0)
    m = _compute_path_metrics(records)
    return PathResult(
        model_id=model_id, strategy=strategy, records=records,
        made_calls=made, bypassed_calls=bypassed, blocked=blocked,
        total_tokens=total_tokens,
        interpretation_checked=m["interpretation_checked"],
        interpretation_correct=m["interpretation_correct"],
        no_call=m["no_call"],
        route_correct=m["route_correct"], route_total=m["route_total"],
        authority_none=vsummary.get("authority_compliant", 0),
        packet_hash_by_case=packet_hash_by_case,
    )
