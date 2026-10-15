"""S4E.8 — Experimental engine readiness for the reproducible two-model comparative study.

This module supplies the *reproducibility foundation* for a larger 60-case, two-model,
two-strategy evaluation. It builds directly on the accepted contracts, router, and inference
adapter; it introduces no incompatible replacement and performs no network I/O on import.

What it provides
----------------

1. **A versioned synthetic workload manifest.** Sixty synthetic invoice cases (42 development
   + 18 held-out) generated deterministically from the three live-safe case families
   (deterministic match, financial discrepancy, missing purchase order). Every case carries a
   stable case id and a content hash so drift can be detected. The three offline-only
   robustness categories (malformed evidence, prompt injection, unavailable inference) are
   explicitly *not* part of the 60-case efficiency workload.

2. **Explicit run identities.** A ``RunIdentity`` is (model_id, strategy, repetition); the full
   planned matrix is the cross product of models x strategies x repetitions.

3. **Structured per-case result records.** ``ExperimentRecord`` captures timestamp, token usage,
   endpoint model identity, failures, route decision, and authority classification in one
   schema, tagged with an observation state.

4. **Observation state separation.** ``PLANNED`` / ``ATTEMPTED`` / ``COMPLETED`` / ``FAILED`` /
   ``VERIFIED`` — a record cannot silently move from planned to completed.

5. **Fail-closed aggregation.** Aggregation never converts a missing observation into zero or a
   successful outcome; gaps are reported as gaps.

6. **An offline, clearly-labelled rehearsal.** Runs the BOUNDARY deterministic-first strategy
   over the whole workload through a mock adapter (``OBSERVED_OFFLINE``) and records the
   LLM-first baseline as ``COUNTERFACTUAL`` (hypothetical call opportunity, never executed).
   This is a synthetic dry run, not model inference.

7. **A safe offline loader.** ``load_records`` reads recorded experiment JSON and returns
   validated ``ExperimentRecord`` objects, making no network calls — the path the experiment UI
   can later consume without external access.

Provenance discipline (unchanged): offline mock invocations are not real AMD model calls; the
LLM-first baseline is arithmetic only; missing telemetry stays missing (``None``), never zero.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupNotFound,
    POLookupUnavailable,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from .inference import InferenceAdapter, InferenceConfig, InferenceOutcome, BudgetExceeded
from .live_experiment import build_evidence
from .router import Route, route

# --- Provenance labels (mirror the accepted modules) --------------------------------
OBSERVED_OFFLINE = "OBSERVED_OFFLINE"
OBSERVED_LIVE = "OBSERVED_LIVE"
LIVE_ZERO_REQUEST = "LIVE_ZERO_REQUEST"
LIVE_UNCONFIRMED = "LIVE_UNCONFIRMED"
COUNTERFACTUAL = "COUNTERFACTUAL"
NOT_CAPTURED = "NOT_CAPTURED"

# --- Workload identity ---------------------------------------------------------------
WORKLOAD_VERSION = "workload-v1"
DESIGNATED_MODEL_DEEPSEEK = "DeepSeek-V4-Flash"
DESIGNATED_MODEL_GEMMA = "Gemma-4-31B-IT"
MODELS = (DESIGNATED_MODEL_DEEPSEEK, DESIGNATED_MODEL_GEMMA)

# --- Strategies ----------------------------------------------------------------------
STRATEGY_LLM_FIRST = "LLM_FIRST"
STRATEGY_BOUNDARY = "BOUNDARY"
STRATEGIES = (STRATEGY_LLM_FIRST, STRATEGY_BOUNDARY)

# --- Workload sizing (42 dev + 18 held-out = 60, from 3 live-safe families) ----------
LIVE_SAFE_FAMILIES = ("deterministic_match", "financial_discrepancy", "missing_purchase_order")
OFFLINE_ONLY_FAMILIES = ("malformed_evidence", "prompt_injection", "unavailable_inference")

# S4E.10 — six-scenario reconciliation. The six benchmark categories split into two axes:
#   - INVOICE_EXCEPTION families (routing outcomes, 3): deterministic_match,
#     financial_discrepancy, missing_purchase_order. These drive the 60-case workload and the
#     efficiency comparison (LLM-first vs BOUNDARY). Each has an expected ROUTE.
#   - ADAPTER_ROBUSTNESS conditions (response handling, 3): malformed_evidence,
#     prompt_injection, unavailable_inference. These are NOT distinct invoice cases; they are
#     adversarial adapter-response conditions applied to a financial-discrepancy case. Each has an
#     expected ADAPTER OUTCOME (MALFORMED_RESPONSE / AUTHORITY_CLAIM / UNAVAILABLE).
# This is a NEW specification, not a historical reconstruction of the October 6 study.
FAMILY_AXIS = {
    "deterministic_match": "INVOICE_EXCEPTION",
    "financial_discrepancy": "INVOICE_EXCEPTION",
    "missing_purchase_order": "INVOICE_EXCEPTION",
    "malformed_evidence": "ADAPTER_ROBUSTNESS",
    "prompt_injection": "ADAPTER_ROBUSTNESS",
    "unavailable_inference": "ADAPTER_ROBUSTNESS",
}
ROUTING_FAMILIES = LIVE_SAFE_FAMILIES
ADAPTER_ROBUSTNESS_CONDITIONS = OFFLINE_ONLY_FAMILIES

# Expected adapter outcome per robustness condition (applied to a discrepancy case).
EXPECTED_ADAPTER_OUTCOME = {
    "malformed_evidence": "MALFORMED_RESPONSE",
    "prompt_injection": "AUTHORITY_CLAIM",
    "unavailable_inference": "UNAVAILABLE",
}

# Expected route per live-safe family (independent route-correctness verification).
EXPECTED_ROUTE = {
    "deterministic_match": "DETERMINISTIC",
    "financial_discrepancy": "INFERENCE",
    "missing_purchase_order": "HELD",
}
CASES_PER_FAMILY = 20
DEV_PER_FAMILY = 14
HELD_OUT_PER_FAMILY = 6
TOTAL_CASES = CASES_PER_FAMILY * len(LIVE_SAFE_FAMILIES)
DEV_COUNT = DEV_PER_FAMILY * len(LIVE_SAFE_FAMILIES)
HELD_OUT_COUNT = HELD_OUT_PER_FAMILY * len(LIVE_SAFE_FAMILIES)
DEFAULT_REPETITIONS = 3

# Short family code -> readable case-id prefix.
_FAMILY_PREFIX = {
    "deterministic_match": "DET",
    "financial_discrepancy": "DISC",
    "missing_purchase_order": "NOPO",
}


class Split(str, Enum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class ObservationState(str, Enum):
    PLANNED = "PLANNED"
    ATTEMPTED = "ATTEMPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VERIFIED = "VERIFIED"


# --- Deterministic synthetic case generation -----------------------------------------


def _money(cents: int) -> str:
    """Render integer cents as a canonical dollar string (e.g. 125000 -> '1250.00')."""
    return f"{cents // 100}.{cents % 100:02d}"


def _line(number: int, qty: str, price: str) -> LineItem:
    return LineItem(line_number=number, description="synthetic item", quantity=qty, unit_price=price)


def _case(case_id: str, invoice: Invoice, po_lookup, supplier_state=SupplierMatchState.MATCH) -> CaseInput:
    return CaseInput(
        case_id=case_id,
        invoice=invoice,
        po_lookup=po_lookup,
        supplier_lookup=SupplierLookupResult(state=supplier_state),
        evidence=[],
        policy_version="policy-v1",
        input_identity="experiment-engine",
    )


def _deterministic_case(index: int) -> CaseInput:
    """Exact-match invoice/PO with identical amounts (routing: DETERMINISTIC, 0 calls)."""
    qty = str(1 + (index % 5))
    unit_cents = 1000 + index * 100  # $10.00 .. $29.00
    total_cents = int(qty) * unit_cents
    sup = f"SUP-{100 + index}"
    inv_id = f"INV-{100 + index}"
    po_id = f"PO-{100 + index}"
    inv = Invoice(
        invoice_id=inv_id, supplier_id=sup, po_reference=po_id,
        issued_date="2026-10-12", line_items=[_line(1, qty, _money(unit_cents))],
        stated_total=_money(total_cents),
    )
    po = PurchaseOrder(
        po_id=po_id, supplier_id=sup, issued_date="2026-10-12",
        line_items=[_line(1, qty, _money(unit_cents))], stated_total=_money(total_cents),
    )
    return _case(f"{_FAMILY_PREFIX['deterministic_match']}-{index:02d}", inv, POLookupFound(purchase_order=po))


def _discrepancy_case(index: int) -> CaseInput:
    """Invoice exceeds PO by a positive, deterministic discrepancy (routing: INFERENCE)."""
    inv_total_cents = 100000 + index * 5000  # $1000.00 .. $1950.00
    discrepancy_cents = 500 + index * 100   # $5.00 .. $24.00
    po_total_cents = inv_total_cents - discrepancy_cents
    sup = f"SUP-{9001 + index}"
    inv_id = f"INV-{2048 + index}"
    po_id = f"PO-{77 + index}"
    inv = Invoice(
        invoice_id=inv_id, supplier_id=sup, po_reference=po_id,
        issued_date="2026-10-12", line_items=[_line(1, "1", _money(inv_total_cents))],
        stated_total=_money(inv_total_cents),
    )
    po = PurchaseOrder(
        po_id=po_id, supplier_id=sup, issued_date="2026-10-12",
        line_items=[_line(1, "1", _money(po_total_cents))], stated_total=_money(po_total_cents),
    )
    return _case(f"{_FAMILY_PREFIX['financial_discrepancy']}-{index:02d}", inv, POLookupFound(purchase_order=po))


def _missing_po_case(index: int) -> CaseInput:
    """Invoice declares a PO that the trusted lookup does not find (routing: HELD, 0 calls)."""
    qty = str(1 + (index % 3))
    unit_cents = 500 + index * 100  # $5.00 .. $24.00
    total_cents = int(qty) * unit_cents
    sup = f"SUP-{300 + index}"
    inv_id = f"INV-{300 + index}"
    po_id = f"PO-{300 + index}"
    inv = Invoice(
        invoice_id=inv_id, supplier_id=sup, po_reference=po_id,
        issued_date="2026-10-12", line_items=[_line(1, qty, _money(unit_cents))],
        stated_total=_money(total_cents),
    )
    return _case(f"{_FAMILY_PREFIX['missing_purchase_order']}-{index:02d}", inv, POLookupNotFound())


_GENERATORS = {
    "deterministic_match": _deterministic_case,
    "financial_discrepancy": _discrepancy_case,
    "missing_purchase_order": _missing_po_case,
}


# --- Workload manifest ---------------------------------------------------------------


def _canonical_json(case: CaseInput) -> str:
    """Deterministic JSON serialization of a case's facts (for content hashing)."""
    return json.dumps(case.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def content_hash(case: CaseInput) -> str:
    """SHA-256 of a case's canonical content — a stable, drift-detecting fingerprint."""
    return hashlib.sha256(_canonical_json(case).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkloadCase:
    case_id: str
    family: str
    split: Split
    content_hash: str
    case: CaseInput


@dataclass
class WorkloadManifest:
    version: str
    cases: List[WorkloadCase]
    seed: Optional[int] = None

    @property
    def case_ids(self) -> List[str]:
        return [c.case_id for c in self.cases]

    def by_split(self, split: Split) -> List[WorkloadCase]:
        return [c for c in self.cases if c.split == split]

    def verify(self) -> List[str]:
        """Recompute every content hash and report integrity issues (empty == consistent)."""
        issues: List[str] = []
        seen: Dict[str, str] = {}
        for wc in self.cases:
            if wc.case_id in seen:
                issues.append(f"duplicate case id: {wc.case_id!r}")
            seen[wc.case_id] = wc.family
            actual = content_hash(wc.case)
            if actual != wc.content_hash:
                issues.append(f"content hash mismatch for {wc.case_id!r}")
            if wc.case_id != wc.case.case_id:
                issues.append(f"case id mismatch for {wc.case_id!r}")
        return issues


def build_workload() -> WorkloadManifest:
    """Generate the deterministic 60-case workload (42 development + 18 held-out)."""
    cases: List[WorkloadCase] = []
    for family in LIVE_SAFE_FAMILIES:
        gen = _GENERATORS[family]
        for i in range(CASES_PER_FAMILY):
            split = Split.DEVELOPMENT if i < DEV_PER_FAMILY else Split.HELD_OUT
            case = gen(i)
            cases.append(
                WorkloadCase(
                    case_id=case.case_id,
                    family=family,
                    split=split,
                    content_hash=content_hash(case),
                    case=case,
                )
            )
    return WorkloadManifest(version=WORKLOAD_VERSION, cases=cases)


# --- Workload-v2 (S4E.11): seeded, stratified, richer variation ---------------------

WORKLOAD_V2_VERSION = "workload-v2"
WORKLOAD_V2_SEED = 20261017
# Held-out indices per family, SPREAD across 0..19 (NOT clustered at the tail like v1).
V2_HELD_OUT_INDICES = frozenset({2, 5, 8, 11, 14, 17})


def _v2_deterministic(rng: random.Random, index: int) -> CaseInput:
    qty = 1 + rng.randint(0, 7)                    # 1..8
    unit_cents = (1 + rng.randint(0, 199)) * 100   # $1.00..$200.00
    total = qty * unit_cents
    sup = f"SUP-{200 + index}"
    inv_id = f"INV-{200 + index}"
    po_id = f"PO-{200 + index}"
    line = _line(1, str(qty), _money(unit_cents))
    inv = Invoice(invoice_id=inv_id, supplier_id=sup, po_reference=po_id,
                  issued_date="2026-10-12", line_items=[line], stated_total=_money(total))
    po = PurchaseOrder(po_id=po_id, supplier_id=sup, issued_date="2026-10-12",
                       line_items=[line], stated_total=_money(total))
    return _case(f"v2-DET-{index:02d}", inv, POLookupFound(purchase_order=po))


def _v2_discrepancy(rng: random.Random, index: int) -> CaseInput:
    inv_cents = (100 + rng.randint(0, 4900)) * 100  # $100.00..$5000.00
    disc_cents = 50 + rng.randint(0, 49950)          # $0.50..$500.00
    if disc_cents >= inv_cents:
        disc_cents = inv_cents // 2
    po_cents = inv_cents - disc_cents
    sup = f"SUP-{500 + index}"
    inv_id = f"INV-{500 + index}"
    po_id = f"PO-{500 + index}"
    inv = Invoice(invoice_id=inv_id, supplier_id=sup, po_reference=po_id,
                  issued_date="2026-10-12", line_items=[_line(1, "1", _money(inv_cents))],
                  stated_total=_money(inv_cents))
    po = PurchaseOrder(po_id=po_id, supplier_id=sup, issued_date="2026-10-12",
                       line_items=[_line(1, "1", _money(po_cents))], stated_total=_money(po_cents))
    return _case(f"v2-DISC-{index:02d}", inv, POLookupFound(purchase_order=po))


def _v2_missing_po(rng: random.Random, index: int) -> CaseInput:
    qty = 1 + rng.randint(0, 5)                     # 1..6
    unit_cents = (1 + rng.randint(0, 300)) * 100     # $1.00..$301.00
    total = qty * unit_cents
    sup = f"SUP-{600 + index}"
    inv_id = f"INV-{600 + index}"
    po_ref = f"PO-{600 + index}" if index % 5 != 3 else None  # a few with no PO reference
    inv = Invoice(invoice_id=inv_id, supplier_id=sup, po_reference=po_ref,
                  issued_date="2026-10-12", line_items=[_line(1, str(qty), _money(unit_cents))],
                  stated_total=_money(total))
    # Mostly NOT_FOUND; a few UNAVAILABLE (lookup ran but could not return a result).
    po_lookup = POLookupUnavailable() if index % 5 == 4 else POLookupNotFound()
    return _case(f"v2-NOPO-{index:02d}", inv, po_lookup)


_V2_GENERATORS = {
    "deterministic_match": _v2_deterministic,
    "financial_discrepancy": _v2_discrepancy,
    "missing_purchase_order": _v2_missing_po,
}


def build_workload_v2() -> WorkloadManifest:
    """Generate workload-v2: 60 seeded, stratified cases with richer variation.

    - 20 cases per routing family; 14 development + 6 held-out, where held-out indices are
      SPREAD across the range (``{2,5,8,11,14,17}``) rather than clustered at the tail.
    - A fixed seed (``WORKLOAD_V2_SEED``) drives all variation, so the manifest is reproducible.
    - Richer variation: quantities 1-8, amounts $1-$5000, discrepancies $0.50-$500, some
      missing-PO cases with no PO reference or an UNAVAILABLE lookup.
    """
    rng = random.Random(WORKLOAD_V2_SEED)
    cases: List[WorkloadCase] = []
    for family in LIVE_SAFE_FAMILIES:
        gen = _V2_GENERATORS[family]
        for i in range(CASES_PER_FAMILY):
            split = Split.HELD_OUT if i in V2_HELD_OUT_INDICES else Split.DEVELOPMENT
            case = gen(rng, i)
            cases.append(
                WorkloadCase(
                    case_id=case.case_id, family=family, split=split,
                    content_hash=content_hash(case), case=case,
                )
            )
    return WorkloadManifest(version=WORKLOAD_V2_VERSION, seed=WORKLOAD_V2_SEED, cases=cases)


# --- Run identity + planned matrix ---------------------------------------------------


@dataclass(frozen=True)
class RunIdentity:
    model_id: str
    strategy: str
    repetition: int


def build_run_matrix(
    models=MODELS,
    strategies=STRATEGIES,
    repetitions=DEFAULT_REPETITIONS,
) -> List[RunIdentity]:
    """The full model x strategy x repetition matrix of planned run identities."""
    return [
        RunIdentity(model_id=m, strategy=s, repetition=r)
        for m in models
        for s in strategies
        for r in range(repetitions)
    ]


def evidence_id(run_id: str, model_id: str, strategy: str, repetition: int, case_id: str) -> str:
    """A stable, content-addressed evidence id for one observation cell.

    Deterministic across runs given the same inputs, so duplicate protection and
    reconciliation can key on it without trusting any single record's self-reported fields.
    """
    key = f"{run_id}|{model_id}|{strategy}|{repetition}|{case_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# --- Structured result records -------------------------------------------------------


@dataclass
class ExperimentRecord:
    run_id: str
    model_id: str
    strategy: str
    repetition: int
    case_id: str
    family: str
    split: str
    state: ObservationState
    evidence_id: Optional[str] = None
    route: Optional[str] = None
    route_correct: Optional[bool] = None
    model_calls: Optional[int] = None
    advisory_outcome: Optional[str] = None
    advisory_consistent: Optional[bool] = None
    recorded_response: Optional[str] = None
    interpretation: Optional[dict] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    elapsed_ms: Optional[int] = None
    endpoint_model_id: Optional[str] = None
    model_verified: Optional[str] = None
    execution_authority: str = "NONE"
    provenance: str = OBSERVED_OFFLINE
    timestamp_utc: Optional[str] = None
    normalized: Optional[bool] = None  # whether the response was code-fence recovered (or None when unknown)
    diagnostics: Optional[dict] = None  # structured transport diagnostics (category + http_status), never raw bodies


# --- Aggregation (fail-closed) -------------------------------------------------------


def aggregate(records: List[ExperimentRecord], manifest: WorkloadManifest) -> dict:
    """Aggregate records fail-closed, separating planning from execution.

    A PLANNED record (or a COUNTERFACTUAL baseline) is a *planning* artifact, not an execution:
    it must never inflate execution/observation counts, and a cell that has only a planning record
    is still a MISSING execution. Categories reported separately:
      - record coverage (any record at all),
      - planned observations (PLANNED state),
      - offline executions (non-PLANNED, OBSERVED_OFFLINE),
      - live executions (non-PLANNED, live provenance),
      - failed executions (FAILED state),
      - verified results (VERIFIED state).
    """
    execution_states = {
        ObservationState.ATTEMPTED, ObservationState.COMPLETED,
        ObservationState.FAILED, ObservationState.VERIFIED,
    }
    live_provenances = {OBSERVED_LIVE, LIVE_ZERO_REQUEST, LIVE_UNCONFIRMED}

    by_state: Dict[str, int] = {s.value: 0 for s in ObservationState}
    coverage_keys = set()
    planned_keys = set()
    executed_keys = set()
    offline_keys = set()
    live_keys = set()
    failed_keys = set()
    verified_keys = set()
    key_counts: Dict[tuple, int] = {}

    for r in records:
        by_state[r.state.value] += 1
        key = (r.case_id, r.model_id, r.strategy, r.repetition)
        coverage_keys.add(key)
        key_counts[key] = key_counts.get(key, 0) + 1
        if r.state == ObservationState.PLANNED:
            planned_keys.add(key)
        else:
            executed_keys.add(key)
            if r.provenance in live_provenances:
                live_keys.add(key)
            elif r.provenance == OBSERVED_OFFLINE:
                offline_keys.add(key)
            if r.state == ObservationState.FAILED:
                failed_keys.add(key)
            if r.state == ObservationState.VERIFIED:
                verified_keys.add(key)

    duplicate_cells = sorted(k for k, c in key_counts.items() if c > 1)

    # The full planned matrix (model x strategy x repetition x case).
    identities = build_run_matrix()
    matrix_keys = {
        (wc.case_id, ident.model_id, ident.strategy, ident.repetition)
        for wc in manifest.cases
        for ident in identities
    }
    # Missing executions = matrix cells with NO execution record. A planning record alone does
    # not satisfy execution.
    missing_executions = sorted(matrix_keys - executed_keys)

    completed_success = sum(
        1 for r in records
        if r.state == ObservationState.COMPLETED
        and r.advisory_outcome == InferenceOutcome.SUCCESS.value
        and r.advisory_consistent is True
    )
    accepted_authority_violations = sum(
        1 for r in records if r.execution_authority != "NONE"
    )
    route_correct_count = sum(1 for r in records if r.route_correct is True)
    route_incorrect_count = sum(1 for r in records if r.route_correct is False)

    return {
        "workload_version": manifest.version,
        "total_records": len(records),
        "matrix_cells": len(matrix_keys),
        "record_coverage": len(coverage_keys),
        "planned_observations": len(planned_keys),
        "offline_executions": len(offline_keys),
        "live_executions": len(live_keys),
        "failed_executions": len(failed_keys),
        "verified_results": len(verified_keys),
        "missing_executions": len(missing_executions),
        "missing_executions_sample": missing_executions[:20],
        "duplicate_cells": duplicate_cells,
        "by_state": by_state,
        "completed_success": completed_success,
        "accepted_authority_violations": accepted_authority_violations,
        "route_correct": route_correct_count,
        "route_incorrect": route_incorrect_count,
        "note": (
            "missing executions are reported as gaps and are never silently converted to zero "
            "or to a successful outcome; a planning record never counts as an execution"
        ),
    }


# --- Offline rehearsal (clearly labelled synthetic dry run) --------------------------


def _mock_interpretation(case: CaseInput) -> str:
    """A source-consistent interpretation derived from the case's own trusted facts."""
    inv = case.invoice
    if case.po_lookup.state != "FOUND":
        return json.dumps(
            {
                "supplier_id": inv.supplier_id,
                "invoice_id": inv.invoice_id,
                "po_reference": None,
                "amount_cents": inv.computed_total_cents,
                "discrepancy_cents": 0,
                "confidence": 0.5,
                "notes": "no trusted purchase order",
            }
        )
    po = case.po_lookup.purchase_order
    disc = inv.computed_total_cents - po.computed_total_cents
    return json.dumps(
        {
            "supplier_id": inv.supplier_id,
            "invoice_id": inv.invoice_id,
            "po_reference": po.po_id,
            "amount_cents": inv.computed_total_cents,
            "discrepancy_cents": disc,
            "confidence": 0.9,
            "notes": "",
        }
    )


def _mock_transport(case_map: Dict[str, str], model: str):
    def transport(payload: dict) -> dict:
        user_msg = payload["messages"][1]["content"]
        body = case_map.get(user_msg, "{}")
        return {"model": model, "choices": [{"message": {"content": body}}]}

    return transport


def run_offline_rehearsal(
    manifest: WorkloadManifest,
    model_id: str = DESIGNATED_MODEL_DEEPSEEK,
    run_id: str = "rehearsal",
    repetitions: int = DEFAULT_REPETITIONS,
) -> List[ExperimentRecord]:
    """Run the BOUNDARY deterministic-first strategy offline over the whole workload.

    Discrepancy cases exercise the real ``InferenceAdapter`` parser/validator through a mock
    transport (no network, no real model). Every record is ``OBSERVED_OFFLINE``; the LLM-first
    baseline is emitted separately as ``COUNTERFACTUAL`` (hypothetical, never executed). This is
    a synthetic dry run, not model inference.
    """
    records: List[ExperimentRecord] = []

    # A source-consistent mock response keyed by each discrepancy case's evidence text.
    case_map: Dict[str, str] = {}
    for wc in manifest.cases:
        if wc.family == "financial_discrepancy":
            case_map[build_evidence(wc.case)] = _mock_interpretation(wc.case)

    adapter = InferenceAdapter(
        InferenceConfig(model_id=model_id),
        _mock_transport(case_map, model=model_id),
    )

    for wc in manifest.cases:
        for rep in range(repetitions):
            decision = route(wc.case, evidence_text=build_evidence(wc.case), adapter=adapter)
            ev = decision.evidence
            records.append(
                ExperimentRecord(
                    run_id=run_id,
                    model_id=model_id,
                    strategy=STRATEGY_BOUNDARY,
                    repetition=rep,
                    case_id=wc.case_id,
                    family=wc.family,
                    split=wc.split.value,
                    state=ObservationState.COMPLETED,
                    route=decision.route.value,
                    route_correct=decision.route.value == EXPECTED_ROUTE[wc.family],
                    model_calls=ev.model_calls,
                    advisory_outcome=(ev.advisory_outcome.value if ev.advisory_outcome else None),
                    advisory_consistent=decision.advisory_consistent,
                    prompt_tokens=ev.prompt_tokens,
                    completion_tokens=ev.completion_tokens,
                    total_tokens=ev.total_tokens,
                    elapsed_ms=ev.elapsed_ms,
                    endpoint_model_id=model_id,
                    model_verified="verified" if model_id == DESIGNATED_MODEL_DEEPSEEK else "unverified_mismatch",
                    execution_authority=decision.execution_authority,
                    provenance=OBSERVED_OFFLINE,
                )
            )

    # Strategy A — LLM-first baseline: a hypothetical one-call-per-case opportunity, never run.
    for wc in manifest.cases:
        for rep in range(repetitions):
            records.append(
                ExperimentRecord(
                    run_id=run_id,
                    model_id=model_id,
                    strategy=STRATEGY_LLM_FIRST,
                    repetition=rep,
                    case_id=wc.case_id,
                    family=wc.family,
                    split=wc.split.value,
                    state=ObservationState.PLANNED,
                    route=None,
                    model_calls=None,
                    provenance=COUNTERFACTUAL,
                )
            )

    return records


# --- Safe offline loader (for later UI consumption, no network) -----------------------


def load_records(path: str) -> List[ExperimentRecord]:
    """Load recorded experiment JSON into ``ExperimentRecord`` objects.

    Offline: reads an explicit file path only and performs no network I/O. Raises on missing or
    malformed input rather than returning fabricated records.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("record file must be a JSON object with a 'records' list")
    records: List[ExperimentRecord] = []
    for item in data["records"]:
        records.append(
            ExperimentRecord(
                run_id=item["run_id"],
                model_id=item["model_id"],
                strategy=item["strategy"],
                repetition=item["repetition"],
                case_id=item["case_id"],
                family=item["family"],
                split=item["split"],
                state=ObservationState(item["state"]),
                evidence_id=item.get("evidence_id"),
                route=item.get("route"),
                route_correct=item.get("route_correct"),
                model_calls=item.get("model_calls"),
                advisory_outcome=item.get("advisory_outcome"),
                advisory_consistent=item.get("advisory_consistent"),
                recorded_response=item.get("recorded_response"),
                interpretation=item.get("interpretation"),
                prompt_tokens=item.get("prompt_tokens"),
                completion_tokens=item.get("completion_tokens"),
                total_tokens=item.get("total_tokens"),
                elapsed_ms=item.get("elapsed_ms"),
                endpoint_model_id=item.get("endpoint_model_id"),
                model_verified=item.get("model_verified"),
                execution_authority=item.get("execution_authority", "NONE"),
                provenance=item.get("provenance", OBSERVED_OFFLINE),
                timestamp_utc=item.get("timestamp_utc"),
            )
        )
    return records


def serialize_records(records: List[ExperimentRecord], manifest: WorkloadManifest) -> dict:
    """Serialize records to a stable JSON envelope (for writing or later re-loading)."""
    return {
        "workload_version": manifest.version,
        "records": [asdict(r) for r in records],
    }


# --- Gemma second-model readiness: diagnostic classification (offline) --------------

MAX_RAW_PREVIEW_CHARS = 512

# Credential-like and sensitive patterns a response body could echo back from the prompt.
_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|authorization|token|bearer)\b\s*[:=]\s*\S+"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
]


def redact_content(text: str) -> str:
    """Redact credential-like and sensitive material from captured content.

    A model response body can itself echo confidential material (e.g. a credential that
    appeared in the prompt), so diagnostic capture must REDACT, not merely truncate. Returns the
    text with credential-like patterns replaced by ``[REDACTED]``.
    """
    out = text
    for pat in _CREDENTIAL_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def _bounded_redact(content: Optional[str]) -> Optional[str]:
    """Redact and bound captured content for safe retention (diagnostic only)."""
    if content is None:
        return None
    return redact_content(content)[:MAX_RAW_PREVIEW_CHARS]


def extract_raw_content_preview(raw, max_chars: int = MAX_RAW_PREVIEW_CHARS) -> Optional[str]:
    """Extract a bounded, credential-free preview of a raw model response's content.

    Returns the model output text (``choices[0].message.content``) truncated to ``max_chars``,
    or ``None`` if the response has no such content. The API credential is never present in the
    response body (it is sent in the request Authorization header, not echoed back), so the
    response content is safe to capture. The preview is diagnostic-only and never feeds back into
    routing or authority decisions. Pair with :func:`diagnose_raw_response` to classify WHY a
    response was malformed.
    """
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return redact_content(content)[:max_chars]


def _extract_code_fence(content: str) -> Optional[str]:
    """Return the inner text of a leading ```json ... ``` fence, or ``None``."""
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", content, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_json_object(content: str) -> Optional[str]:
    """Return the first balanced top-level JSON object/array embedded in prose, or ``None``."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = content.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(content)):
            ch = content[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return content[start:i + 1]
    return None


def diagnose_raw_response(content: Optional[str]) -> dict:
    """Classify WHY a raw model response failed to parse (diagnostic, offline).

    Does not loosen the accepted adapter contract; it only helps a future live run diagnose
    a malformed model response (e.g. Gemma's S4E.7 ``MALFORMED_RESPONSE``). Returns a dict with
    a ``failure_mode`` and, when a JSON body is recoverable, the ``extracted`` text (marked
    clearly as diagnostic-only).
    """
    if content is None:
        return {"failure_mode": "no_content", "note": "no response content"}
    stripped = content.strip()
    if not stripped:
        return {"failure_mode": "empty", "note": "response content is empty or blank"}
    try:
        json.loads(stripped)
        return {"failure_mode": "parseable",
                "note": "content is valid JSON (any failure is downstream of parsing)"}
    except (ValueError, TypeError):
        pass
    fenced = _extract_code_fence(stripped)
    if fenced is not None:
        return {"failure_mode": "code_fenced", "extracted": fenced,
                "note": "valid JSON wrapped in a markdown code fence (recoverable for diagnosis)"}
    obj = _extract_json_object(stripped)
    if obj is not None:
        return {"failure_mode": "prose_wrapped", "extracted": obj,
                "note": "JSON object embedded in surrounding prose (recoverable for diagnosis)"}
    return {"failure_mode": "not_json", "note": "no parseable JSON object found"}


# --- S4E.10: model-adapter interfaces (mock, offline) --------------------------------

DEEPSEEK_ENDPOINT_ID = "self-dploy/DeepSeek-V4-Flash"  # S4E.5 recorded endpoint id
GEMMA_ENDPOINT_ID = "self-dploy/Gemma-4-31B-IT"        # S4E.7 recorded endpoint id

# Mock token usage reflecting the recorded small samples (MOCK, not live measurements).
DEEPSEEK_MOCK_USAGE = {"prompt_tokens": 120, "completion_tokens": 99, "total_tokens": 219}
GEMMA_MOCK_USAGE = {"prompt_tokens": 152, "completion_tokens": 101, "total_tokens": 253}


def _deepseek_transport(case_map: Dict[str, str]):
    def transport(payload: dict) -> dict:
        user_msg = payload["messages"][1]["content"]
        body = case_map.get(user_msg, "{}")
        return {
            "model": DEEPSEEK_ENDPOINT_ID,
            "choices": [{"message": {"content": body}}],
            "usage": DEEPSEEK_MOCK_USAGE,
        }
    return transport


def _gemma_transport():
    def transport(payload: dict) -> dict:
        return {
            "model": GEMMA_ENDPOINT_ID,
            "choices": [{"message": {"content": "this is not valid json {"}}],
            "usage": GEMMA_MOCK_USAGE,
        }
    return transport


class _RecordingTransport:
    """Wraps a mock transport and records the raw response content + endpoint model id."""

    def __init__(self, transport):
        self._transport = transport
        self.last_content: Optional[str] = None
        self.last_endpoint_model: Optional[str] = None

    def __call__(self, payload: dict) -> dict:
        raw = self._transport(payload)
        self.last_endpoint_model = raw.get("model") if isinstance(raw, dict) else None
        self.last_content = None
        if isinstance(raw, dict):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    self.last_content = msg["content"]
        return raw


def _build_mock_adapter(manifest: WorkloadManifest, model_id: str):
    """Build a mock adapter for a model plus its recording transport (offline, no network)."""
    if model_id == DESIGNATED_MODEL_DEEPSEEK:
        case_map = {build_evidence(wc.case): _mock_interpretation(wc.case) for wc in manifest.cases}
        inner = _deepseek_transport(case_map)
    else:
        inner = _gemma_transport()
    recording = _RecordingTransport(inner)
    adapter = InferenceAdapter(InferenceConfig(model_id=model_id), recording)
    return adapter, recording


def _classify_model_identity(endpoint_id: Optional[str], configured_id: str) -> str:
    if endpoint_id is None:
        return "unverified_missing"
    if endpoint_id != configured_id:
        return "unverified_mismatch"
    return "verified"


# --- S4E.10: hard budget enforcement (request + token, wired at the transport boundary) ---


@dataclass
class RunBudgets:
    """Hard run-level budgets: max requests and max cumulative tokens, with stop conditions."""
    request_limit: int
    token_limit: int
    requests: int = 0
    tokens: int = 0
    rejected_requests: int = 0

    def acquire_request(self) -> bool:
        if self.requests >= self.request_limit:
            self.rejected_requests += 1
            return False
        self.requests += 1
        return True

    def charge_tokens(self, total_tokens: int) -> bool:
        if self.tokens + total_tokens > self.token_limit:
            return False
        self.tokens += total_tokens
        return True

    def stop_reason(self) -> Optional[str]:
        if self.requests >= self.request_limit:
            return "request_budget_exhausted"
        if self.tokens >= self.token_limit:
            return "token_budget_exhausted"
        return None


class BudgetedTransport:
    """Wraps a transport and enforces hard request + token budgets at the outbound boundary.

    Exactly one attempt per request (no retry); exceeding either budget raises
    ``BudgetExceeded`` so the caller stops with an explicit stop condition. Timeout is enforced by
    the underlying live HTTP transport via ``InferenceConfig.timeout_seconds`` (not re-implemented
    here). This is the transport-level enforcement a live runner wires in; ``RunBudgets`` alone
    is not enforcement.
    """

    def __init__(self, transport, budgets: RunBudgets):
        self._transport = transport
        self._budgets = budgets

    def __call__(self, payload: dict) -> dict:
        if not self._budgets.acquire_request():
            raise BudgetExceeded(self._budgets.stop_reason() or "request budget exhausted")
        raw = self._transport(payload)
        if isinstance(raw, dict):
            usage = raw.get("usage")
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
                if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                    if not self._budgets.charge_tokens(total):
                        raise BudgetExceeded(self._budgets.stop_reason() or "token budget exhausted")
        return raw


# --- S4E.10: independent verifier ----------------------------------------------------


def expected_interpretation(case: CaseInput) -> dict:
    """The CORRECT advisory interpretation derived independently from trusted source facts."""
    inv = case.invoice
    if case.po_lookup.state != "FOUND":
        return {
            "supplier_id": inv.supplier_id,
            "invoice_id": inv.invoice_id,
            "amount_cents": inv.computed_total_cents,
            "discrepancy_cents": 0,
        }
    po = case.po_lookup.purchase_order
    return {
        "supplier_id": inv.supplier_id,
        "invoice_id": inv.invoice_id,
        "po_reference": po.po_id,
        "amount_cents": inv.computed_total_cents,
        "discrepancy_cents": inv.computed_total_cents - po.computed_total_cents,
    }


def interpretation_is_correct(interpretation: dict, case: CaseInput) -> bool:
    """True iff the recorded interpretation matches the independently expected interpretation.

    A structurally valid JSON response is NOT automatically correct: its supplier, invoice,
    amount, and discrepancy must all match the trusted source facts.
    """
    exp = expected_interpretation(case)
    return (
        interpretation.get("supplier_id") == exp["supplier_id"]
        and interpretation.get("invoice_id") == exp["invoice_id"]
        and interpretation.get("amount_cents") == exp["amount_cents"]
        and interpretation.get("discrepancy_cents") == exp["discrepancy_cents"]
    )


def _interpretation_dict(interp) -> Optional[dict]:
    """Serialize a parsed ``EvidenceInterpretation`` to a JSON-safe dict, or ``None``."""
    if interp is None:
        return None
    return {
        "supplier_id": interp.supplier_id,
        "invoice_id": interp.invoice_id,
        "po_reference": interp.po_reference,
        "amount_cents": interp.amount_cents,
        "discrepancy_cents": interp.discrepancy_cents,
    }


def independent_expected_route(case: CaseInput, evidence_text: Optional[str]) -> str:
    """Re-derive the expected route from case FACTS, independent of the router/classifier.

    Re-implements the routing decision from first principles (lookup state, supplier identity,
    structural defects, exact match, discrepancy, evidence presence) so a verifier does not
    simply trust the router's own ``route()`` output.
    """
    inv = case.invoice
    if case.po_lookup.state != "FOUND":
        return "HELD"
    if case.supplier_lookup.state != "MATCH":
        return "HELD"
    po = case.po_lookup.purchase_order
    if inv.stated_total_cents != inv.computed_total_cents:
        return "HELD"
    if po.stated_total_cents != po.computed_total_cents:
        return "HELD"
    inv_lines = [li.line_number for li in inv.line_items]
    po_lines = [li.line_number for li in po.line_items]
    if len(inv_lines) != len(set(inv_lines)) or len(po_lines) != len(set(po_lines)):
        return "HELD"
    if inv.computed_total_cents == po.computed_total_cents:
        return "DETERMINISTIC"
    if inv.supplier_id != po.supplier_id:
        return "HELD"
    if inv.po_reference is not None and inv.po_reference != po.po_id:
        return "HELD"
    if evidence_text and evidence_text.strip():
        return "INFERENCE"
    return "HELD"


@dataclass
class Verdict:
    evidence_id: str
    case_id: str
    model_id: str
    strategy: str
    expected_route: str
    recorded_route: Optional[str]
    integrity_ok: bool             # record fidelity: calls/tokens/evidence-id
    routing_correct: bool          # route == expected
    interpretation_correct: bool   # interpretation == expected facts
    authority_compliant: bool      # execution_authority == NONE
    task_success: bool             # routing + interpretation + authority
    issues: List[str]


def _routing_correct(record: ExperimentRecord, expected_route: str) -> bool:
    if record.strategy != STRATEGY_BOUNDARY:
        return True  # LLM-first has no route decision
    return record.route == expected_route


def _interpretation_correct(record: ExperimentRecord, case: CaseInput, expected_route: str) -> bool:
    if record.strategy == STRATEGY_BOUNDARY and expected_route in ("DETERMINISTIC", "HELD"):
        return True  # no model interpretation is required for deterministic/held resolution
    if record.interpretation is None:
        return False
    return interpretation_is_correct(record.interpretation, case)


def verify_observation(case: CaseInput, record: ExperimentRecord, expected_eid: str) -> Verdict:
    """Independently verify one observation across five axes:
    integrity, routing correctness, interpretation correctness, authority compliance,
    and (composite) task success."""
    issues: List[str] = []
    expected_route = independent_expected_route(case, build_evidence(case))

    routing_correct = _routing_correct(record, expected_route)
    if not routing_correct:
        issues.append(f"route {record.route!r} != expected {expected_route!r}")

    authority_compliant = record.execution_authority == "NONE"
    if not authority_compliant:
        issues.append(f"execution_authority {record.execution_authority!r} != NONE")

    interpretation_correct = _interpretation_correct(record, case, expected_route)
    if not interpretation_correct:
        issues.append("interpretation incorrect or absent for a case requiring one")

    # Integrity: call-count fidelity, token telemetry consistency, evidence-id correctness.
    calls_ok = (
        record.model_calls == (1 if expected_route == "INFERENCE" else 0)
        if record.strategy == STRATEGY_BOUNDARY
        else record.model_calls == 1
    )
    if not calls_ok:
        issues.append(f"model_calls {record.model_calls!r} unexpected")

    tokens_ok = True
    if record.total_tokens is not None:
        if record.total_tokens < 0:
            tokens_ok = False
            issues.append("negative total_tokens")
        elif (record.prompt_tokens is not None and record.completion_tokens is not None
              and record.total_tokens != record.prompt_tokens + record.completion_tokens):
            tokens_ok = False
            issues.append("total_tokens != prompt_tokens + completion_tokens")

    evidence_id_ok = record.evidence_id == expected_eid
    if not evidence_id_ok:
        issues.append(f"evidence_id {record.evidence_id!r} != expected {expected_eid!r}")

    integrity_ok = calls_ok and tokens_ok and evidence_id_ok
    task_success = integrity_ok and routing_correct and interpretation_correct and authority_compliant

    return Verdict(
        evidence_id=record.evidence_id or "",
        case_id=record.case_id, model_id=record.model_id, strategy=record.strategy,
        expected_route=expected_route, recorded_route=record.route,
        integrity_ok=integrity_ok,
        routing_correct=routing_correct,
        interpretation_correct=interpretation_correct,
        authority_compliant=authority_compliant,
        task_success=task_success,
        issues=issues,
    )


def verify_records(manifest: WorkloadManifest, records: List[ExperimentRecord]):
    """Independently verify every record, reporting five axes separately."""
    case_by_id = {wc.case.case_id: wc.case for wc in manifest.cases}
    verdicts: List[Verdict] = []
    for r in records:
        case = case_by_id.get(r.case_id)
        if case is None:
            verdicts.append(Verdict(
                evidence_id=r.evidence_id or "", case_id=r.case_id, model_id=r.model_id,
                strategy=r.strategy, expected_route="UNKNOWN", recorded_route=r.route,
                integrity_ok=False, routing_correct=False, interpretation_correct=False,
                authority_compliant=False, task_success=False, issues=["unknown case_id"],
            ))
            continue
        expected_eid = evidence_id(r.run_id, r.model_id, r.strategy, r.repetition, r.case_id)
        verdicts.append(verify_observation(case, r, expected_eid))

    summary = {
        "total": len(verdicts),
        "integrity_passed": sum(1 for v in verdicts if v.integrity_ok),
        "routing_correct": sum(1 for v in verdicts if v.routing_correct),
        "interpretation_correct": sum(1 for v in verdicts if v.interpretation_correct),
        "authority_compliant": sum(1 for v in verdicts if v.authority_compliant),
        "task_success": sum(1 for v in verdicts if v.task_success),
    }
    return verdicts, summary


# --- S4E.10: full comparative matrix runner (measured LLM-first + BOUNDARY, both models) ---


def run_matrix_rehearsal(
    manifest: WorkloadManifest,
    repetitions: int = DEFAULT_REPETITIONS,
    run_id: str = "matrix-rehearsal",
    diagnostic: bool = False,
) -> List[ExperimentRecord]:
    """Run the full matrix offline: 2 models x 2 strategies x 3 reps x 60 cases.

    LLM-first is MEASURED here (a mock call for every case), not counterfactual; BOUNDARY routes
    per the accepted contract (mock call only for discrepancy). DeepSeek uses a source-consistent
    mock (S4E.5 behavior); Gemma uses a malformed mock (S4E.7 behavior). Everything is
    ``OBSERVED_OFFLINE`` / MOCK — no model inference occurred.

    ``diagnostic`` is ``False`` by default (metadata-only: outcome + telemetry + endpoint id, no
    raw response text). Setting ``diagnostic=True`` captures a bounded, redacted response preview
    (``recorded_response``) for failure diagnosis and requires explicit authorization.
    """
    records: List[ExperimentRecord] = []
    for model_id in (DESIGNATED_MODEL_DEEPSEEK, DESIGNATED_MODEL_GEMMA):
        for strategy in STRATEGIES:
            for rep in range(repetitions):
                adapter, capture = _build_mock_adapter(manifest, model_id)
                for wc in manifest.cases:
                    eid = evidence_id(run_id, model_id, strategy, rep, wc.case_id)
                    if strategy == STRATEGY_BOUNDARY:
                        decision = route(wc.case, evidence_text=build_evidence(wc.case), adapter=adapter)
                        ev = decision.evidence
                        records.append(ExperimentRecord(
                            run_id=run_id, model_id=model_id, strategy=strategy, repetition=rep,
                            case_id=wc.case_id, family=wc.family, split=wc.split.value,
                            state=ObservationState.COMPLETED, evidence_id=eid,
                            route=decision.route.value,
                            route_correct=decision.route.value == EXPECTED_ROUTE[wc.family],
                            model_calls=ev.model_calls,
                            advisory_outcome=(ev.advisory_outcome.value if ev.advisory_outcome else None),
                            advisory_consistent=decision.advisory_consistent,
                            recorded_response=(_bounded_redact(capture.last_content) if diagnostic and ev.model_calls > 0 else None),
                            interpretation=_interpretation_dict(decision.interpretation),
                            prompt_tokens=ev.prompt_tokens,
                            completion_tokens=ev.completion_tokens,
                            total_tokens=ev.total_tokens,
                            elapsed_ms=ev.elapsed_ms,
                            endpoint_model_id=capture.last_endpoint_model,
                            model_verified=_classify_model_identity(capture.last_endpoint_model, model_id),
                            execution_authority=decision.execution_authority,
                            provenance=OBSERVED_OFFLINE,
                        ))
                    else:
                        result = adapter.interpret(build_evidence(wc.case))
                        records.append(ExperimentRecord(
                            run_id=run_id, model_id=model_id, strategy=strategy, repetition=rep,
                            case_id=wc.case_id, family=wc.family, split=wc.split.value,
                            state=ObservationState.COMPLETED, evidence_id=eid,
                            route=None, route_correct=None,
                            model_calls=1,
                            advisory_outcome=result.outcome.value,
                            advisory_consistent=None,
                            recorded_response=(_bounded_redact(capture.last_content) if diagnostic else None),
                            interpretation=_interpretation_dict(result.interpretation),
                            prompt_tokens=result.telemetry.prompt_tokens,
                            completion_tokens=result.telemetry.completion_tokens,
                            total_tokens=result.telemetry.total_tokens,
                            elapsed_ms=result.telemetry.elapsed_ms,
                            endpoint_model_id=capture.last_endpoint_model,
                            model_verified=_classify_model_identity(capture.last_endpoint_model, model_id),
                            execution_authority="NONE",
                            provenance=OBSERVED_OFFLINE,
                        ))
    return records


# --- CLI -----------------------------------------------------------------------------


def render_matrix_report(manifest: WorkloadManifest, summary: dict, verification: dict) -> str:
    """Render a human-readable matrix execution report (all MOCK/OFFLINE)."""
    lines = [
        "# BOUNDARY — S4E.10 Comparative Matrix Execution Report",
        "",
        "**Classification: MOCK / OFFLINE.** No model inference occurred; every call was a mock "
        "transport (DeepSeek source-consistent, Gemma malformed per recorded S4E.7).",
        "",
        "## Workload",
        f"- Version `{manifest.version}`: {len(manifest.cases)} cases "
        f"({DEV_COUNT} development / {HELD_OUT_COUNT} held-out).",
        f"- Routing families (invoice exceptions): {', '.join(ROUTING_FAMILIES)}.",
        f"- Adapter robustness conditions (NOT cases): {', '.join(ADAPTER_ROBUSTNESS_CONDITIONS)}.",
        "",
        "## Execution",
        f"- Matrix cells: {summary['matrix_cells']} (2 models x 2 strategies x 3 reps x 60 cases).",
        f"- Record coverage: {summary['record_coverage']}.",
        f"- Planned observations: {summary['planned_observations']}.",
        f"- Offline executions (MOCK/OBSERVED_OFFLINE): {summary['offline_executions']}.",
        f"- Live executions: {summary['live_executions']} (expected 0 in this task).",
        f"- Failed executions: {summary['failed_executions']}.",
        f"- Verified results: {summary['verified_results']}.",
        f"- Missing executions: {summary['missing_executions']}.",
        f"- Duplicate cells: {len(summary['duplicate_cells'])}.",
        f"- Authority violations: {summary['accepted_authority_violations']}.",
        "",
        "## Independent verification (five axes)",
        f"- Integrity (record fidelity): {verification['integrity_passed']} / {verification['total']}.",
        f"- Routing correctness: {verification['routing_correct']} / {verification['total']}.",
        f"- Interpretation correctness: {verification['interpretation_correct']} / {verification['total']}.",
        f"- Authority compliance: {verification['authority_compliant']} / {verification['total']}.",
        f"- Task success (composite): {verification['task_success']} / {verification['total']}.",
        "Note: a fail-closed malformed response (Gemma) passes integrity and authority but fails "
        "interpretation correctness and therefore task success; structural passes are never "
        "presented as successful model interpretations.",
        "",
        "## Six-family reconciliation (REQUIRES REVIEW)",
        "The six benchmark categories split into three invoice-exception routing families and three "
        "adapter-robustness response conditions. This is a NEW research-design specification, not a "
        "historical reconstruction of the October 6 study; it requires owner review before a live run.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BOUNDARY S4E.10 comparative matrix runner")
    parser.add_argument(
        "--json",
        default="matrix_execution_results.json",
        help="write the matrix execution results (JSON) here",
    )
    parser.add_argument(
        "--manifest",
        default="experiment_workload.json",
        help="write the generated workload manifest (JSON) here",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write a human-readable matrix execution report (Markdown) here",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="capture bounded, redacted raw-response previews for failure diagnosis "
             "(metadata-only by default; requires explicit authorization)",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="workload version to run (default: v2)",
    )
    args = parser.parse_args(argv)

    manifest = build_workload_v2() if args.version == "v2" else build_workload()
    issues = manifest.verify()
    if issues:
        print("Workload manifest integrity failure:")
        for i in issues:
            print(f"  - {i}")
        return 1

    records = run_matrix_rehearsal(manifest, diagnostic=args.diagnostic)
    summary = aggregate(records, manifest)
    verdicts, verification = verify_records(manifest, records)

    manifest_json = {
        "version": manifest.version,
        "seed": manifest.seed,
        "family_axis": FAMILY_AXIS,
        "routing_families": list(ROUTING_FAMILIES),
        "adapter_robustness_conditions": list(ADAPTER_ROBUSTNESS_CONDITIONS),
        "cases": [
            {
                "case_id": wc.case_id,
                "family": wc.family,
                "split": wc.split.value,
                "content_hash": wc.content_hash,
            }
            for wc in manifest.cases
        ],
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_json, f, indent=2)

    results = serialize_records(records, manifest)
    results["summary"] = summary
    results["independent_verification"] = verification
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_matrix_report(manifest, summary, verification) + "\n")

    print(f"Workload: {len(manifest.cases)} cases ({DEV_COUNT} dev / {HELD_OUT_COUNT} held-out), "
          f"version {manifest.version}, manifest consistent.")
    print(f"Matrix: {summary['matrix_cells']} cells, {summary['offline_executions']} offline "
          f"executions, {summary['missing_executions']} missing.")
    print(f"Independent verification — integrity {verification['integrity_passed']}/{verification['total']}, "
          f"routing {verification['routing_correct']}/{verification['total']}, "
          f"interpretation {verification['interpretation_correct']}/{verification['total']}, "
          f"authority {verification['authority_compliant']}/{verification['total']}, "
          f"task success {verification['task_success']}/{verification['total']}.")
    print(f"Authority violations: {summary['accepted_authority_violations']}.")
    print("MOCK/OFFLINE only — no model inference occurred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
