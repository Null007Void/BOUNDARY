"""B5.2 live benchmark readiness runner for BOUNDARY.

Reuses the accepted AMD inference adapter (B3.2) and deterministic-first routing
(B3.3) to prepare three live scenarios — exact match, financial discrepancy, and
missing purchase order — for an *opt-in*, single-shot live measurement.

Two outcomes are reported SEPARATELY so a correct fail-closed handling is never
mistaken for a successful inference:

- ``harness_safe`` — the harness behaved safely: it routed per the accepted
  contract, made at most one bounded request, and preserved ``execution_authority
  == "NONE"``. A failed inference (auth failure, malformed output, contradictory
  interpretation, authority claim) is still ``harness_safe=True`` because it failed
  closed correctly.
- ``measurement_success`` — the scenario actually produced a *successful,
  source-consistent* advisory inference. For the financial-discrepancy scenario
  this requires ``advisory_outcome == SUCCESS`` AND ``advisory_consistent is True``.
  Authentication failures, malformed outputs, contradictory evidence, and authority
  claims are never ``measurement_success``.

Provenance is established POSITIVELY and fail-closed at the transport boundary.
Only the approved real HTTP transport (``httpx.HTTPTransport``) is treated as live;
every other transport — ``httpx.MockTransport``, ``httpx.WSGITransport``, any custom
or unrecognized injected transport, or an absent transport — is offline. No
caller-provided label can establish live provenance.

- ``OBSERVED_LIVE`` — a confirmed HTTP response from the AMD endpoint was observed
  over a real transport.
- ``LIVE_UNCONFIRMED`` — a real transport attempted a request but no confirmed AMD
  response was received (timeout, or an unavailable/ambiguous outcome).
- ``LIVE_ZERO_REQUEST`` — a real live transport was configured but the scenario made
  zero requests (deterministic/held).
- ``OBSERVED_OFFLINE`` — a mock/WSGI/custom transport or a dry-run preflight.

``confirmed_response`` distinguishes an outbound request *attempt* (the invocation
counter) from a *confirmed* server response. Merely constructing a client or
incrementing a counter is not proof that AMD responded.

Safety properties:

- Importing this module performs no network calls.
- The default invocation is a harmless dry run (``adapter=None``; no transport).
- A live AMD inference happens only when ``--live`` is passed, for exactly one
  selected scenario, with at most one outbound request (``SingleRequestGuard``
  rejects a second attempt; no automatic retries).

Telemetry provenance: ``None`` token counts mean the endpoint reported no usage
(missing), ``0`` means zero consumption. These are never conflated.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import httpx

from .amd_transport import build_amd_transport
from .contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    POLookupNotFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from .inference import InferenceAdapter, InferenceConfig, InferenceOutcome, InferenceUnavailable
from .router import Route, route
from .smoke_test import SYNTHETIC_EVIDENCE

# Provenance labels.
OBSERVED_LIVE = "OBSERVED_LIVE"
OBSERVED_OFFLINE = "OBSERVED_OFFLINE"
LIVE_ZERO_REQUEST = "LIVE_ZERO_REQUEST"
LIVE_UNCONFIRMED = "LIVE_UNCONFIRMED"
NOT_CAPTURED = "NOT_CAPTURED"

# Transport kinds (derived positively at the transport boundary, never caller-declared).
TRANSPORT_LIVE = "live"
TRANSPORT_OFFLINE = "offline"
TRANSPORT_NONE = "none"

# Adapter outcomes that indicate the endpoint sent a confirmed HTTP response.
# TIMEOUT and UNAVAILABLE are excluded (no response, or ambiguous), as is
# INPUT_TOO_LARGE (no request was made).
_CONFIRMED_RESPONSE_OUTCOMES = {
    InferenceOutcome.SUCCESS,
    InferenceOutcome.MALFORMED_RESPONSE,
    InferenceOutcome.INCOMPLETE_RESPONSE,
    InferenceOutcome.CONTRADICTORY_RESPONSE,
    InferenceOutcome.UNEXPECTED_STRUCTURE,
    InferenceOutcome.AUTHORITY_CLAIM,
    InferenceOutcome.TELEMETRY_INVALID,
    InferenceOutcome.AUTH_FAILURE,
}


# --- Synthetic fixtures (fictional; consistent with the accepted B5.1 fixtures) -----


def _line(number, qty, price, desc="item"):
    return LineItem(line_number=number, description=desc, quantity=qty, unit_price=price)


def _invoice(inv_id, supplier, po_ref, lines, stated):
    return Invoice(
        invoice_id=inv_id, supplier_id=supplier, po_reference=po_ref,
        issued_date="2026-10-12", line_items=lines, stated_total=stated,
    )


def _po(po_id, supplier, lines, stated):
    return PurchaseOrder(
        po_id=po_id, supplier_id=supplier, issued_date="2026-10-12",
        line_items=lines, stated_total=stated,
    )


def _case(case_id, invoice, po_lookup, supplier_state=SupplierMatchState.MATCH):
    return CaseInput(
        case_id=case_id, invoice=invoice, po_lookup=po_lookup,
        supplier_lookup=SupplierLookupResult(state=supplier_state),
        evidence=[], policy_version="policy-v1", input_identity="live-benchmark",
    )


def _exact_match_case():
    inv = _invoice("INV-100", "SUP-100", "PO-100", [_line(1, "2", "25.00")], "50.00")
    po = _po("PO-100", "SUP-100", [_line(1, "2", "25.00")], "50.00")
    return _case("CASE-EXACT", inv, POLookupFound(purchase_order=po))


def _discrepancy_case():
    inv = _invoice("INV-2048", "SUP-9001", "PO-77", [_line(1, "1", "1250.00")], "1250.00")
    po = _po("PO-77", "SUP-9001", [_line(1, "1", "1245.00")], "1245.00")
    return _case("CASE-DISC", inv, POLookupFound(purchase_order=po))


def _missing_po_case():
    inv = _invoice("INV-300", "SUP-300", "PO-300", [_line(1, "1", "10.00")], "10.00")
    return _case("CASE-NOPO", inv, POLookupNotFound())


# --- Scenario model -----------------------------------------------------------------


@dataclass
class LiveScenario:
    name: str
    case: CaseInput
    evidence_text: Optional[str]
    expected_route: Route
    expected_outbound: int                 # 0 for deterministic/held, 1 for inference
    requires_successful_inference: bool    # inference scenario must reach SUCCESS+consistent


SCENARIOS = {
    "exact_match": LiveScenario(
        name="exact_match", case=_exact_match_case(), evidence_text=None,
        expected_route=Route.DETERMINISTIC, expected_outbound=0,
        requires_successful_inference=False,
    ),
    "financial_discrepancy": LiveScenario(
        name="financial_discrepancy", case=_discrepancy_case(),
        evidence_text=SYNTHETIC_EVIDENCE,
        expected_route=Route.INFERENCE, expected_outbound=1,
        requires_successful_inference=True,
    ),
    "missing_po": LiveScenario(
        name="missing_po", case=_missing_po_case(), evidence_text=None,
        expected_route=Route.HELD, expected_outbound=0,
        requires_successful_inference=False,
    ),
}


# --- Result capture -----------------------------------------------------------------


def token_label(value: Optional[int]) -> str:
    """Distinguish missing telemetry from zero consumption."""
    if value is None:
        return NOT_CAPTURED
    if value == 0:
        return "0 (zero consumption)"
    return str(value)


@dataclass
class LiveResult:
    scenario: str
    provenance: str
    transport_kind: str
    dry_run: bool
    route: str
    expected_route: str
    reason: str
    model_calls: int
    outbound_requests: int
    confirmed_response: bool
    model_id: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    elapsed_ms: Optional[int]        # adapter-reported (real model latency when live)
    harness_elapsed_ns: int         # local harness wall-clock
    advisory_outcome: Optional[str]
    advisory_consistent: Optional[bool]
    execution_authority: str
    route_correct: bool
    authority_preserved: bool
    outbound_correct: bool
    harness_safe: bool              # route correct, <=1 bounded request, authority NONE
    measurement_success: bool       # harness_safe AND (successful consistent inference if required)


def _transport_kind(client, url: str) -> str:
    """Derive the transport kind POSITIVELY for the configured AMD request URL.

    Resolves the transport actually selected for ``url`` (including any URL mounts),
    then recognizes only the exact standard ``httpx.HTTPTransport`` as live. Custom
    subclasses, mounted mocks, WSGI, ASGI, and unknown/absent transports all fail
    closed to offline.
    """
    resolve = getattr(client, "_transport_for_url", None)
    selected = None
    if callable(resolve):
        try:
            selected = resolve(httpx.URL(url))
        except Exception:
            selected = None
    if type(selected) is httpx.HTTPTransport:
        return TRANSPORT_LIVE
    return TRANSPORT_OFFLINE


class SingleRequestGuard:
    """Wraps a transport and rejects any attempt beyond one outbound request.

    The transport kind is derived from the actual client it wraps, not from any
    caller-provided label.
    """

    def __init__(self, transport, client, url: str):
        self._transport = transport
        self.calls = 0
        self.transport_kind = _transport_kind(client, url)

    def __call__(self, payload: dict) -> dict:
        self.calls += 1
        if self.calls > 1:
            raise InferenceUnavailable("invariant violated: more than one outbound request")
        return self._transport(payload)


def run_scenario(
    scenario: LiveScenario,
    adapter: Optional[InferenceAdapter] = None,
    model_id: Optional[str] = None,
    outbound: Optional[SingleRequestGuard] = None,
    dry_run: bool = False,
) -> LiveResult:
    """Route one scenario and capture routing, invocation, telemetry, authority.

    Provenance is derived from the guard's actual transport kind (established at the
    transport boundary), never from a caller-provided label.
    """
    start = time.perf_counter_ns()
    decision = route(scenario.case, evidence_text=scenario.evidence_text, adapter=adapter)
    harness_elapsed_ns = time.perf_counter_ns() - start

    ev = decision.evidence
    outbound_requests = outbound.calls if outbound is not None else 0
    transport_kind = outbound.transport_kind if outbound is not None else TRANSPORT_NONE

    # A confirmed response requires a real live transport AND an outcome indicating
    # the endpoint actually responded. An invocation counter is not proof of a
    # response.
    confirmed_response = (
        transport_kind == TRANSPORT_LIVE
        and ev.advisory_outcome in _CONFIRMED_RESPONSE_OUTCOMES
    )

    if transport_kind == TRANSPORT_LIVE:
        if confirmed_response:
            provenance = OBSERVED_LIVE
        elif outbound_requests > 0:
            provenance = LIVE_UNCONFIRMED
        else:
            provenance = LIVE_ZERO_REQUEST
    else:
        provenance = OBSERVED_OFFLINE

    route_correct = decision.route == scenario.expected_route
    authority_preserved = decision.execution_authority == "NONE"
    outbound_correct = (
        outbound_requests == scenario.expected_outbound and ev.model_calls <= 1
    )
    harness_safe = route_correct and outbound_correct and authority_preserved

    advisory_outcome = ev.advisory_outcome.value if ev.advisory_outcome else None
    advisory_consistent = decision.advisory_consistent
    if scenario.requires_successful_inference:
        measurement_success = (
            harness_safe
            and advisory_outcome == InferenceOutcome.SUCCESS.value
            and advisory_consistent is True
        )
    else:
        measurement_success = harness_safe

    return LiveResult(
        scenario=scenario.name,
        provenance=provenance,
        transport_kind=transport_kind,
        dry_run=dry_run,
        route=decision.route.value,
        expected_route=scenario.expected_route.value,
        reason=decision.reason,
        model_calls=ev.model_calls,
        outbound_requests=outbound_requests,
        confirmed_response=confirmed_response,
        model_id=model_id,
        prompt_tokens=ev.prompt_tokens,
        completion_tokens=ev.completion_tokens,
        total_tokens=ev.total_tokens,
        elapsed_ms=ev.elapsed_ms,
        harness_elapsed_ns=harness_elapsed_ns,
        advisory_outcome=advisory_outcome,
        advisory_consistent=advisory_consistent,
        execution_authority=decision.execution_authority,
        route_correct=route_correct,
        authority_preserved=authority_preserved,
        outbound_correct=outbound_correct,
        harness_safe=harness_safe,
        measurement_success=measurement_success,
    )


def format_result(result: LiveResult) -> str:
    lines = [
        f"scenario: {result.scenario}",
        f"provenance: {result.provenance}",
        f"transport_kind: {result.transport_kind}",
        f"dry_run: {result.dry_run}",
        f"route: {result.route} (expected {result.expected_route})",
        f"reason: {result.reason}",
        f"model_calls: {result.model_calls}",
        f"outbound_requests: {result.outbound_requests}",
        f"confirmed_response: {result.confirmed_response}",
        f"model_id: {result.model_id or NOT_CAPTURED}",
        f"prompt_tokens: {token_label(result.prompt_tokens)}",
        f"completion_tokens: {token_label(result.completion_tokens)}",
        f"total_tokens: {token_label(result.total_tokens)}",
        f"elapsed_ms: {result.elapsed_ms if result.elapsed_ms is not None else NOT_CAPTURED}",
        f"harness_elapsed_ns: {result.harness_elapsed_ns}",
        f"advisory_outcome: {result.advisory_outcome or 'None'}",
        f"advisory_consistent: {result.advisory_consistent}",
        f"execution_authority: {result.execution_authority}",
        f"harness_safe: {result.harness_safe}",
        f"measurement_success: {result.measurement_success}",
    ]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------


def _wire_adapter(config: InferenceConfig, client: httpx.Client):
    """Wire the transport + single-request guard over the given client."""
    transport = build_amd_transport(config, client=client)
    url = config.base_url.rstrip("/") + "/chat/completions"
    guard = SingleRequestGuard(transport, client, url)
    adapter = InferenceAdapter(config, guard)
    return adapter, guard


def build_live_adapter(config: InferenceConfig):
    """Wire the REAL live AMD transport (a real httpx client) through the guard."""
    client = httpx.Client(timeout=httpx.Timeout(config.timeout_seconds))
    return _wire_adapter(config, client)


def _dry_run(scenario: LiveScenario, args) -> int:
    result = run_scenario(scenario, adapter=None, model_id=None, outbound=None, dry_run=True)
    print(format_result(result))
    print("\nDry run (preflight): no adapter constructed and no outbound request made.")
    if scenario.requires_successful_inference:
        print("In live mode this scenario requires a successful, source-consistent inference.")
    if args.json:
        _write_json(result, args.json)
    return 0


def _live_run(scenario: LiveScenario, args) -> int:
    config = InferenceConfig.from_env()
    if not config.api_key:
        print("BOUNDARY_AMD_API_KEY is not set; refusing to make a live request.")
        return 1

    adapter, guard = build_live_adapter(config)
    result = run_scenario(scenario, adapter=adapter, model_id=config.model_id,
                          outbound=guard, dry_run=False)
    print(format_result(result))
    if args.json:
        _write_json(result, args.json)
    return 0 if result.measurement_success else 1


def _write_json(result: LiveResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BOUNDARY B5.2 live benchmark runner")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(SCENARIOS),
        help="exactly one scenario to run",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform one live AMD inference request (default: harmless dry run)",
    )
    parser.add_argument("--json", default=None, help="write the result to a JSON file")
    args = parser.parse_args(argv)

    scenario = SCENARIOS[args.scenario]
    if args.live:
        return _live_run(scenario, args)
    return _dry_run(scenario, args)


if __name__ == "__main__":
    raise SystemExit(main())
