"""S4E.4 — Bounded DeepSeek-vs-DeepSeek live benchmark experiment runner.

The experiment measures two strategies over the three safe synthetic scenarios
(exact match, financial discrepancy, missing purchase order) with two repetitions:

- **Strategy A — LLM-first baseline**: one bounded advisory-only model request per
  scenario, every repetition (3 x 2 = 6 requests planned).
- **Strategy B — BOUNDARY deterministic-first**: route per the accepted contract and
  invoke the model only for the eligible financial discrepancy (1 x 2 = 2 requests).

The cumulative outbound-request limit (8) is enforced in executable code by
``RequestBudget`` at the transport boundary; a ninth request is rejected. The runner
derives its transport from the adapter, so mismatched transport/budget wiring is
impossible.

Execution modes
---------------

- **Mock preflight (default).** ``python -m boundary.live_experiment`` runs a fully
  offline mock-only run. No network, no credentials.
- **Live run (explicit, gated).** ``python -m boundary.live_experiment --live`` wires the
  accepted AMD transport (``build_amd_transport``) through the bounded transport and
  issues real requests. Live mode is entered ONLY via the explicit ``--live`` flag, and
  is refused before any network activity unless the API key, the designated model, and
  the explicitly approved AMD Token Factory endpoint are all present and valid.

Budget scope (important)
------------------------

``RequestBudget`` is **per-invocation**, not a durable cross-process limit. Each process
issues up to 8 requests; running the command N times issues up to N×8. It must never be
presented as a persistent cap. Results are written only on completion (overwrite, never
append), so an interrupted run does not contaminate a prior run's evidence.

Evidence-integrity guarantees (review corrections)
---------------------------------------------------

- Per-attempt state is isolated: a budget-rejected request never inherits the previous
  request's model identity, prompt, response, or verification result.
- Confirmation is based on positively observed transport behavior (a response was
  actually returned), not on inference outcomes alone.
- Mock response fields (model, usage) are never presented as live-provider evidence; the
  previously recorded B5 token counts are not reused as default mock measurements, and
  missing provider telemetry stays missing.
- Baseline source consistency is checked against trusted facts and distinguished from
  successful schema parsing; a missing trusted purchase order is "not verifiable".
- The trusted lookup state (NOT_FOUND vs UNAVAILABLE) is preserved, not conflated.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .benchmark_live import SCENARIOS
from .contracts import POState
from .inference import InferenceAdapter, InferenceOutcome, InferenceUnavailable
from .router import route

DESIGNATED_MODEL = "DeepSeek-V4-Flash"
MAX_OUTBOUND_REQUESTS = 8
REPETITIONS = 2
SAFE_SCENARIOS = ["exact_match", "financial_discrepancy", "missing_po"]

MOCK_PROVENANCE = "OBSERVED_OFFLINE"
LIVE_PROVENANCE = "OBSERVED_LIVE"
LIVE_UNCONFIRMED_PROVENANCE = "LIVE_UNCONFIRMED"
LIVE_ZERO_REQUEST_PROVENANCE = "LIVE_ZERO_REQUEST"

# The explicitly approved AMD Token Factory inference endpoint (B3.2). A live run
# requires exactly this endpoint and never silently falls back to an unrelated default.
APPROVED_AMD_ENDPOINT = "https://developer.amd.com.cn/radeon/api/v1"


# --- Gap 1: complete synthetic evidence -------------------------------------------


def _dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"${sign}{c // 100}.{c % 100:02d}"


def build_evidence(case) -> str:
    """Deterministic synthetic evidence derived from the case's actual facts.

    Preserves the actual trusted lookup state: NOT_FOUND (definitively nothing) is
    distinguished from UNAVAILABLE (lookup could not return a result).
    """
    inv = case.invoice
    state = case.po_lookup.state
    ref = inv.po_reference or "(none)"
    if state == POState.NOT_FOUND:
        return (
            f"Supplier record for {inv.supplier_id} references invoice {inv.invoice_id} "
            f"with a stated total of {_dollars(inv.stated_total_cents)} and a purchase "
            f"order {ref}. The trusted purchase order lookup found no purchase order {ref}."
        )
    if state == POState.UNAVAILABLE:
        return (
            f"Supplier record for {inv.supplier_id} references invoice {inv.invoice_id} "
            f"with a stated total of {_dollars(inv.stated_total_cents)} and a purchase "
            f"order {ref}. The trusted purchase order lookup is unavailable."
        )
    po = case.po_lookup.purchase_order
    if inv.computed_total_cents == po.computed_total_cents:
        return (
            f"Supplier record for {inv.supplier_id} references invoice {inv.invoice_id} "
            f"with a stated total of {_dollars(inv.stated_total_cents)} and purchase "
            f"order {po.po_id}. The trusted purchase order {po.po_id} states a total of "
            f"{_dollars(po.stated_total_cents)}. The invoice and purchase order amounts match."
        )
    return (
        f"Supplier record for {inv.supplier_id} references invoice {inv.invoice_id} "
        f"with a stated total of {_dollars(inv.stated_total_cents)} and purchase "
        f"order {po.po_id}. The trusted purchase order {po.po_id} states a total of "
        f"{_dollars(po.stated_total_cents)}. The discrepancy is ambiguous."
    )


def scenario_evidence(name: str) -> str:
    return build_evidence(SCENARIOS[name].case)


# --- Gap 2: model identity verification -------------------------------------------


def verify_model_identity(endpoint_model_id: Optional[str]) -> str:
    """Classify the endpoint-reported model identity against the designated model."""
    if endpoint_model_id is None:
        return "unverified_missing"
    if endpoint_model_id != DESIGNATED_MODEL:
        return "unverified_mismatch"
    return "verified"


# --- Baseline source-consistency ---------------------------------------------------


def baseline_source_consistency(case, interpretation) -> str:
    """Classify a baseline interpretation against trusted facts.

    Known trusted invoice facts (supplier, invoice id, computed amount, and the
    invoice's declared PO reference) are validated first and independently of the PO
    lookup state, so a contradiction is ``source_inconsistent`` even when the trusted
    PO is missing. When the invoice facts do not contradict but the trusted PO is
    missing, the discrepancy cannot be independently verified (``not_verifiable``). A
    fully matching interpretation with both trusted invoice and PO facts is
    ``source_consistent``.
    """
    inv = case.invoice

    # 1. Known trusted invoice facts — validated unconditionally.
    if interpretation.supplier_id != inv.supplier_id:
        return "source_inconsistent"
    if interpretation.invoice_id != inv.invoice_id:
        return "source_inconsistent"
    if interpretation.amount_cents != inv.computed_total_cents:
        return "source_inconsistent"
    if inv.po_reference is not None and interpretation.po_reference is not None:
        if interpretation.po_reference != inv.po_reference:
            return "source_inconsistent"

    # 2. Invoice facts do not contradict; a missing trusted PO blocks independent
    #    discrepancy verification.
    if case.po_lookup.state != POState.FOUND:
        return "not_verifiable"

    # 3. Both trusted invoice and PO facts are available; the interpretation must
    #    identify the appropriate trusted PO (an omitted reference is not enough).
    po = case.po_lookup.purchase_order
    expected_disc = inv.computed_total_cents - po.computed_total_cents
    if interpretation.discrepancy_cents != expected_disc:
        return "source_inconsistent"
    if interpretation.po_reference != po.po_id:
        return "source_inconsistent"
    return "source_consistent"


# --- Gap 3: bounded experiment runner ---------------------------------------------


class RequestBudget:
    """Enforces the cumulative outbound-request limit across a single experiment run.

    Per-invocation only: a fresh budget is created each process; it is NOT a durable
    cross-process limit.
    """

    def __init__(self, limit: int = MAX_OUTBOUND_REQUESTS):
        self.limit = limit
        self.attempted = 0
        self.rejected = 0

    def acquire(self) -> bool:
        if self.attempted >= self.limit:
            self.rejected += 1
            return False
        self.attempted += 1
        return True


@dataclass
class RequestEvent:
    strategy: str
    scenario: str
    repetition: int
    endpoint_model_id: Optional[str]
    model_verified: str
    outcome: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    elapsed_ms: Optional[int]
    confirmed: bool
    execution_authority: str
    source_consistency: Optional[str]
    provenance: str
    payload: Optional[dict] = None


class ExperimentTransport:
    """Wraps the adapter transport: enforces the budget and captures the endpoint-reported
    model identity and request payload at the outbound boundary.

    ``live`` marks whether the wrapped transport is the real AMD HTTP transport (so
    provenance can distinguish OBSERVED_LIVE / LIVE_UNCONFIRMED from OBSERVED_OFFLINE).
    Per-attempt state is reset BEFORE the budget check so a rejected request can never
    inherit the previous request's model identity or payload.
    """

    def __init__(self, transport, budget: RequestBudget, live: bool = False):
        self._transport = transport
        self._budget = budget
        self.live = live
        self.last_endpoint_model: Optional[str] = None
        self.last_payload: Optional[dict] = None
        self.last_content: Optional[str] = None
        self.got_response: bool = False

    def __call__(self, payload: dict) -> dict:
        # Isolate state per attempt (before any rejection path).
        self.last_payload = payload
        self.last_endpoint_model = None
        self.last_content = None
        self.got_response = False
        if not self._budget.acquire():
            raise InferenceUnavailable("experiment request budget exhausted")
        raw = self._transport(payload)
        self.got_response = True  # the transport actually returned a response
        if isinstance(raw, dict):
            self.last_endpoint_model = raw.get("model")
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    self.last_content = message["content"]
        return raw


def _provenance(transport: ExperimentTransport, confirmed: bool) -> str:
    if not transport.live:
        return MOCK_PROVENANCE
    return LIVE_PROVENANCE if confirmed else LIVE_UNCONFIRMED_PROVENANCE


def _event(
    strategy: str,
    scenario: str,
    repetition: int,
    outcome: Optional[InferenceOutcome],
    telemetry,
    transport: ExperimentTransport,
    source_consistency: Optional[str] = None,
) -> RequestEvent:
    return RequestEvent(
        strategy=strategy,
        scenario=scenario,
        repetition=repetition,
        endpoint_model_id=transport.last_endpoint_model,
        model_verified=verify_model_identity(transport.last_endpoint_model),
        outcome=(outcome.value if outcome is not None else None),
        prompt_tokens=telemetry.prompt_tokens if telemetry else None,
        completion_tokens=telemetry.completion_tokens if telemetry else None,
        total_tokens=telemetry.total_tokens if telemetry else None,
        elapsed_ms=telemetry.elapsed_ms if telemetry else None,
        confirmed=transport.got_response,
        execution_authority="NONE",
        source_consistency=source_consistency,
        provenance=_provenance(transport, transport.got_response),
        payload=transport.last_payload,
    )


def run_experiment(adapter: InferenceAdapter, repetitions: int = REPETITIONS, on_event=None):
    """Run Strategy A (LLM-first) then Strategy B (deterministic-first).

    The transport is derived from the adapter, so the runner can never operate with a
    transport/budget different from the one the adapter actually consumes. ``on_event``
    (optional) is called with each ``RequestEvent`` as soon as it is produced, so a
    caller can persist attempted-request evidence incrementally.
    """
    transport = getattr(adapter, "_transport", None)
    if not isinstance(transport, ExperimentTransport):
        raise TypeError(
            "adapter must be wired to an ExperimentTransport for the bounded experiment"
        )

    events: List[RequestEvent] = []
    route_records: List[dict] = []

    # Strategy A — LLM-first: invoke the model for every scenario.
    for rep in range(repetitions):
        for name in SAFE_SCENARIOS:
            case = SCENARIOS[name].case
            result = adapter.interpret(scenario_evidence(name))
            sc = None
            if result.interpretation is not None:
                sc = baseline_source_consistency(case, result.interpretation)
            event = _event("A", name, rep, result.outcome, result.telemetry, transport, sc)
            events.append(event)
            if on_event is not None:
                on_event(event)

    # Strategy B — BOUNDARY deterministic-first: route each scenario.
    for rep in range(repetitions):
        for name in SAFE_SCENARIOS:
            decision = route(
                SCENARIOS[name].case,
                evidence_text=scenario_evidence(name),
                adapter=adapter,
            )
            made = decision.evidence.model_calls
            if made > 0:
                ev = decision.evidence
                sc = (
                    "source_consistent" if decision.advisory_consistent is True
                    else "source_inconsistent" if decision.advisory_consistent is False
                    else None
                )
                event = RequestEvent(
                    strategy="B",
                    scenario=name,
                    repetition=rep,
                    endpoint_model_id=transport.last_endpoint_model,
                    model_verified=verify_model_identity(transport.last_endpoint_model),
                    outcome=(ev.advisory_outcome.value if ev.advisory_outcome else None),
                    prompt_tokens=ev.prompt_tokens,
                    completion_tokens=ev.completion_tokens,
                    total_tokens=ev.total_tokens,
                    elapsed_ms=ev.elapsed_ms,
                    confirmed=transport.got_response,
                    execution_authority=decision.execution_authority,
                    source_consistency=sc,
                    provenance=_provenance(transport, transport.got_response),
                    payload=transport.last_payload,
                )
                events.append(event)
                if on_event is not None:
                    on_event(event)
            route_records.append(
                {
                    "strategy": "B",
                    "scenario": name,
                    "repetition": rep,
                    "route": decision.route.value,
                    "requests_made": made,
                    "execution_authority": decision.execution_authority,
                    "provenance": (
                        (LIVE_ZERO_REQUEST_PROVENANCE if transport.live else MOCK_PROVENANCE)
                        if made == 0 else None
                    ),
                }
            )

    return events, route_records


# --- Live configuration validation (refuse before any network activity) -----------


class LiveConfigError(Exception):
    """The live-run configuration is unsafe; the run was refused before network activity."""


def validate_live_config(config) -> Optional[str]:
    """Return an error string if the config is unsafe for a live run, else None.

    Checks (in order, before any network activity):
    1. API key present.
    2. Model id is the designated DeepSeek-V4-Flash.
    3. Base URL is the explicitly approved AMD Token Factory endpoint.
    """
    if not config.api_key:
        return "BOUNDARY_AMD_API_KEY is not set"
    if config.model_id != DESIGNATED_MODEL:
        return f"model_id {config.model_id!r} is not the designated model {DESIGNATED_MODEL!r}"
    if config.base_url != APPROVED_AMD_ENDPOINT:
        return f"base_url {config.base_url!r} is not the approved AMD Token Factory endpoint"
    return None


def build_live_experiment():
    """Build a live experiment: the accepted AMD transport wrapped in the bounded,
    provenance-tagged transport. Raises :class:`LiveConfigError` before any network
    activity if the configuration is unsafe."""
    from .amd_transport import build_amd_transport
    from .inference import InferenceConfig

    config = InferenceConfig.from_env()
    err = validate_live_config(config)
    if err is not None:
        raise LiveConfigError(err)

    budget = RequestBudget()
    transport = ExperimentTransport(build_amd_transport(config), budget, live=True)
    adapter = InferenceAdapter(config, transport)
    return adapter, transport, budget


# --- Mock transport for the offline preflight -------------------------------------


def _correct_interpretation(case) -> str:
    """A case-appropriate (source-consistent) interpretation derived from trusted facts."""
    inv = case.invoice
    if case.po_lookup.state != POState.FOUND:
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


def build_mock_transport(
    model: str = DESIGNATED_MODEL,
    content: Optional[str] = None,
    usage: Optional[dict] = None,
    exc: Optional[Exception] = None,
):
    """Return a plain-function mock transport (no HTTP, no network)."""
    case_map = {
        build_evidence(SCENARIOS[name].case): _correct_interpretation(SCENARIOS[name].case)
        for name in SAFE_SCENARIOS
    }

    def transport(payload: dict) -> dict:
        if exc is not None:
            raise exc
        if content is not None:
            body = content
        else:
            user_msg = payload["messages"][1]["content"]
            body = case_map.get(user_msg, "{}")
        resp = {"model": model, "choices": [{"message": {"content": body}}]}
        if usage is not None:
            resp["usage"] = usage
        return resp

    return transport


def build_preflight(
    model: str = DESIGNATED_MODEL,
    content: Optional[str] = None,
    exc: Optional[Exception] = None,
):
    """Build a default-offline preflight: a mock adapter + bounded transport + budget."""
    from .inference import InferenceConfig

    budget = RequestBudget()
    transport = ExperimentTransport(build_mock_transport(model=model, content=content, exc=exc), budget)
    adapter = InferenceAdapter(InferenceConfig(model_id=DESIGNATED_MODEL), transport)
    return adapter, transport, budget


# --- Results serialization + CLI ---------------------------------------------------


def _new_run_id() -> str:
    """Return a unique run identifier (timestamp + short random suffix).

    Scopes live output filenames so a completed run never silently overwrites a
    previous run's evidence.
    """
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _write_exclusive(path: str, obj) -> None:
    """Write JSON to ``path`` using exclusive-create (fail if the file already exists)."""
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_exclusive_text(path: str, text: str) -> None:
    """Write plain UTF-8 text (e.g. Markdown) using exclusive-create — never JSON."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def _results(provenance: str, budget: RequestBudget, events, route_records) -> dict:
    return {
        "provenance": provenance,
        "designated_model": DESIGNATED_MODEL,
        "repetitions": REPETITIONS,
        "scenarios": SAFE_SCENARIOS,
        "request_budget": {
            "limit": budget.limit,
            "attempted": budget.attempted,
            "rejected": budget.rejected,
        },
        "budget_scope": (
            "per-invocation only (not a durable cross-process limit); each run issues "
            "up to 8 requests"
        ),
        "strategy_a_requests": sum(1 for e in events if e.strategy == "A"),
        "strategy_b_requests": sum(1 for e in events if e.strategy == "B"),
        "events": [asdict(e) for e in events],
        "route_records": route_records,
    }


def _write(results: dict, json_path: str, report_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_render(results) + "\n")


def _mock_main(args) -> int:
    json_path = args.json or "live_experiment_results.json"
    report_path = args.report or "live_experiment_report.md"
    adapter, transport, budget = build_preflight()
    events, route_records = run_experiment(adapter)
    results = _results("OBSERVED_OFFLINE (mock preflight)", budget, events, route_records)
    _write(results, json_path, report_path)
    print(_render(results))
    return 0


def _live_main(args) -> int:
    try:
        adapter, transport, budget = build_live_experiment()
    except LiveConfigError as exc:
        print(f"BOUNDARY live run refused: {exc}")
        return 1

    # Unique run id scopes all output so a completed run never silently overwrites a
    # previous run's evidence. Exclusive-create enforces immutability.
    run_id = _new_run_id()
    events_path = f"live_experiment_live_{run_id}_events.jsonl"
    results_path = args.json or f"live_experiment_live_{run_id}_results.json"
    report_path = args.report or f"live_experiment_live_{run_id}_report.md"

    # Validate output destinations BEFORE any outbound request. A preexisting path is
    # rejected safely so live requests are not consumed only to fail at write time.
    for p in (events_path, results_path, report_path):
        if os.path.exists(p):
            print(f"BOUNDARY live run refused: output destination already exists: {p}")
            return 1

    recorded: List[dict] = []
    ef = open(events_path, "x", encoding="utf-8")

    def on_event(event):
        recorded.append(asdict(event))
        ef.write(json.dumps(asdict(event)) + "\n")
        ef.flush()

    try:
        events, route_records = run_experiment(adapter, on_event=on_event)
    except BaseException:
        # Interrupted: preserve attempted-request evidence with an explicit partial
        # status (never combined with a completed run's claims).
        _write_exclusive(
            f"live_experiment_live_{run_id}_partial.json",
            {"run_id": run_id, "status": "partial", "attempted": budget.attempted, "events": recorded},
        )
        raise
    finally:
        ef.close()

    results = _results("LIVE (AMD Token Factory)", budget, events, route_records)
    results["run_id"] = run_id
    results["status"] = "completed"
    _write_exclusive(results_path, results)
    _write_exclusive_text(report_path, _render(results))
    print(_render(results))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BOUNDARY S4E.4 live-experiment runner")
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform the LIVE benchmark against AMD Token Factory (default: offline mock preflight)",
    )
    parser.add_argument("--json", default=None, help="machine-readable output path")
    parser.add_argument("--report", default=None, help="human-readable report path")
    args = parser.parse_args(argv)

    if args.live:
        return _live_main(args)
    return _mock_main(args)


def _render(results: dict) -> str:
    budget = results["request_budget"]
    lines = [
        f"# BOUNDARY — Live Experiment ({results['provenance']})",
        "",
        f"- Designated model: {results['designated_model']}",
        f"- Scenarios: {', '.join(results['scenarios'])}",
        f"- Repetitions: {results['repetitions']}",
        f"- Request budget: {budget['attempted']} attempted / {budget['limit']} limit / "
        f"{budget['rejected']} rejected",
        f"- Strategy A (LLM-first) requests: {results['strategy_a_requests']}",
        f"- Strategy B (BOUNDARY) requests: {results['strategy_b_requests']}",
        f"- Budget scope: {results['budget_scope']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
