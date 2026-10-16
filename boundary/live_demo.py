"""Protected live AMD demonstration (Phase B).

Disabled by default. When enabled by an operator, it makes **at most one** genuine AMD
Token Factory inference request for the single approved synthetic discrepancy invoice
``INV-500``, using the existing classifier/guard/router and the accepted AMD transport —
never a standalone toy chat endpoint.

Safety and honesty:
- **Disabled by default, fail-closed.** The endpoint is non-operational unless *both*
  ``BOUNDARY_LIVE_DEMO_ENABLED=1`` and the operator secret and the AMD credential are set.
- **Operator-gated.** A strong operator secret is compared in constant time; anonymous
  visitors can never reach an AMD call.
- **One request, no retry, no concurrency.** A single execution lock admits one run at a
  time; the AMD transport performs exactly one request and never retries.
- **No fabrication.** The actual route, model identifier, latency, token usage, receipt and
  provenance are returned *only* when a real response was received; a provider error is
  reported as an error, never a fallback success.
- **Authority NONE.** ``execution_authority`` stays ``NONE`` regardless of model output; no
  journal commit, procurement mutation, or financial execution occurs.
- **No secrets.** The AMD credential and operator secret are read from environment variables
  only and are never returned, logged, or embedded in errors/URLs.
"""

from __future__ import annotations

import os
import secrets
import threading

from .amd_transport import build_amd_transport
from .batch import build_batch
from .inference import InferenceAdapter, InferenceConfig
from .live_experiment import build_evidence
from .router import route

ENV_ENABLED = "BOUNDARY_LIVE_DEMO_ENABLED"
ENV_OPERATOR_SECRET = "BOUNDARY_LIVE_DEMO_OPERATOR_SECRET"

MAX_OPERATOR_TOKEN_LEN = 512

_execution_lock = threading.Lock()


class LiveDemoError(Exception):
    """Base class for live-demo control-flow errors (safe to surface)."""


class LiveDemoDisabled(LiveDemoError):
    """The demo is disabled or missing required configuration."""


class LiveDemoBusy(LiveDemoError):
    """An execution is already in progress (single-flight guard)."""


def is_enabled() -> bool:
    """True only when the demo is explicitly enabled AND fully configured."""
    return (
        os.environ.get(ENV_ENABLED, "0") == "1"
        and bool(os.environ.get(ENV_OPERATOR_SECRET, ""))
        and bool(os.environ.get("BOUNDARY_AMD_API_KEY", ""))
    )


def authenticate(token: str | None) -> bool:
    """Constant-time comparison against the operator secret; False when unconfigured."""
    expected = os.environ.get(ENV_OPERATOR_SECRET, "")
    if not expected:
        return False
    token = token or ""
    if len(token) > MAX_OPERATOR_TOKEN_LEN:
        return False
    return secrets.compare_digest(token, expected)


def build_inv500_case():
    """The single approved synthetic discrepancy case, from the frozen workload-v2 sample."""
    manifest = build_batch(batch_id="live-demo", selected_invoice_id="INV-500")
    return manifest.invoices[0].case


def _serialize(decision) -> dict:
    ev = decision.evidence
    result: dict = {
        "case_id": decision.case_id,
        "route": decision.route.value,
        "reason": decision.reason,
        "model_calls": ev.model_calls,
        "advisory_outcome": ev.advisory_outcome.value if ev.advisory_outcome else None,
        "advisory_consistent": decision.advisory_consistent,
        "execution_authority": decision.execution_authority,
    }
    # Real telemetry/identity is surfaced only when a model call was actually made.
    if ev.model_calls:
        result.update({
            "endpoint_model_id": ev.endpoint_model_id,
            "normalized": ev.normalized,
            "receipt": ev.receipt,
            "prompt_tokens": ev.prompt_tokens,
            "completion_tokens": ev.completion_tokens,
            "total_tokens": ev.total_tokens,
            "elapsed_ms": ev.elapsed_ms,
            "diagnostics": ev.diagnostics,
        })
    if decision.interpretation is not None:
        result["interpretation"] = {
            "supplier_id": decision.interpretation.supplier_id,
            "invoice_id": decision.interpretation.invoice_id,
            "po_reference": decision.interpretation.po_reference,
            "amount_cents": decision.interpretation.amount_cents,
            "discrepancy_cents": decision.interpretation.discrepancy_cents,
            "confidence": decision.interpretation.confidence,
        }
    return result


def run_live_demo() -> dict:
    """Run exactly one authorized AMD inference for INV-500, or raise a safe error."""
    if not is_enabled():
        raise LiveDemoDisabled("live demo is disabled or unconfigured")
    if not _execution_lock.acquire(blocking=False):
        raise LiveDemoBusy("a live demonstration is already in progress")
    try:
        config = InferenceConfig.from_env()
        if not config.api_key:
            raise LiveDemoDisabled("AMD credential not configured")
        adapter = InferenceAdapter(config, build_amd_transport(config))
        case = build_inv500_case()
        evidence_text = build_evidence(case)
        decision = route(case, evidence_text=evidence_text, adapter=adapter)
        return _serialize(decision)
    finally:
        _execution_lock.release()
