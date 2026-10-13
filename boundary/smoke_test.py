"""Live smoke test for the AMD inference adapter (B3.2).

Importing this module performs no network calls. A live inference is made only
when ``--live`` is passed explicitly::

    python -m boundary.smoke_test --live

The smoke test runs a single synthetic invoice-evidence inference and prints
the sanitized result and usage telemetry. It never prints the API credential
and never performs any simulated accounting commit.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .amd_transport import build_amd_transport
from .inference import InferenceAdapter, InferenceConfig, InferenceOutcome, InferenceResult

SYNTHETIC_EVIDENCE = (
    "Supplier record for SUP-9001 (Acme Industrial) references invoice INV-2048 "
    "with a stated total of $1,250.00 and a purchase order PO-77. A separate "
    "line item suggests $1,245.00. The discrepancy is ambiguous."
)


def build_live_adapter() -> tuple[InferenceConfig, InferenceAdapter]:
    """Build the adapter wired to the live AMD HTTP transport."""
    config = InferenceConfig.from_env()
    adapter = InferenceAdapter(config, build_amd_transport(config))
    return config, adapter


def format_result(result: InferenceResult) -> str:
    """Return a sanitized human-readable summary (never includes credentials)."""
    lines: List[str] = []
    lines.append(f"outcome: {result.outcome.value}")
    if result.interpretation is not None:
        i = result.interpretation
        lines.append(
            f"interpretation: supplier={i.supplier_id} invoice={i.invoice_id} "
            f"amount_cents={i.amount_cents} discrepancy_cents={i.discrepancy_cents} "
            f"confidence={i.confidence}"
        )
    if result.telemetry is not None:
        t = result.telemetry
        lines.append(
            f"telemetry: model={t.model_id} prompt_tokens={t.prompt_tokens} "
            f"completion_tokens={t.completion_tokens} total_tokens={t.total_tokens} "
            f"elapsed_ms={t.elapsed_ms}"
        )
        if t.error_class:
            lines.append(f"error_class: {t.error_class}")
    if result.reasons:
        lines.append(f"reasons: {result.reasons}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AMD inference live smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform one live inference against the configured AMD endpoint",
    )
    args = parser.parse_args(argv)

    if not args.live:
        print("Dry run: no live request made. Pass --live to perform one live inference.")
        return 0

    config = InferenceConfig.from_env()
    if not config.api_key:
        print("BOUNDARY_AMD_API_KEY is not set; refusing to make a live request.")
        return 1

    adapter = InferenceAdapter(config, build_amd_transport(config))
    result = adapter.interpret(SYNTHETIC_EVIDENCE)
    print(format_result(result))
    return 0 if result.outcome == InferenceOutcome.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
