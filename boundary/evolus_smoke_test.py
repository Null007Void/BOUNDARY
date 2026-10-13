"""Opt-in live smoke test for the Evolus document integration (B4.3).

Importing this module performs no network calls. A live submission is made only
when ``--live`` is passed explicitly together with an explicit ``--file`` path::

    python -m boundary.evolus_smoke_test --live --file /path/to/BOUNDARY_synthetic_invoice_INV-2048.txt

The smoke test reads the file's actual bytes (rejecting missing/unreadable files
before any POST), submits them as a ``file`` attachment, reconciles the
asynchronous execution, and verifies the extraction matches the known synthetic
invoice fixture. It never prints the API credential.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .evolus import EvolusOutcome
from .evolus_http import EvolusExecutionResult, EvolusHttpClient, EvolusHttpConfig

# Expected identity and total for the known synthetic invoice fixture.
EXPECTED_INVOICE_ID = "INV-2048"
EXPECTED_STATED_TOTAL = "USD 1,250.00"


def build_live_client() -> EvolusHttpClient:
    """Build a client from environment configuration (makes no network call)."""
    return EvolusHttpClient(EvolusHttpConfig.from_env())


def format_result(result: EvolusExecutionResult) -> str:
    lines: List[str] = []
    lines.append(f"outcome: {result.outcome.value}")
    if result.execution_id:
        lines.append(f"execution_id: {result.execution_id}")
    if result.status:
        lines.append(f"status: {result.status}")
    if result.authority:
        lines.append(f"authority: {result.authority}")
    if result.evidence_status:
        lines.append(f"evidence_status: {result.evidence_status}")
    if result.extraction is not None:
        e = result.extraction
        lines.append(
            f"extraction: documentType={e.get('documentType')} supplierId={e.get('supplierId')} "
            f"invoiceId={e.get('invoiceId')} purchaseOrderReference={e.get('purchaseOrderReference')} "
            f"statedTotal={e.get('statedTotal')}"
        )
    if result.advisory_analysis is not None:
        lines.append(f"advisory_analysis: {result.advisory_analysis}")
    if result.original_file_name:
        lines.append(f"original_file: {result.original_file_name}")
    if result.original_file_sha256:
        lines.append(f"original_file_sha256: {result.original_file_sha256}")
    if result.errors:
        lines.append(f"errors: {result.errors}")
    if result.reasons:
        lines.append(f"reasons: {result.reasons}")
    return "\n".join(lines)


def verify_identity(result: EvolusExecutionResult) -> bool:
    """Return True when the extraction matches the known synthetic fixture."""
    if result.extraction is None:
        return False
    return (
        result.extraction.get("invoiceId") == EXPECTED_INVOICE_ID
        and result.extraction.get("statedTotal") == EXPECTED_STATED_TOTAL
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evolus document-integration live smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform one live submission + reconciliation of the synthetic invoice",
    )
    parser.add_argument("--file", help="path to the synthetic invoice document")
    args = parser.parse_args(argv)

    if not args.live:
        print("Dry run: no live request made. Pass --live --file <path> to submit.")
        return 0
    if not args.file:
        print("--file is required for a live submission.")
        return 1

    path = Path(args.file)
    try:
        file_bytes = path.read_bytes()
    except OSError as exc:
        print(f"Unable to read document {args.file}: {exc}")
        return 1

    config = EvolusHttpConfig.from_env()
    if not config.api_key:
        print("BOUNDARY_EVOLUS_API_KEY is not set; refusing to make a live request.")
        return 1

    client = EvolusHttpClient(config)
    result = client.run(filename=path.name, file_bytes=file_bytes, content_type="text/plain")
    print(format_result(result))

    if result.outcome != EvolusOutcome.SUCCESS:
        return 1
    if not verify_identity(result):
        print("identity mismatch: extraction does not match the expected synthetic fixture")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
