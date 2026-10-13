"""Railway / container startup entry point for the BOUNDARY offline demo.

Seeds the synthetic demo journal (idempotently) and then serves the FastAPI app on
Railway's ``PORT`` environment variable. Requires no AMD/Evolus credentials; the
served application is fully offline and read-only.

Usage (local or Railway)::

    python -m boundary.serve

Environment variables:

- ``PORT`` — the port to bind (Railway supplies this; defaults to 8000).
- ``BOUNDARY_DB_PATH`` — the SQLite journal path (defaults to ``boundary.db``).
"""

from __future__ import annotations

import os
import sys

import uvicorn

from .app import create_app
from .demo import run_demo

DEFAULT_DB_PATH = "boundary.db"


def build_app():
    """Seed the synthetic demo journal and return the configured FastAPI app.

    Reads ``BOUNDARY_DB_PATH`` (default ``boundary.db``). Idempotent: safe on a
    fresh deployment (creates the journal) and on restart (no duplicate commits).
    Raises on an unwritable path so a misconfigured deployment fails fast.
    """
    db_path = os.environ.get("BOUNDARY_DB_PATH", DEFAULT_DB_PATH)
    run_demo(db_path)
    return create_app(db_path)


def _parse_port(raw: str) -> tuple[int | None, str | None]:
    """Parse and validate the listen port (1-65535 inclusive).

    Returns ``(port, None)`` on success or ``(None, error_message)`` on failure.
    """
    try:
        port = int(raw)
    except ValueError:
        return None, f"invalid PORT {raw!r} (must be an integer 1-65535)"
    if not 1 <= port <= 65535:
        return None, f"invalid PORT {port} (must be an integer 1-65535)"
    return port, None


def main() -> int:
    # Validate configuration BEFORE touching the journal so a bad PORT has no
    # database side effects on startup.
    port, port_error = _parse_port(os.environ.get("PORT", "8000"))
    if port_error is not None:
        print(f"BOUNDARY startup failed: {port_error}", file=sys.stderr)
        return 1

    try:
        app = build_app()
    except Exception as exc:  # noqa: BLE001 — surface a clear startup error
        print(
            f"BOUNDARY startup failed: cannot initialize journal at "
            f"{os.environ.get('BOUNDARY_DB_PATH', DEFAULT_DB_PATH)!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"BOUNDARY demo served on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
