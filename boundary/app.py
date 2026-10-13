"""Offline FastAPI demonstration interface for BOUNDARY.

A minimal case-list and case-detail UI. All routes are read-only: the web layer
never commits a simulated approval, and no anonymous HTTP mutation endpoint is
exposed. The journal is opened in SQLite read-only mode, so HTTP GET requests
never initialize, migrate, or otherwise mutate it. ``/health`` distinguishes a
usable local journal from a missing/unavailable one (no AMD/Evolus claims).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime, timezone
from decimal import Decimal

from .classifier import classify
from .demo import demo_cases
from .evidence import AMD_EVIDENCE, B5_1_COUNTERFACTUAL, B5_1_OBSERVED_OFFLINE, EVOLUS_EVIDENCE
from .guard import evaluate
from .ledger import ActionJournal
from .routing_demo import authority_claim_rejection, routing_for_case
from .verifier import verify

DEFAULT_DB_PATH = "boundary.db"

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def format_cents(cents: int | None) -> str:
    """Format integer USD cents as a dollar string, exactly (no floating point)."""
    if cents is None:
        return "n/a"
    dollars = Decimal(cents) / Decimal(100)
    return f"${dollars:.2f}"


def pretty_timestamp(ts: str | None) -> str:
    """Format an ISO-8601 timestamp for human reading, converting to UTC only when
    the timestamp is timezone-aware; a naive timestamp is never labeled UTC."""
    if not ts:
        return "\u2014"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _journal_state(db_path: str) -> str:
    """Classify the journal as ready, uninitialized, or unavailable.

    Never creates or modifies the database.
    """
    if not os.path.exists(db_path):
        return "uninitialized"
    try:
        journal = ActionJournal(db_path, read_only=True)
        try:
            journal.count()  # validates the schema is present and queryable
        finally:
            journal.close()
        return "ready"
    except sqlite3.Error:
        return "unavailable"


def create_app(db_path: str = DEFAULT_DB_PATH) -> FastAPI:
    app = FastAPI(title="BOUNDARY — offline invoice-exception demo")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.autoescape = True  # safe HTML escaping on all rendered values
    templates.env.filters["cents"] = format_cents
    templates.env.filters["ts"] = pretty_timestamp
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    def _case_by_id(case_id: str):
        for label, case in demo_cases():
            if case.case_id == case_id:
                return label, case
        return None

    @app.get("/health")
    def health():
        state = _journal_state(db_path)
        if state == "ready":
            return {
                "status": "ok",
                "journal": "ready",
                "offline": True,
                "amd_connected": False,
                "evolus_connected": False,
            }
        return JSONResponse(
            status_code=503,
            content={
                "status": state,
                "journal": state,
                "offline": True,
                "amd_connected": False,
                "evolus_connected": False,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def case_list(request: Request):
        state = _journal_state(db_path)
        rows = []
        if state == "ready":
            journal = ActionJournal(db_path, read_only=True)
            try:
                for label, case in demo_cases():
                    record = journal.get(case.case_id)
                    rows.append({"case": case, "label": label, "committed": record is not None})
            finally:
                journal.close()
        return templates.TemplateResponse(request, "cases.html", {"rows": rows, "state": state})

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case_detail(request: Request, case_id: str):
        found = _case_by_id(case_id)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown case")
        label, case = found

        proposal = classify(case)
        guard = evaluate(case)

        # Offline B3 routing demonstration (advisory only; never changes the B1 trace).
        routing = routing_for_case(case)
        authority_claim = authority_claim_rejection(case)

        state = _journal_state(db_path)
        record = None
        verification = None
        if state == "ready":
            journal = ActionJournal(db_path, read_only=True)
            try:
                record = journal.get(case_id)
                verification = verify(journal, case, receipt=record)
            finally:
                journal.close()

        return templates.TemplateResponse(
            request,
            "case_detail.html",
            {
                "case": case,
                "label": label,
                "proposal": proposal,
                "guard": guard,
                "record": record,
                "verification": verification,
                "routing": routing,
                "authority_claim": authority_claim,
                "state": state,
            },
        )

    @app.get("/evidence", response_class=HTMLResponse)
    def evidence_page(request: Request):
        return templates.TemplateResponse(
            request,
            "evidence.html",
            {
                "amd": AMD_EVIDENCE,
                "evolus": EVOLUS_EVIDENCE,
                "b51_observed": B5_1_OBSERVED_OFFLINE,
                "b51": B5_1_COUNTERFACTUAL,
            },
        )

    return app


app = create_app()
