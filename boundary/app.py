"""Offline FastAPI demonstration interface for BOUNDARY.

A minimal case-list and case-detail UI. All routes are read-only: the web layer
never commits a simulated approval, and no anonymous HTTP mutation endpoint is
exposed. The journal is opened in SQLite read-only mode, so HTTP GET requests
never initialize, migrate, or otherwise mutate it. ``/health`` distinguishes a
usable local journal from a missing/unavailable one (no AMD/Evolus claims).
"""

from __future__ import annotations

import json
import os
import sqlite3
from urllib.parse import parse_qsl
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime, timezone
from decimal import Decimal

from .classifier import classify
from .demo import demo_cases
from .evidence import (
    AMD_EVIDENCE,
    B5_1_COUNTERFACTUAL,
    B5_1_OBSERVED_OFFLINE,
    EVOLUS_EVIDENCE,
)
from .public_experiment import (
    GEMMA_HISTORY,
    GEMMA_SOURCE_CONSISTENCY_NOTE,
    amd_demonstration,
    gemma_demonstration,
    load_public_experiment,
)
from .guard import evaluate
from .intake import IntakeError, build_case, evaluate_intake
from .ledger import ActionJournal
from .routing_demo import authority_claim_rejection, routing_for_case
from .verifier import verify
from .batch import MODELS, STRATEGIES, run_batch_durable

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


def batch_paths(db_path: str) -> dict:
    """Durable state file paths for the batch workflow (isolated from the demo journal)."""
    return {
        "journal": f"{db_path}.batch.db",
        "result": f"{db_path}.batch-result.json",
        "events": f"{db_path}.batch-events.jsonl",
        "progress": f"{db_path}.batch-progress.json",
    }


def initiate_batch(db_path: str, **kwargs) -> dict:
    """Explicit bounded local initiation: run the batch once and persist durable state.

    This is the ONLY entry point that executes the batch. The web views never call it.
    """
    paths = batch_paths(db_path)
    return run_batch_durable(
        journal_path=paths["journal"],
        event_path=paths["events"],
        result_path=paths["result"],
        progress_path=paths["progress"],
        **kwargs,
    )


def read_batch_result(db_path: str):
    """Read the durable batch result (read-only); ``None`` when not yet initiated."""
    paths = batch_paths(db_path)
    if os.path.exists(paths["result"]):
        with open(paths["result"], encoding="utf-8") as f:
            return json.load(f)
    return None


def read_batch_state(db_path: str):
    """Read the current durable batch state (read-only), reconciled by run identity.

    Returns a completed result, a running progress projection, or ``None`` when not initiated.
    A newer active run (progress ``run_id`` differing from the stored result's ``run_id``) must
    win over an older completed result; an old result never hides a newer batch.
    """
    paths = batch_paths(db_path)
    result = None
    progress = None
    if os.path.exists(paths["result"]):
        try:
            with open(paths["result"], encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            result = None
    if os.path.exists(paths["progress"]):
        try:
            with open(paths["progress"], encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            progress = None
    if progress is not None and (result is None or progress.get("run_id") != result.get("run_id")):
        progress["state"] = "running"
        return progress
    if result is not None:
        result["state"] = "completed"
        return result
    return None


def _read_event_log(db_path: str, run_id: Optional[str] = None) -> list:
    """Read the durable batch event log (read-only); ``[]`` when absent or unreadable.

    When ``run_id`` is given, only events for that run are returned, so a running view never
    inherits earlier-run activity.
    """
    paths = batch_paths(db_path)
    events = []
    if os.path.exists(paths["events"]):
        try:
            with open(paths["events"], encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if run_id is not None and e.get("run_id") != run_id:
                        continue
                    events.append(e)
        except Exception:
            pass
    return events


def _running_path_states(db_path: str, run_id: Optional[str] = None) -> list:
    """The four model/strategy paths with per-path state derived from *committed* observations.

    Only committed OBSERVATION events (the authoritative source) are counted; nothing is
    invented. Each path's counts are the completed observations for the selected run, so a
    completed observation remains visible even when its later display marker was never written,
    and a later run never inherits an earlier run's activity.
    """
    if run_id is None:
        state = read_batch_state(db_path)
        run_id = state.get("run_id") if state else None
    per_path = {f"{m}/{s}": {"dispatched": 0, "bypassed": 0, "blocked": 0}
                for m in MODELS for s in STRATEGIES}
    for e in _read_event_log(db_path, run_id=run_id):
        if e.get("phase") != "OBSERVATION":
            continue
        path = e.get("path")
        if path not in per_path:
            continue
        payload = e.get("payload", {})
        if payload.get("state") == "FAILED":
            per_path[path]["blocked"] += 1
        elif (payload.get("model_calls") or 0) > 0:
            per_path[path]["dispatched"] += 1
        else:
            per_path[path]["bypassed"] += 1
    return [{"path": f"{m}/{s}", "model": m, "strategy": s, **per_path[f"{m}/{s}"]}
            for m in MODELS for s in STRATEGIES]


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
        return templates.TemplateResponse(
            request,
            "cases.html",
            {"rows": rows, "state": state, "amd": AMD_EVIDENCE,
             "s4e15_amd": amd_demonstration()},
        )

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
                "s4e15_amd": amd_demonstration(),
                "evolus": EVOLUS_EVIDENCE,
                "b51_observed": B5_1_OBSERVED_OFFLINE,
                "b51": B5_1_COUNTERFACTUAL,
                "gemma": gemma_demonstration(),
                "gemma_history": GEMMA_HISTORY,
                "gemma_source_note": GEMMA_SOURCE_CONSISTENCY_NOTE,
            },
        )

    @app.get("/experiment", response_class=HTMLResponse)
    def experiment_page(request: Request):
        return templates.TemplateResponse(
            request,
            "experiment.html",
            {"amd": AMD_EVIDENCE,
             "s4e15_amd": amd_demonstration(),
             "gemma": gemma_demonstration(),
             "gemma_history": GEMMA_HISTORY,
             "gemma_source_note": GEMMA_SOURCE_CONSISTENCY_NOTE,
             "historical": load_public_experiment()},
        )

    # --- S4E.16 batch workflow (operational + comparison views) ------------------------

    # Execution is separated from observation: the module-level ``initiate_batch`` runs the
    # batch and persists durable state; the GET views below only READ that durable state and
    # never execute the batch or mutate the journal.
    def _read_batch_state():
        return read_batch_state(db_path)

    def _path_states(result: dict) -> dict:
        """invoice_id -> {path: {phase, route, model_calls, outcome}} derived from the *validated*
        committed observations, not the auxiliary display markers."""
        states: dict = {}
        for o in result.get("durable_observations", []):
            invoice_id = o.get("invoice_id")
            path = f"{o.get('model_id')}/{o.get('strategy')}"
            model_calls = o.get("model_calls") or 0
            if o.get("state") == "FAILED":
                phase = "BLOCKED"
            elif model_calls == 0:
                phase = "BYPASS"
            else:
                phase = "REQUEST"
            states.setdefault(invoice_id, {})[path] = {
                "phase": phase,
                "route": o.get("route"),
                "model_calls": model_calls,
                "outcome": o.get("advisory_outcome"),
                "block_reason": None,
            }
        return states

    def _running(request: Request, state: dict):
        return templates.TemplateResponse(
            request, "batch_running.html",
            {"state": state, "paths": _running_path_states(db_path)})

    @app.get("/batch", response_class=HTMLResponse)
    def batch_operational(request: Request):
        result = _read_batch_state()
        if result is None:
            return templates.TemplateResponse(request, "batch_empty.html", {})
        if result.get("state") == "running":
            return _running(request, result)
        return templates.TemplateResponse(
            request, "batch.html",
            {
                "result": result,
                "paths": result["paths"],
                "invoices": result["invoices"],
                "operational": result["operational"],
                "progress": result["progress"],
                "measurements": result["measurements"],
            },
        )

    @app.get("/comparison", response_class=HTMLResponse)
    def batch_comparison(request: Request):
        result = _read_batch_state()
        if result is None:
            return templates.TemplateResponse(request, "batch_empty.html", {})
        if result.get("state") == "running":
            return _running(request, result)
        states = _path_states(result)
        path_keys = [f"{m}/{s}" for m in MODELS for s in STRATEGIES]
        return templates.TemplateResponse(
            request, "comparison.html",
            {
                "result": result,
                "paths": result["paths"],
                "invoices": result["invoices"],
                "path_keys": path_keys,
                "states": states,
                "operational": result["operational"],
                "measurements": result["measurements"],
            },
        )

    @app.get("/batch/{invoice_id}", response_class=HTMLResponse)
    def batch_invoice(request: Request, invoice_id: str):
        result = _read_batch_state()
        if result is None:
            return templates.TemplateResponse(request, "batch_empty.html", {})
        if result.get("state") == "running":
            return _running(request, result)
        invoice = next((i for i in result["invoices"] if i["invoice_id"] == invoice_id), None)
        if invoice is None:
            raise HTTPException(status_code=404, detail="unknown invoice")
        states = _path_states(result)
        op = next((o for o in result["operational"] if o["invoice_id"] == invoice_id), None)
        return templates.TemplateResponse(
            request, "batch_invoice.html",
            {
                "invoice": invoice,
                "states": states.get(invoice_id, {}),
                "path_keys": [f"{m}/{s}" for m in MODELS for s in STRATEGIES],
                "operational": op,
            },
        )

    # --- S4E.25 interactive synthetic invoice intake --------------------------------
    # A narrowly bounded POST endpoint evaluates a synthetic invoice through the accepted
    # deterministic pipeline. It never writes the journal, never contacts a provider, and
    # never grants execution authority (execution_authority stays NONE). Visitor
    # submissions are evaluated in-memory and never persisted.

    SCENARIO_PREFILL = {
        "exact": {"invoice_id": "INV-700", "supplier_id": "SUP-SYN-100",
                  "po_reference": "PO-SYN-100", "amount": "1000.00",
                  "label": "Exact match"},
        "discrepancy": {"invoice_id": "INV-701", "supplier_id": "SUP-SYN-100",
                        "po_reference": "PO-SYN-100", "amount": "1100.00",
                        "label": "Amount discrepancy"},
        "missing_po": {"invoice_id": "INV-702", "supplier_id": "SUP-SYN-100",
                       "po_reference": "PO-MISSING", "amount": "1000.00",
                       "label": "Missing purchase order"},
    }

    @app.get("/try", response_class=HTMLResponse)
    def try_intake(request: Request, scenario: str = ""):
        prefill = SCENARIO_PREFILL.get(scenario, {})
        values = {
            "invoice_id": prefill.get("invoice_id", ""),
            "supplier_id": prefill.get("supplier_id", ""),
            "po_reference": prefill.get("po_reference", ""),
            "amount": prefill.get("amount", ""),
        }
        return templates.TemplateResponse(request, "try.html", {
            "scenarios": SCENARIO_PREFILL,
            "active_scenario": scenario,
            "values": values,
            "error": None,
            "result": None,
        })

    MAX_BODY_BYTES = 8192
    MAX_FORM_FIELDS = 8

    @app.post("/evaluate", response_class=HTMLResponse)
    async def evaluate_invoice(request: Request):
        # Support only URL-encoded form; reject anything else before reading the body.
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/x-www-form-urlencoded":
            raise HTTPException(status_code=415, detail="unsupported content type")

        # Bound the DECLARED length first (cheap), then the ACTUAL streamed body, so a
        # misleading Content-Length cannot smuggle an oversized body past the limit.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            if not content_length.isdigit():
                raise HTTPException(status_code=400, detail="invalid content-length")
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request too large")

        body = bytearray()
        try:
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request too large")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="malformed request body") from None

        try:
            encoded = body.decode("utf-8")
            hex_digits = "0123456789abcdefABCDEF"
            for offset, char in enumerate(encoded):
                if char == "%" and (
                    offset + 2 >= len(encoded)
                    or any(
                        digit not in hex_digits
                        for digit in encoded[offset + 1:offset + 3]
                    )
                ):
                    raise ValueError("invalid percent escape")
            pairs = parse_qsl(
                encoded, keep_blank_values=True,
                strict_parsing=True, max_num_fields=MAX_FORM_FIELDS,
                errors="strict",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="malformed form") from exc

        form = {k: v for k, v in pairs}
        if len(form) != len(pairs):
            raise HTTPException(status_code=400, detail="duplicate fields")
        allowed_fields = {"invoice_id", "supplier_id", "po_reference", "amount"}
        if set(form) - allowed_fields:
            raise HTTPException(status_code=400, detail="unexpected form fields")

        values = {
            "invoice_id": form.get("invoice_id", ""),
            "supplier_id": form.get("supplier_id", ""),
            "po_reference": form.get("po_reference", ""),
            "amount": form.get("amount", ""),
        }
        try:
            case = build_case(values["invoice_id"], values["supplier_id"],
                              values["po_reference"], values["amount"])
            result = evaluate_intake(case)
        except IntakeError as exc:
            return templates.TemplateResponse(request, "try.html", {
                "scenarios": SCENARIO_PREFILL, "active_scenario": "",
                "values": values, "error": str(exc), "result": None,
            }, status_code=400)
        return templates.TemplateResponse(request, "try.html", {
            "scenarios": SCENARIO_PREFILL, "active_scenario": "",
            "values": values, "error": None, "result": result,
        })

    return app


app = create_app()
