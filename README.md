# BOUNDARY

A synthetic invoice-exception workflow demonstrating **deterministic-first governance**
for financial document processing, built for the AMD Developer Hackathon ACT III.

**Team:** Boundary Systems

BOUNDARY resolves routine invoice/purchase-order cases with deterministic logic first,
invokes model inference only for a bounded, explicitly identifiable category, and keeps
model output strictly advisory — no AI result ever grants approval or execution authority.

> All financial records, suppliers, and transactions are fictional. The offline web
> demonstration performs no integrations and handles no real accounts; there are no real
> payments or financial-system actions. Separate AMD and Evolus integration *tests* were
> conducted using synthetic data only.

## Why deterministic-first governance matters

A naive "LLM-first" workflow sends every case to a model and hopes the output is right.
BOUNDARY inverts this:

1. **Deterministic resolution first** — exact matches are resolved with zero model calls.
2. **Selective inference** — a model is consulted only for a nonzero-discrepancy case with
   genuine evidence text, and only after deterministic checks pass.
3. **Authority separation** — model output is advisory (`execution_authority = NONE`). In the
   demo, journal commits are performed by the deterministic pipeline after the guard's approval
   check (a simulated approval — no separate human-authorization gate is implemented), and every
   commit is independently verified against the source case.

This keeps the blast radius of a model error small and makes every decision traceable.

## Architecture

The pipeline is a sequence of narrow, independently tested stages:

```
CaseInput
  -> classifier  (deterministic exact-match proposal; reason codes; authority NONE)
  -> guard       (deterministic APPROVE/HOLD permission check)
  -> journal     (SQLite; simulated approval record or hold)
  -> verifier    (read-only; re-classifies and re-reads the journal)
```

Separate modules add selective routing (`router`, `inference`, `amd_transport`) for the
B3 advisory path and document extraction (`evolus`, `evolus_http`) for the B4 path. The
web application (`app.py`) is **offline** (no live AMD/Evolus transports) and exposes a
bounded interactive intake alongside read-only browsing: `GET /try` and `POST /evaluate`
run the deterministic pipeline on a visitor's synthetic invoice in memory — never
persisted, never contacting a provider, and never granting execution authority.

## The three demo cases

| Case | Facts | Classifier / guard | Journal | Verification |
|------|-------|--------------------|---------|--------------|
| CASE-1 | Exact $50 match | APPROVE | committed (simulated) | VERIFIED |
| CASE-2 | $5 discrepancy | HOLD | no commit | NOT_ELIGIBLE |
| CASE-3 | PO unavailable | HOLD | no commit | NOT_ELIGIBLE |

- **APPROVE / committed** is a *simulated* journal record — not a payment or a real human
  authorization.
- **VERIFIED** means the journal record matches the source case and execution constraints —
  not invoice authenticity or settlement.
- **NOT_ELIGIBLE** means ineligible for the simulated exact-match approval path — not a
  certification of the held case.

## Public routes and offline vs. recorded inference

The browser application is **offline**: it performs no AMD inference and no Evolus
extraction, and requires no credentials.

- **`/` (Cases)** and **`/cases/<id>`** — read-only traces of the three demo cases.
- **`/try`** and **`POST /evaluate`** — a bounded interactive intake: a visitor supplies a
  synthetic invoice, and BOUNDARY runs the deterministic pipeline in memory. Nothing is
  stored, no provider is contacted, and execution authority stays `NONE`.
- **`/batch`**, **`/comparison`**, and **`/batch/<invoice_id>`** — the six-invoice synthetic
  batch and four-path (model × strategy) comparison. Model/strategy paths use offline
  simulated adapters; their decision-call counts are **not** live AMD API requests.
- **`/experiment`** and **`/evidence`** — the recorded evidence: the S4E.15 two-model AMD
  comparison (16 genuine requests across DeepSeek and Google Gemma), the S4E.10 offline
  720-observation benchmark, the separate S4E.5 DeepSeek pilot, the S4E.17 integrated
  Evolus-to-AMD trial, and the **October 16 Gemma integration completion** — a separate
  post-restoration experiment in which both Gemma paths (LLM-first and BOUNDARY) succeeded
  through the same validated Evolus extraction. See the **Gemma integration completion**
  section on `/evidence` and `/experiment`.

The AMD and Evolus results shown on **Evidence & Performance** (`/evidence`) and
**Experiment** (`/experiment`) are **separately recorded live integration tests**, clearly
labeled `RECORDED LIVE TEST — NOT RE-RUN BY THIS PAGE`. The offline simulated adapters used
by the batch/comparison views are a distinct category from those genuine recorded requests.

## Running locally

Requirements: Python 3.11+ and the dependencies in `requirements.txt`.

```bash
pip install -r requirements.txt

# Serve the offline demo (seeds the synthetic journal, then serves on PORT, default 8000)
python -m boundary.serve

# Or, run the demo pipeline directly and serve with uvicorn
python -m boundary.demo --db boundary.db
uvicorn boundary.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` for the case list, `/try` for the interactive synthetic
intake, `/batch` and `/comparison` for the six-invoice batch and four-path comparison,
`/experiment` and `/evidence` for the recorded evidence, and any `/cases/<id>` for a trace.

### Tests

```bash
python -m pytest
```

## Railway deployment

- **Start command** (Procfile): `web: python -m boundary.serve`
- **Python version** (`.python-version`): `3.11`
- `boundary/serve.py` seeds the demo journal idempotently and serves on Railway's `PORT`.
- **No credentials required** — the offline demo reads only `PORT` (and optionally
  `BOUNDARY_DB_PATH`).
- **Persistence**: the SQLite journal lives in the container's ephemeral filesystem and is
  re-seeded on each deploy; no external database is required.
- **Health**: `/health` reports `journal: ready` and truthfully reports
  `amd_connected: false`, `evolus_connected: false`.

### Deployment limitations

- Ephemeral, per-deploy journal (no durable persistence).
- No horizontal scaling guarantees (a single in-memory/on-disk SQLite journal).
- The live AMD/Evolus integrations are not wired into the deployed app; they were run
  as separate, recorded integration tests.

## Non-production status and synthetic-data restrictions

- All cases, suppliers, invoices, and purchase orders are **synthetic**.
- The offline web demonstration performs no integrations and requires no credentials; the
  separately-recorded AMD and Evolus integration tests used synthetic data only.
- No real financial systems, payments, financial-system actions, or customer financial records
  are involved.
- This is a competition demonstration, not production software.

## Disclosure

This submission was developed with AI coding assistance under human supervision. Work
proceeded in explicitly
authorized, independently reviewed phases (B0–B6); each phase was accepted by an
independent reviewer before the next began. Model output within the system is advisory
only and never grants execution or financial authority.

No claim is made here about code predating the competition window.
