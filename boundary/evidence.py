"""Curated, synthetic, public-intended evidence for the Evidence & Performance page.

These values are RECORDED results from independently reviewed live integration tests
(S4E.5 AMD pilot, B4 Evolus extraction) and the offline B5.1 benchmark. They are NOT
re-run by the web application, which is fully offline. No credentials, API keys, secrets,
private URLs, or personal account identifiers are present.
"""

# --- AMD — S4E.5 controlled pilot, recorded October 13 ------------------------------

AMD_EVIDENCE = {
    "provenance": "RECORDED LIVE TEST",
    "status": "MEASURED",
    "recorded_at": "October 13, 2026",
    "model_id": "DeepSeek-V4-Flash",
    "endpoint_model_id": "self-dploy/DeepSeek-V4-Flash",
    "model_verified": "unverified_mismatch",
    "strategy_a": {"label": "LLM-first", "requests": 6, "tokens": 1314},
    "strategy_b": {"label": "BOUNDARY deterministic-first", "requests": 2, "tokens": 473},
    "confirmed_responses": 8,
    "request_reduction": "66.7%",
    "token_reduction": "64.0%",
    "note": (
        "Controlled eight-request pilot across three synthetic scenarios (two repetitions). "
        "Strategy A (LLM-first) made six requests (1314 tokens); Strategy B (BOUNDARY "
        "deterministic-first) made two (473 tokens). Request reduction is measured; token "
        "reduction is measured for this scenario set. No per-request latency improvement and "
        "no measured cost savings are claimed."
    ),
}

# --- Evolus — B4, recorded October 13 -------------------------------------------------

EVOLUS_EVIDENCE = {
    "provenance": "RECORDED LIVE TEST",
    "status": "DEMONSTRATED",
    "recorded_at": "October 13, 2026",
    "document_type": "INVOICE",
    "invoice_id": "INV-2048",
    "supplier_id": "SUP-9001",
    "po_reference": "PO-77",
    "stated_total": "USD 1,250.00",
    "advisory_analysis": "present",
    "evidence_status": "unverified",
    "authority": "NONE",
    "note": (
        "A genuine API-driven synthetic document extraction completed in a separate "
        "live integration test; it was not jointly executed with AMD by this demo."
    ),
}

# --- Offline B5.1 benchmark — OBSERVED_OFFLINE and COUNTERFACTUAL --------------------

B5_1_OBSERVED_OFFLINE = {
    "label": "OBSERVED_OFFLINE",
    "categories": 6,
    "observations": 18,
    "repetitions_per_category": 3,
    "note": (
        "Six synthetic routing scenarios, 18 offline observations (three repetitions "
        "per category). These are actual offline routing measurements."
    ),
}

B5_1_COUNTERFACTUAL = {
    "label": "COUNTERFACTUAL",
    "categories": 6,
    "call_opportunities_avoided": 2,
    "note": (
        "A hypothetical LLM-first comparison: two of six call opportunities avoided by "
        "deterministic-first routing. This is arithmetic — NOT measured token, latency, "
        "or cost savings."
    ),
}
