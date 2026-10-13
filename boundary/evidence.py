"""Curated, synthetic, public-intended evidence for the Evidence & Performance page.

These values are RECORDED results from independently reviewed live integration
tests (B5.2 AMD inference, B4 Evolus extraction) and the offline B5.1 benchmark.
They are NOT re-run by the web application, which is fully offline. No credentials,
API keys, secrets, private URLs, or personal account identifiers are present.
"""

# --- AMD — B5.2, recorded October 13 -------------------------------------------------

AMD_EVIDENCE = {
    "provenance": "RECORDED LIVE TEST",
    "recorded_at": "October 13, 2026",
    "model_id": "DeepSeek-V4-Flash",
    "scenarios": [
        {
            "name": "Exact match",
            "route": "DETERMINISTIC",
            "amd_requests": 0,
            "tokens": None,
            "latency_ms": None,
            "advisory": None,
            "source_consistent": None,
        },
        {
            "name": "Financial discrepancy",
            "route": "INFERENCE",
            "amd_requests": 1,
            "tokens": {"prompt": 127, "completion": 118, "total": 245},
            "latency_ms": 3974,
            "advisory": "SUCCESS",
            "source_consistent": True,
        },
        {
            "name": "Missing PO",
            "route": "HELD",
            "amd_requests": 0,
            "tokens": None,
            "latency_ms": None,
            "advisory": None,
            "source_consistent": None,
        },
    ],
    "note": (
        "Three distinct synthetic scenarios, one observation each. No averages, "
        "typical latency, or measured financial savings are claimed."
    ),
}

# --- Evolus — B4, recorded October 13 -------------------------------------------------

EVOLUS_EVIDENCE = {
    "provenance": "RECORDED LIVE TEST",
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
