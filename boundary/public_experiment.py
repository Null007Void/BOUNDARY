"""Hash-verified historical S4E.15 evidence for the public application."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parent / "evidence_data" / "s4e20"

_EXPECTED_SHA256 = {
    "two_model_evidence.json":
        "14f3c51f46a7f3f2fb10a79fbd1265fb6e3fabd209c15d12cf346d9b38e9735f",
    "replay_data.json":
        "5b0fbf3ef92590630f2c2eb5719030aa6c0e001c9d4359a05c6b99339b449972",
    "integrated_trial_summary.json":
        "c8ebe578ef1e406ea15fc89ae216faee5fe6cc45039ae7cb25c72c87b8a7fbc4",
}

def _read_pinned_json(filename: str) -> dict:
    data = (_DATA_ROOT / filename).read_bytes()
    # Canonicalize line endings to LF before hashing, so a Windows (CRLF) checkout and a
    # Linux/GitHub (LF) checkout verify identically — without weakening integrity checks:
    # any substantive content change still changes the canonical bytes and fails closed.
    text = data.decode("utf-8").replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    canonical = text.encode("utf-8")

    if hashlib.sha256(canonical).hexdigest() != _EXPECTED_SHA256[filename]:
        raise ValueError(f"Historical evidence checksum mismatch: {filename}")
    return json.loads(canonical.decode("utf-8"))

def load_public_experiment() -> dict:
    """Load recorded evidence without executing inference or writing state."""
    experiment = _read_pinned_json("two_model_evidence.json")
    replay = _read_pinned_json("replay_data.json")
    integration = _read_pinned_json("integrated_trial_summary.json")

    if (experiment.get("experiment_id") != "s4e15-live"
            or replay.get("experiment_id") != "s4e15-live"
            or not all(replay.get(k) is True for k in (
                "read_only", "makes_no_provider_calls",
                "is_historical_record"
            ))):
        raise ValueError("Historical replay contract invalid")

    counts = experiment["record_counts"]
    if (len(replay["cases"]) != 6
            or counts["total_call_attempts"] != 16
            or counts["zero_call_observations"] != 8):
        raise ValueError("Historical experiment counts invalid")

    paths = []
    for model_id in ("DeepSeek-V4-Flash", "Gemma-4-31B-IT"):
        for strategy in ("LLM_FIRST", "BOUNDARY"):
            key = f"{model_id} / {strategy}"
            path = experiment["paths"][key]
            summary = path["summary"]

            if (summary["model_id"] != model_id
                    or summary["strategy"] != strategy
                    or summary["requests"] != len(path["records"])):
                raise ValueError(f"Historical path invalid: {key}")

            paths.append(summary)


    # Separate historical mock workload from genuine AMD inference.
    offline_path = _DATA_ROOT.parent / "s4e10_offline_summary.json"
    offline = json.loads(offline_path.read_text(encoding="utf-8"))

    expected = {
        "source_sha256": "d4717404671d37ffad495358b3a6ff711361b58fc933b54f1e75918932ea1f49",
        "classification": "MOCK / OBSERVED_OFFLINE",
        "observations": 720,
        "live_model_requests": 0,
        "integrity_passed": 720,
        "task_success": 480,
        "task_failed": 240,
        "authority_violations": 0,
    }
    if any(offline.get(k) != v for k, v in expected.items()):
        raise ValueError("Offline evidence contract mismatch")

    return {
        "experiment_id": experiment["experiment_id"],
        "offline_evaluation": offline,
        "fingerprint": experiment["workload_fingerprint"],
        "paths": paths,
        "cases": replay["cases"],
        "zero_call_count": 8,
        "zero_call_note": replay["zero_call_observations"]["note"],
        "limitations": experiment["limitations"],
        "integration": integration,
        "is_historical_record": True,
        "makes_no_provider_calls": True,
    }


def gemma_demonstration() -> dict:
    """Derive the Gemma metrics from the hash-verified S4E.15 records (no duplicate hardcoding)."""
    experiment = _read_pinned_json("two_model_evidence.json")
    llm = experiment["paths"]["Gemma-4-31B-IT / LLM_FIRST"]["summary"]
    boundary = experiment["paths"]["Gemma-4-31B-IT / BOUNDARY"]["summary"]
    return {
        "status": "DEMONSTRATED — RECORDED LIVE",
        "provenance": "RECORDED LIVE TEST",
        "model_id": "Gemma-4-31B-IT",
        "endpoint_model_id": "self-dploy/Gemma-4-31B-IT",
        "model_verified": "unverified_mismatch",
        "authority": "NONE",
        "llm_first": {
            "requests": llm["requests"],
            "tokens": llm["total_tokens"],
            "source_consistent": llm["source_consistent"],
            "source_inconsistent": llm["source_inconsistent"],
        },
        "boundary": {
            "requests": boundary["requests"],
            "tokens": boundary["total_tokens"],
            "source_consistent": boundary["source_consistent"],
            "source_inconsistent": boundary["source_inconsistent"],
        },
        "total_requests": llm["requests"] + boundary["requests"],
        "source_consistent_overall": llm["source_consistent"] + boundary["source_consistent"],
    }


# Transparency disclosure: the actual Gemma history, in order.
GEMMA_HISTORY = [
    "The first October 13, 2026 Gemma smoke attempt produced code-fenced JSON that "
    "failed structured-output parsing.",
    "Gemma-specific fenced-JSON normalization followed by strict parsing addressed that problem.",
    "Subsequent live tests successfully demonstrated Gemma inference (eight confirmed requests).",
    "Later, Gemma disappeared from the authenticated AMD Token Factory model catalog, "
    "preventing additional requests. This is an observed availability limitation — not a "
    "claim that AMD permanently discontinued Gemma.",
    "The separate Evolus-to-AMD integrated trial remains incomplete: DeepSeek succeeded on "
    "two paths, Gemma LLM-first became unavailable, and Gemma BOUNDARY was not dispatched.",
    "The exact underlying serving-model identity has not been independently verified "
    "(the endpoint reported \"self-dploy/Gemma-4-31B-IT\", classified \"unverified_mismatch\").",
]

GEMMA_SOURCE_CONSISTENCY_NOTE = (
    "The single source-inconsistent Gemma record is the absent-purchase-order evaluator "
    "convention (the invoice amount versus the evaluator's convention of zero), not a "
    "measured interpretation error."
)


def amd_demonstration() -> dict:
    """Derive the complete S4E.15 AMD summary (DeepSeek + Gemma) from the hash-verified records."""
    experiment = _read_pinned_json("two_model_evidence.json")

    def _summary(model, strategy):
        return experiment["paths"][f"{model} / {strategy}"]["summary"]

    ds_lf = _summary("DeepSeek-V4-Flash", "LLM_FIRST")
    ds_b = _summary("DeepSeek-V4-Flash", "BOUNDARY")
    ge_lf = _summary("Gemma-4-31B-IT", "LLM_FIRST")
    ge_b = _summary("Gemma-4-31B-IT", "BOUNDARY")

    deepseek_requests = ds_lf["requests"] + ds_b["requests"]
    deepseek_tokens = ds_lf["total_tokens"] + ds_b["total_tokens"]
    gemma_requests = ge_lf["requests"] + ge_b["requests"]
    gemma_tokens = ge_lf["total_tokens"] + ge_b["total_tokens"]
    llm_requests = ds_lf["requests"] + ge_lf["requests"]
    llm_tokens = ds_lf["total_tokens"] + ge_lf["total_tokens"]
    boundary_requests = ds_b["requests"] + ge_b["requests"]
    boundary_tokens = ds_b["total_tokens"] + ge_b["total_tokens"]

    return {
        "status": "DEMONSTRATED — RECORDED LIVE",
        "provenance": "RECORDED LIVE TEST",
        "authority": "NONE",
        "deepseek": {"requests": deepseek_requests, "tokens": deepseek_tokens},
        "gemma": {"requests": gemma_requests, "tokens": gemma_tokens},
        "combined": {"requests": deepseek_requests + gemma_requests,
                     "tokens": deepseek_tokens + gemma_tokens},
        "llm_first": {"requests": llm_requests, "tokens": llm_tokens},
        "boundary": {"requests": boundary_requests, "tokens": boundary_tokens},
        "request_reduction_percent": round((llm_requests - boundary_requests) / llm_requests * 100, 1),
        "token_reduction_percent": round((llm_tokens - boundary_tokens) / llm_tokens * 100, 1),
    }
