"""Hash-verified historical S4E.15 evidence for the public application."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parent / "evidence_data" / "s4e20"

_EXPECTED_SHA256 = {
    "two_model_evidence.json":
        "38fd47db6263cee80e4e5ee873ecc09471ba0af4d39e76edcbf832aa56d2c6cb",
    "replay_data.json":
        "826fa8ba6ac9ba181887b2edc0e91356a09f54d7f1eebd377bb44166f5b5a5af",
    "integrated_trial_summary.json":
        "e20d2d97822b2a6a0a6002069a2011a8ddb183c838bca3a4677b26e526a0cf72",
}

def _read_pinned_json(filename: str) -> dict:
    data = (_DATA_ROOT / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != _EXPECTED_SHA256[filename]:
        raise ValueError(f"Historical evidence checksum mismatch: {filename}")
    return json.loads(data.decode("utf-8"))

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
