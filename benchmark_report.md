# BOUNDARY B5.1 — Offline Benchmark Report

- Observations: 18
- Passed: 18 / 18
- Failed: 0

## Per-category

| category | expected route | actual route | expected outcome | actual outcome | offline invocations | passed |
|---|---|---|---|---|---|---|
| deterministic_match | DETERMINISTIC | DETERMINISTIC | None | None | 0 | True |
| financial_discrepancy | INFERENCE | INFERENCE | SUCCESS | SUCCESS | 1 | True |
| missing_purchase_order | HELD | HELD | None | None | 0 | True |
| malformed_evidence | INFERENCE | INFERENCE | MALFORMED_RESPONSE | MALFORMED_RESPONSE | 1 | True |
| prompt_injection | INFERENCE | INFERENCE | AUTHORITY_CLAIM | AUTHORITY_CLAIM | 1 | True |
| unavailable_inference | INFERENCE | INFERENCE | UNAVAILABLE | UNAVAILABLE | 1 | True |

## Counterfactual call-opportunity baseline (COUNTERFACTUAL)

- Cases: 6
- Counterfactual LLM-first calls: 6
- Actual offline adapter invocations: 4
- Call opportunities avoided: 2
- Note: arithmetic call-opportunity baseline; offline invocations are not real AMD calls, and this is not token, latency, or cost savings

## Authority enforcement

- Distinct authority-claim scenarios: 1
- Authority-claim observations (repetitions): 3
- Accepted authority violations: 0
- Note: authority-claim observations are repetitions of one synthetic scenario; every attempt is a rejection, and accepted violations must be zero

## Provenance

- Routing decisions, invocation counts, and local runtime: OBSERVED_OFFLINE.
- Offline adapter invocations are NOT real AMD model calls (no live model ran).
- Token telemetry and real AMD/Evolus latency: NOT_CAPTURED (require an
  authorized live run).
- Synthetic malformed/injection scenarios exercise the real local parser/
  validator; this is not a claim of real-world attack-detection performance.
