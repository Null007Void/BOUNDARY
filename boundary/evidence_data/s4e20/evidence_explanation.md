# BOUNDARY — Recorded Two-Model Evidence (S4E.15)

This package presents the accepted S4E.15 live two-model experiment as recorded, public-safe evidence. It is a historical record, not a fresh live run.

## Four comparison paths

- DeepSeek-V4-Flash / LLM-first
- DeepSeek-V4-Flash / BOUNDARY
- Gemma-4-31B-IT / LLM-first
- Gemma-4-31B-IT / BOUNDARY

## What was measured

Six synthetic development cases (two deterministic-match, two financial-discrepancy, two missing-PO), one repetition, 16 model requests (6 LLM-first + 2 BOUNDARY per model), all confirmed and parsed, with `execution_authority = NONE` throughout.

## Important limitations

- The three `source_inconsistent` cells are the absent-PO discrepancy convention (invoice amount vs. the evaluator's convention of zero), not measured extraction errors.
- Both serving identifiers are namespaced (`self-dploy/...`) and differ from the configured names, recorded as `unverified_mismatch`.
- Recorded playback makes **zero provider calls** and is **never** presented as a fresh live run.
