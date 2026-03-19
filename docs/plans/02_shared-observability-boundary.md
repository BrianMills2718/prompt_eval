# Plan 02: Shared Observability Boundary

**Status:** ✅ Complete
**Type:** implementation
**Priority:** Highest
**Blocked By:** None
**Blocks:** prompt_eval / llm_client architectural convergence

---

## Gap

**Current:** historically `prompt_eval` could run experiments and persist local
JSON results, but it had no authoritative shared representation in
`llm_client` observability.

**Target:** `prompt_eval` emits authoritative shared runs/items/aggregates into
`llm_client`, and can reconstruct a prompt-eval result family from that shared
backend without depending on local JSON files.

**Why:** prompt experiments should participate in the same provenance, cost
tracking, prompt identity, and cross-project analysis surface as the rest of
the ecosystem.

---

## References Reviewed

- `docs/adr/0001-llm-client-substrate-boundary.md`
- `docs/adr/0002-prompt-eval-run-family-mapping.md`
- `docs/UNCERTAINTIES.md`
- `prompt_eval/runner.py`
- `prompt_eval/observability.py`
- `prompt_eval/query.py`
- `tests/test_runner.py`
- `tests/test_query.py`

---

## Files Affected

- `prompt_eval/runner.py`
- `prompt_eval/observability.py`
- `prompt_eval/query.py`
- `prompt_eval/experiment.py`
- `prompt_eval/optimize.py`
- `tests/test_runner.py`
- `tests/test_query.py`
- `docs/UNCERTAINTIES.md`
- `docs/adr/0002-prompt-eval-run-family-mapping.md`

---

## Plan

### Steps

1. Define one canonical mapping from prompt_eval experiments to shared
   `llm_client` runs, items, and aggregates.
2. Dual-write experiment execution into the shared backend by default.
3. Reconstruct `EvalResult` families from the shared backend by `execution_id`.
4. Persist corpus-level evaluator outputs as first-class shared aggregates.
5. Verify round-trip behavior with focused tests.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_runner.py` | `test_emits_llm_client_runs_per_variant_and_replicate` | One shared run per `PromptVariant x replicate` |
| `tests/test_runner.py` | `test_prompt_ref_is_recorded_in_run_provenance` | Prompt identity is preserved in shared provenance |
| `tests/test_query.py` | `test_reconstructs_eval_result_from_shared_runs` | Read-side reconstruction returns coherent `EvalResult` |
| `tests/test_query.py` | `test_reconstructs_corpus_metrics_from_shared_aggregates` | Corpus aggregates survive shared round-trip |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_runner.py tests/test_query.py -q` | Core shared-boundary behavior remains verified |
| `pytest tests/ -q` | Full package behavior remains intact |

---

## Acceptance Criteria

- [x] Experiments emit shared `llm_client` runs/items/aggregates by default
- [x] `EvalResult.execution_id` is recorded and reusable
- [x] `load_result_from_observability()` reconstructs a coherent result family
- [x] Corpus-level evaluator outputs round-trip through shared aggregates
- [x] Local JSON persistence remains available as compatibility/export path

---

## Notes

This program is complete. The remaining open questions are not about the shared
observability boundary itself; they are about prompt asset policy and package
scope, tracked in Plan 03 and `docs/UNCERTAINTIES.md`.
