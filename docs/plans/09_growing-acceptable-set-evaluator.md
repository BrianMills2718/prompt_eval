# Plan 09: Growing Acceptable Set Evaluator

**Status:** 📋 Planned
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** multi-correct-answer evaluation reuse and downstream adoption

---

## Gap

**Current:** there is a local prototype for a persistent acceptable-alternatives
cache, but it is not yet part of the canonical `prompt_eval` program. The
prototype is synchronous, uses an ad hoc judge `dict`, and is not yet proven
through the real async `run_experiment()` evaluator path. Downstream planning
in `onto-canon6` already assumes this feature exists.

**Target:** promote the pattern into a real `prompt_eval` feature:

1. an async-capable evaluator wrapper that fits `run_experiment()`,
2. a typed judge-decision contract,
3. local SQLite persistence with explicit dataset/dimension scoping,
4. manual inspection/override support,
5. focused tests proving both the cache behavior and runner integration.

**Why:** this is real prompt-centric evaluation infrastructure, not a side
experiment. Without formalization, downstream repos are planning against an API
that is still local prototype code.

---

## References Reviewed

- `prompt_eval/golden_set.py` (local prototype)
- `tests/test_golden_set.py` (local prototype tests)
- `prompt_eval/runner.py`
- `prompt_eval/evaluators.py`
- `docs/adr/0004-growing-acceptable-set-evaluator.md`
- `onto-canon6/docs/adr/0019-adopt-ancestor-aware-evaluation-with-growing-acceptable-sets.md`
- `onto-canon6/src/onto_canon6/evaluation/ancestor_evaluator.py`

---

## Files Affected

- `prompt_eval/golden_set.py`
- `prompt_eval/__init__.py`
- `tests/test_golden_set.py`
- `tests/test_runner.py`
- `README.md`
- `docs/API_REFERENCE.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/CLAUDE.md`
- `ISSUES.md`

---

## Plan

### Steps

1. Keep the architecture explicit through ADR 0004:
   - prompt-centric evaluation helper,
   - async-capable contract,
   - local SQLite sidecar in v1,
   - typed judge decision.
2. Replace the ad hoc synchronous prototype surface with a real evaluator
   wrapper that supports async judge calls.
3. Add a typed judge-decision model and fail loudly on malformed results.
4. Prove the feature through both direct cache tests and integration with
   `run_experiment()`.
5. Only after the behavior is proven, make the public export intentional and
   update README/API docs.

### Thin Slice Status

- [ ] Phase 1: align the prototype surface with the ADR contract
- [ ] Phase 2: add typed judge-decision validation and fail-loud behavior
- [ ] Phase 3: prove async integration through `run_experiment()`
- [ ] Phase 4: finalize public export and docs

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_golden_set.py` | `test_malformed_judge_decision_fails_loudly` | typed judge contract is enforced |
| `tests/test_golden_set.py` | `test_async_judge_is_supported` | async judge path works |
| `tests/test_runner.py` | new integration coverage | acceptable-set evaluator works through `run_experiment()` |
| `tests/test_golden_set.py` | `test_override_missing_entry_fails_loudly` or equivalent | manual review semantics are explicit |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_golden_set.py -q` | direct acceptable-set behavior stays correct |
| `pytest tests/test_runner.py -q` | runner integration and evaluator behavior stay coherent |
| `pytest tests/test_evaluators.py -q` | evaluator contracts remain consistent |

---

## Acceptance Criteria

- [ ] The acceptable-set helper matches `prompt_eval`'s async evaluator model
- [ ] Judge decisions are validated through a typed contract rather than ad hoc dict access
- [ ] Cache semantics are explicit and fail loudly on malformed input or missing records
- [ ] The feature is proven through direct tests and real runner integration
- [ ] Public docs describe the local SQLite sidecar honestly as evaluator aid, not authoritative experiment record

---

## Notes

Implementation should stay narrow:

- v1 is for string-like alternative acceptance, not arbitrary structured
  semantic equivalence
- v1 uses a local SQLite sidecar
- shared observability integration is explicitly out of scope for this slice

The current local prototype is useful evidence, but it should not be treated as
the final public contract without this plan being completed.
