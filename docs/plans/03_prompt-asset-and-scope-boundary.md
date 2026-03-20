# Plan 03: Prompt Asset And Scope Boundary

**Status:** ✅ Complete
**Type:** design
**Priority:** High
**Blocked By:** None
**Blocks:** boundary ambiguity around prompt assets and package scope

---

## Gap

**Current:** `prompt_eval` already supported both inline message lists and
explicit prompt assets, but the repo had not locked the long-term policy for
either prompt-definition style or package scope.

**Target:** record and verify the long-term boundary:

- prompt assets are the preferred path when available,
- inline messages remain a permanent supported input,
- `prompt_eval` remains prompt-centric rather than broadening into generic
  non-prompt optimization.

**Why:** without those decisions, docs and future code would keep oscillating
between fake deprecation pressure for inline prompts and accidental scope creep
into adjacent tooling.

---

## References Reviewed

- `README.md`
- `docs/UNCERTAINTIES.md`
- `docs/adr/0001-llm-client-substrate-boundary.md`
- `onto-canon6/src/onto_canon6/evaluation/prompt_eval_service.py`
- `qualitative_coding/scripts/optimize_thematic_prompt.py`
- `prompt_eval/prompt_assets.py`

---

## Files Affected

- `README.md`
- `CLAUDE.md`
- `docs/UNCERTAINTIES.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/CLAUDE.md`
- `docs/adr/0006-prompt-asset-preference-and-scope-boundary.md`
- `docs/API_REFERENCE.md`

---

## Plan

### Steps

1. Review real downstream consumers rather than deciding from abstraction alone.
2. Lock the inline-message policy explicitly.
3. Lock the package-scope decision explicitly.
4. Update roadmap, uncertainty, and public docs so the chosen boundary becomes
   canonical.

### Thin Slice Status

- [x] Phase 1: gather downstream evidence from maintained consumers
- [x] Phase 2: record the prompt-asset preference and inline-message policy
- [x] Phase 3: record the package-scope decision
- [x] Phase 4: refresh roadmap, uncertainty, and public docs to match

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_prompt_assets.py` | `test_builds_variant_from_shared_prompt_asset` | Prompt-asset path remains first-class |
| `tests/test_runner.py` | existing inline-variant coverage | Inline message variants remain a supported input rather than an accidental regression |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_prompt_assets.py tests/test_runner.py -q` | Prompt identity and inline-message execution both remain supported |

---

## Acceptance Criteria

- [x] Inline-message compatibility policy is explicitly documented
- [x] Prompt-asset preference is explicitly documented and consistent
- [x] Package scope for non-prompt optimization is explicit
- [x] The boundary is recorded through an ADR, roadmap update, and uncertainty resolution

---

## Notes

Resolved by ADR 0006.

Decision summary:

- Prompt assets are preferred when available.
- Inline message lists remain a permanent supported input.
- `prompt_eval` stays prompt-centric; broader non-prompt optimization belongs
  outside this package.
