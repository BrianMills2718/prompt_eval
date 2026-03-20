# Plan 07: Statistical Engine Modernization

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** statistical-engine drift and overclaim risk

---

## Gap

**Current:** `prompt_eval.stats` still implements bootstrap and Welch-style
comparison by hand. That is acceptable for lightweight internal use, but it is
the wrong place to keep bespoke inferential math. The current docs are already
careful not to overclaim, but the engine should still move to an off-the-shelf
library.

**Target:** modernize the current `compare_variants()` engine without widening
its contract:

1. use SciPy-backed inference for the current IID-style comparison helper,
2. keep the public API compact,
3. document explicitly that this does not resolve the separate paired/clustered
   design question for stronger external claims.

**Why:** the immediate engineering problem is the hand-rolled engine. The
separate experimental-design problem should remain explicit rather than being
buried under better math.

---

## References Reviewed

- `prompt_eval/stats.py`
- `tests/test_stats.py`
- `README.md`
- `docs/API_REFERENCE.md`
- `ISSUES.md`
- `docs/UNCERTAINTIES.md`
- `docs/adr/0005-statistical-inference-boundary.md`

---

## Files Affected

- `prompt_eval/stats.py`
- `tests/test_stats.py`
- `pyproject.toml`
- `README.md`
- `docs/API_REFERENCE.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/CLAUDE.md`
- `ISSUES.md`
- `docs/UNCERTAINTIES.md`
- `docs/adr/README.md`

---

## Plan

### Steps

1. Record the inference boundary:
   - off-the-shelf stats engine for current comparison methods,
   - current API remains lightweight IID-style comparison,
   - paired/clustered comparison remains a separate open design step.
2. Replace the hand-rolled Welch path with a SciPy-backed unequal-variance
   implementation while preserving `compare_variants()` shape.
3. Replace the hand-rolled bootstrap path with a SciPy-backed bootstrap
   implementation while preserving `compare_variants()` shape.
4. Refresh tests and docs so the repo is honest about both the improved engine
   and the remaining paired/clustered design question.

### Thin Slice Status

- [x] Phase 1: record the inference boundary in ADRs/plans/issues
- [x] Phase 2: replace the Welch path with SciPy-backed unequal-variance inference
- [x] Phase 3: replace the bootstrap path with SciPy-backed bootstrap inference
- [x] Phase 4: refresh docs and uncertainty records to match the implemented engine

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_stats.py` | new Welch-detail assertions | Welch path surfaces SciPy-backed inferential detail rather than the old approximation wording |
| `tests/test_stats.py` | new bootstrap-detail assertions | Bootstrap path surfaces SciPy-backed CI detail |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_stats.py -q` | comparison behavior stays correct |
| `pytest tests/test_mcp_server.py -q` | MCP compare tool still returns the expected comparison shape |

---

## Acceptance Criteria

- [x] The statistical-inference boundary is explicit in ADRs/plans/docs
- [x] `compare_variants(method="welch")` uses SciPy-backed unequal-variance inference
- [x] `compare_variants(method="bootstrap")` uses SciPy-backed bootstrap inference
- [x] Public docs stay honest that current comparison is still a lightweight IID-style helper
- [x] The paired/clustered design question is tracked explicitly rather than implied solved

---

## Notes

This plan is intentionally narrow. It modernizes the current comparison engine.
It does **not** claim that pooled-trial inference is the final design for
externally defensible evaluation.

Verified implementation:

- `prompt_eval.stats` now uses SciPy-backed unequal-variance inference for the
  `welch` path.
- `prompt_eval.stats` now uses SciPy-backed percentile bootstrap for the
  `bootstrap` path.
- `pyproject.toml` now declares the SciPy dependency explicitly.
- Focused tests cover the updated inference-detail strings without widening the
  public `ComparisonResult` shape.
