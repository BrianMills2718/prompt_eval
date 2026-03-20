# Plan 08: Paired-By-Input Comparison Mode

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** repeated-measures drift in statistical comparison

---

## Gap

**Current:** `compare_variants()` already uses SciPy-backed inference, but the
current pooled comparison treats trials like IID draws. Real `prompt_eval`
experiments are usually structured as `variant x input x replicate`, so
stronger within-experiment comparison needs an explicit mode keyed by
`input_id`.

**Target:** add a stronger comparison mode without hiding the choice:

1. keep pooled comparison as the default for backward compatibility,
2. add an explicit `comparison_mode="paired_by_input"` option,
3. aggregate replicates to per-input means and compare matched inputs with
   off-the-shelf paired inference,
4. fail loudly if the paired contract is not satisfied.

**Why:** stronger statistical claims should come from an explicit unit of
analysis, not from silently reinterpreting pooled trial scores.

---

## References Reviewed

- `prompt_eval/stats.py`
- `tests/test_stats.py`
- `prompt_eval/experiment.py`
- `prompt_eval/mcp_server.py`
- `docs/adr/0005-statistical-inference-boundary.md`
- `docs/adr/0007-paired-by-input-comparison-mode.md`
- `onto-canon6/src/onto_canon6/evaluation/prompt_eval_service.py`

---

## Files Affected

- `prompt_eval/stats.py`
- `prompt_eval/mcp_server.py`
- `tests/test_stats.py`
- `tests/test_mcp_server.py`
- `README.md`
- `docs/API_REFERENCE.md`
- `docs/UNCERTAINTIES.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/CLAUDE.md`
- `docs/adr/0007-paired-by-input-comparison-mode.md`

---

## Plan

### Steps

1. Record the comparison-mode decision in an ADR before changing behavior.
2. Extend `compare_variants()` with an explicit `comparison_mode` contract.
3. Implement `paired_by_input` using:
   - per-input replicate means,
   - paired bootstrap resampling across matched inputs,
   - paired t-test for the parametric path.
4. Fail loudly if paired mode does not have matched scored `input_id`s.
5. Refresh MCP/docs/tests so the new mode is explicit and verified.

### Thin Slice Status

- [x] Phase 1: record the comparison-mode boundary
- [x] Phase 2: implement paired-by-input statistics in `prompt_eval.stats`
- [x] Phase 3: expose the mode through the MCP comparison helper
- [x] Phase 4: refresh tests and docs to match the new contract

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_stats.py` | `test_paired_t_detects_clear_difference` | paired-by-input mode compares matched per-input means |
| `tests/test_stats.py` | `test_paired_bootstrap_reports_mode` | paired bootstrap stays explicit and informative |
| `tests/test_stats.py` | `test_paired_mode_requires_matching_input_ids` | unmatched scored inputs fail loudly |
| `tests/test_stats.py` | `test_paired_mode_rejects_welch` | invalid method/mode combination fails loudly |
| `tests/test_mcp_server.py` | `test_compare_variants_paired_by_input` | MCP-facing comparison surface exposes the new mode |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_stats.py -q` | statistical helper contract stays coherent |
| `pytest tests/test_mcp_server.py -q` | MCP surface stays aligned with the public comparison API |

---

## Acceptance Criteria

- [x] `compare_variants()` keeps pooled comparison as the explicit default
- [x] `compare_variants()` supports `comparison_mode="paired_by_input"`
- [x] paired mode aggregates replicates to matched per-input means
- [x] paired mode fails loudly on unmatched scored `input_id`s
- [x] docs and MCP surface describe the new mode honestly

---

## Notes

This plan resolves the open repeated-measures question for the current API
without turning `prompt_eval` into a full mixed-effects statistics package.
The supported stronger mode is now explicit and keyed by `input_id`.
