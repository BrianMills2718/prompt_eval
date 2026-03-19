# Plan NN: [Name]

**Status:** 📋 Planned
**Type:** implementation  <!-- implementation | design | program -->
**Priority:** High | Medium | Low
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** What exists now

**Target:** What we want

**Why:** Why this matters

---

## References Reviewed

> Cite the specific files and docs reviewed before planning.

- `prompt_eval/example.py:10-40` - current implementation
- `docs/UNCERTAINTIES.md` - boundary or open-question context
- `README.md` - repo-level contract

---

## Files Affected

> Declare likely touch points upfront.

- `prompt_eval/example.py` (modify)
- `tests/test_example.py` (create/modify)
- `docs/plans/NN_example.md` (update while executing)

---

## Plan

### Steps

1. State the thinnest first slice
2. State the integration point that gets verified when wired
3. State the next slice only if the first one passes

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_example.py` | `test_happy_path` | Basic behavior works |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_related.py -q` | Nearby behavior stays intact |

---

## Acceptance Criteria

- [ ] Focused tests pass
- [ ] Existing integration points pass when wired
- [ ] Docs and uncertainty notes are updated if the boundary changed

---

## Notes

- Call out real blockers, assumptions, and risks.
- Do not leave `prompt_eval/` changes unlabeled as "trivial."
- Match verification commands to this repo's real structure; `prompt_eval` does
  not currently have a formal `tests/e2e/` hierarchy or a relationship
  validator like `llm_client`.
