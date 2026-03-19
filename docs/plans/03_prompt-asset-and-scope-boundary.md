# Plan 03: Prompt Asset And Scope Boundary

**Status:** ⏸️ Blocked
**Type:** design
**Priority:** High
**Blocked By:** ecosystem prompt-asset adoption evidence and package-scope decision
**Blocks:** the next default implementation slice in `prompt_eval`

---

## Gap

**Current:** `prompt_eval` can already run experiments from inline message lists
and can also build `PromptVariant`s from explicit `prompt_ref`s. That means the
mechanism exists, but the long-term contract is still undecided.

Two questions remain open:

1. Are inline message lists a permanent supported input or a compatibility path
   that should eventually be deprecated once prompt assets are broadly adopted?
2. Does `prompt_eval` remain strictly prompt-centric, or should it expand into
   broader non-prompt optimization for code, retrieval, or workflow behavior?

**Target:** document and lock the long-term boundary:

- explicit prompt assets are the preferred path when available,
- inline message compatibility has an explicit policy,
- non-prompt optimization has an explicit yes/no package boundary.

**Why:** without those decisions, new work risks teaching the wrong preferred
path or widening the package without a clear architectural mandate.

---

## References Reviewed

- `README.md`
- `docs/UNCERTAINTIES.md`
- `docs/adr/0001-llm-client-substrate-boundary.md`
- `prompt_eval/experiment.py`
- `prompt_eval/prompt_assets.py`
- `prompt_eval/runner.py`
- `prompt_eval/optimize.py`

---

## Files Affected

- `README.md`
- `AGENTS.md`
- `docs/UNCERTAINTIES.md`
- `docs/plans/01_master-roadmap.md`
- `prompt_eval/experiment.py` (only if compatibility policy changes)
- `prompt_eval/prompt_assets.py` (only if promotion/deprecation behavior changes)
- downstream consumer docs/tests if inline-message policy changes

---

## Plan

### Steps

1. Decide the compatibility policy for inline message lists:
   - permanent supported input, or
   - compatibility path with eventual deprecation conditions.
2. Decide whether `prompt_eval` stays prompt-centric or broadens into
   non-prompt optimization.
3. Update docs, examples, and any deprecation or guidance surfaces to match the
   chosen boundary.
4. Only after the boundary is explicit, implement any behavior or warning
   changes.

### Unblock Conditions

This plan is unblocked only when both of the following happen:

1. **Inline-message policy decision**
   - maintainership explicitly decides that inline messages are either:
     - indefinite supported input, or
     - a compatibility path with stated deprecation conditions.
2. **Package-scope decision**
   - maintainership explicitly decides whether `prompt_eval` remains
     prompt-centric or broadens into non-prompt optimization.

### Decision Ritual

This blocker resolves only through the canonical control surface:

- ADR, or
- roadmap + uncertainty update approved by the maintainer.

Do not treat ad hoc code changes or one-off downstream usage as resolution by
themselves.

### Evidence To Gather

Useful inputs before making the decision:

- whether maintained downstream users are actually adopting `prompt_ref` as
  their primary path,
- whether inline-message experiments are still common enough to justify
  indefinite first-class support,
- whether there is a concrete non-prompt optimization consumer that should live
  in this package rather than beside it.

---

## Required Tests

### New Tests (TDD)

If the blocker is resolved and implementation begins:

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_prompt_assets.py` | `test_builds_variant_from_shared_prompt_asset` | Prompt-asset path remains first-class |
| `tests/test_runner.py` | new or existing coverage depending on decision | Inline-message compatibility or deprecation behavior is explicit |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_prompt_assets.py tests/test_runner.py -q` | Prompt identity and runner behavior stay coherent |
| `pytest tests/ -q` | No regression across execution, optimization, or storage |

---

## Acceptance Criteria

- [ ] Inline-message compatibility policy is explicitly documented
- [ ] Prompt-asset preference is explicitly documented and consistent
- [ ] Package scope for non-prompt optimization is explicit
- [ ] Any resulting behavior changes are verified by focused tests

---

## Notes

This plan is blocked on product/ecosystem decisions, not missing mechanics.
`prompt_eval` already supports both inline messages and explicit prompt assets.
The missing piece is the long-term contract and the decision ritual that locks
it.
