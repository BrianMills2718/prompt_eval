# Plan 06: Prompts As Data Cleanup

**Status:** ✅ Complete
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** prompt-policy drift inside `prompt_eval`

---

## Gap

**Current:** `prompt_eval` still embeds some production LLM prompts directly in
Python source, including the scalar judge prompt, the dimensional judge prompt,
and the instruction-rewrite prompt.

**Target:** move those prompts into YAML/Jinja templates under a local
`prompts/` directory and render them through `llm_client.render_prompt()` so
prompt source is explicit in code, prompt policy is consistent with the wider
ecosystem, and prompt updates are inspectable without editing Python control
flow.

**Why:** the repo already depends on `llm_client` prompt rendering and explicit
prompt provenance elsewhere. Leaving internal evaluator and optimizer prompts
as inline strings creates policy drift and hides prompt identity.

---

## References Reviewed

- `prompt_eval/evaluators.py`
- `prompt_eval/optimize.py`
- `prompt_eval/prompt_assets.py`
- `README.md`
- `docs/plans/01_master-roadmap.md`
- `docs/adr/0003-explicit-experiment-semantics.md`

---

## Files Affected

- `prompt_eval/evaluators.py`
- `prompt_eval/optimize.py`
- a prompt helper module plus local prompt templates used by `prompt_eval`
- `tests/test_evaluators.py`
- `tests/test_optimize.py`
- `README.md`
- `docs/API_REFERENCE.md`

---

## Plan

### Steps

1. Extract the scalar judge prompt into a local YAML template and keep
   evaluator behavior unchanged.
2. Extract the dimensional judge prompt into a local YAML template and keep structured
   output behavior unchanged.
3. Extract the instruction-rewrite prompt into a local YAML template and keep rewrite
   semantics unchanged.
4. Keep prompt source explicit via named template-path constants and focused
   tests; do not invent fake shared `prompt_ref`s for local-only prompts.
5. Refresh docs so `prompt_eval` no longer teaches inline Python prompts as the
   preferred path.

### Thin Slice Status

- [x] Phase 1: move the scalar judge prompt into a local YAML template
- [x] Phase 2: move the dimensional judge prompt into a local YAML template
- [x] Phase 3: move the instruction-search rewrite prompt into a local YAML template
- [x] Phase 4: add focused template-path tests and package-data wiring
- [x] Phase 5: refresh docs and roadmap status to match the implemented state

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_evaluators.py` | new template-path assertions | judge helpers preserve behavior while using YAML templates |
| `tests/test_optimize.py` | new template-path assertions | instruction-search rewrite prompt uses YAML templates |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_evaluators.py tests/test_optimize.py -q` | helper behavior stays intact |
| `pytest tests/test_prompt_assets.py -q` | prompt-asset integration remains coherent |

---

## Acceptance Criteria

- [x] Production LLM prompts in `prompt_eval` are stored as YAML/Jinja
  templates rather than embedded Python strings
- [x] Judge and optimizer helper calls preserve current behavior
- [x] Prompt source is explicit and inspectable where these prompts are used
- [x] No new prompt examples are introduced without explicit review

---

## Notes

This plan is intentionally narrow. It is about prompt storage and provenance,
not broader redesign of evaluator or optimizer semantics. Shared `prompt_ref`
provenance remains reserved for prompts that truly live in the shared
`llm_client` prompt-asset namespace.

Verified implementation:

- `prompt_eval.prompt_templates` owns the local template-path constants and
  render helpers.
- `prompt_eval/prompts/*.yaml` stores the scalar judge, dimensional judge, and
  instruction-search rewrite prompts as data.
- `prompt_eval.evaluators` and `prompt_eval.optimize` render those templates
  through `llm_client.render_prompt()`.
- Focused tests assert the helper calls go through the expected template paths.
