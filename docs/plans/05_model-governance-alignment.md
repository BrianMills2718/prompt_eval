# Plan 05: Explicit Experiment Semantics And Model Governance Alignment

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** undocumented raw-model drift across prompt_eval defaults

---

## Gap

**Current:** `prompt_eval` still bakes raw model IDs like `gpt-5-mini` into
package defaults and convenience entry points. That reduces friction, but it
also hides experiment-semantic choices from agents and users. The result is
extra architectural complexity with less explicit control over what is actually
being tested.

**Target:** make the package-level contract explicit:

1. experiment-semantic choices are caller-declared rather than silently chosen
   by package defaults,
2. explicit raw model IDs remain available as deliberate overrides,
3. prompt/model comparison remains first-class rather than treated as policy
   violation,
4. `llm_client` task-selection buckets remain an internal convenience for
   helpers like judging, not the main public experiment-design abstraction,
5. remaining hidden defaults are tracked and removed in thin slices.

**Why:** without this distinction, `prompt_eval` either drifts away from shared
model governance or keeps hiding the most important experiment choices behind
package defaults. The priority here is explicit experiment intent, not
friction-free convenience.

---

## References Reviewed

- `README.md`
- `AGENTS.md`
- `docs/plans/01_master-roadmap.md`
- `prompt_eval/evaluators.py`
- `prompt_eval/experiment.py`
- `prompt_eval/optimize.py`
- `prompt_eval/prompt_assets.py`
- `prompt_eval/mcp_server.py`
- `tests/test_experiment.py`
- `tests/test_optimize.py`
- `tests/test_prompt_assets.py`
- `docs/adr/0003-explicit-experiment-semantics.md`
- `llm_client/models.py`

---

## Files Affected

- `prompt_eval/evaluators.py`
- `prompt_eval/mcp_server.py`
- `prompt_eval/experiment.py`
- `prompt_eval/optimize.py`
- `prompt_eval/prompt_assets.py`
- `README.md`
- `docs/API_REFERENCE.md`
- `AGENTS.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/CLAUDE.md`
- `tests/test_evaluators.py`
- `tests/test_mcp_server.py`
- `tests/test_experiment.py`
- `tests/test_optimize.py`
- `tests/test_prompt_assets.py`

---

## Plan

### Steps

1. Move judge defaults to `llm_client.get_model("judging")` while preserving
   explicit `judge_model` overrides.
2. Record the architecture decision that experiment-semantic choices should be
   explicit:
   - subject-model choice is not silently invented by the package,
   - `llm_client` task buckets are helper vocabulary, not the main public API,
   - operational defaults may remain when they do not hide experiment meaning.
3. Remove hidden subject-model defaults from public experiment and optimization
   surfaces without erasing legitimate model-comparison use cases.
4. Update docs and tests to explain the distinction between experiment-semantic
   choices and operational defaults.

### Thin Slice Status

- [x] Phase 1: judge defaults use task-based selection when no explicit
  `judge_model`/`judge_models` override is provided
- [x] Phase 2: record the explicit-experiment-semantics architecture decision
  in ADRs, roadmap, and operator docs
- [x] Phase 3: make subject-model choice fail loud in `PromptVariant`,
  `SearchSpace`, `FewShotPool`, `build_prompt_variant_from_ref()`, and
  instruction-search helpers
- [x] Phase 4: refresh public docs/examples and MCP-facing language to match
  the final fail-loud contract

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_evaluators.py` | `test_default_judge_model_uses_task_selection` | single-score judge defaults resolve through `llm_client` policy |
| `tests/test_evaluators.py` | `test_default_dimensional_judge_uses_task_selection` | dimensional judge defaults resolve through `llm_client` policy |
| `tests/test_experiment.py` | new fail-loud omission tests | `PromptVariant` no longer silently chooses a subject model |
| `tests/test_optimize.py` | new fail-loud omission tests | optimization helpers no longer silently choose subject models |
| `tests/test_prompt_assets.py` | new fail-loud omission tests | prompt-asset helper requires explicit subject model |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest tests/test_evaluators.py -q` | judge behavior stays coherent |
| `pytest tests/test_mcp_server.py -q` | MCP-facing judge entry points still work |

---

## Acceptance Criteria

- [x] Judge defaults use `llm_client` task-based selection when no explicit
  override is given
- [x] Explicit raw `judge_model` and `judge_models` overrides still work
- [x] The repo documents that experiment-semantic choices should be explicit
  rather than hidden behind package defaults
- [x] Public experiment and optimization helpers stop silently choosing the
  subject model
- [x] Remaining raw-model/default surfaces are classified as either legitimate
  override surfaces or migration candidates
- [x] Public docs and examples teach explicit subject-model choice instead of
  compatibility defaults

---

## Notes

This plan does **not** assume `prompt_eval` can eliminate all raw model IDs.
That would be the wrong goal. Experiments often need explicit model comparison.
The real goal is to stop silently choosing experiment-semantic parameters while
preserving explicit override surfaces for real experiment design.

Verified implementation now covers:

- `PromptVariant.model` is caller-required.
- `SearchSpace.models` and `FewShotPool.model` are caller-required.
- `instruction_search()` and `optimize(..., strategy="instruction_search")`
  fail loudly unless `model` and `rewrite_model` are explicitly provided.
- `build_prompt_variant_from_ref()` requires an explicit subject model.
- README/API docs and MCP-facing tests reflect the final contract.
