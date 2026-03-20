# Plan 01: Prompt Eval Master Roadmap

**Status:** ✅ Complete
**Type:** program
**Priority:** Highest
**Blocked By:** None
**Blocks:** repo-wide execution clarity

---

## Gap

**Current:** `prompt_eval` already has real architectural decisions and real
integration work, but until this cleanup it had no trustworthy roadmap. The
repo taught a placeholder plan index, duplicated stale root docs, and no clear
answer to "what remains versus what is already done?"

**Target:** one canonical roadmap states:

1. what `prompt_eval` is,
2. which long-term programs are complete versus still open,
3. what counts as success or failure for each program,
4. whether there is an unblocked next slice or a real blocker.

**Why:** without that control surface, the repo drifts into either fake
"everything is planned" scaffolding or endless one-slice-at-a-time execution
with no durable source of truth.

---

## References Reviewed

- `README.md` - repo-level overview and boundary statement
- `docs/UNCERTAINTIES.md` - real open versus resolved architecture questions
- `docs/adr/0001-llm-client-substrate-boundary.md` - substrate boundary
- `docs/adr/0002-prompt-eval-run-family-mapping.md` - shared run mapping
- `prompt_eval/runner.py` - experiment execution path
- `prompt_eval/query.py` - shared read-side reconstruction
- `prompt_eval/prompt_assets.py` - explicit prompt asset bridge

---

## Files Affected

- `README.md` (active docs contract)
- `AGENTS.md` (repo-operating instructions)
- `CLAUDE.md` (thin pointer)
- `docs/plans/CLAUDE.md` (plan index)
- `docs/plans/02_shared-observability-boundary.md`
- `docs/plans/03_prompt-asset-and-scope-boundary.md`
- `docs/plans/04_documentation-surface-hardening.md`
- `docs/UNCERTAINTIES.md`

---

## Plan

### Steps

1. Keep the shared observability boundary documented as complete once its
   acceptance criteria remain true.
2. Keep documentation cleanup recorded as a completed program instead of
   rediscovering the same drift repeatedly.
3. Update the roadmap only when a new real program appears or a completed
   program needs to be superseded.

### Canonical Execution Rule

Agents working in `prompt_eval` should:

1. anchor implementation work to this roadmap and one child plan,
2. define acceptance criteria before editing code,
3. continue through consecutive unblocked slices after each passing checkpoint,
4. stop only for:
   - a real blocker,
   - a user reprioritization,
   - an unresolved design decision that repo context cannot settle.

Passing one thin slice is not, by itself, a reason to stop.

---

## Required Tests

### New Tests (TDD)

Documentation-only program. No new runtime tests required.

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python scripts/meta/sync_plan_status.py --check` | Plan index and plan files stay consistent |
| markdown link scan over active docs | Canonical docs stop pointing at missing files |

---

## Acceptance Criteria

- [x] One canonical roadmap exists for `prompt_eval`
- [x] Child plans reflect real completed or blocked programs
- [x] The roadmap states whether there is a default next slice or a real blocker
- [x] Root operator docs and plan docs agree on current repo state

---

## Notes

### Repo-Level Definition Of Done

The long-term `prompt_eval` program is done only when all of the following are
true:

1. `prompt_eval` is consistently described as the prompt-evaluation and
   optimization layer rather than a second runtime substrate.
2. Shared `llm_client` observability is the authoritative execution record for
   experiment families.
3. Prompt identity is explicit and queryable when prompt assets exist, with
   inline messages treated either as compatibility input or intentionally
   retained policy.
4. Package scope remains intentionally prompt-centric rather than drifting into
   generic workflow or non-prompt optimization.
5. The docs/plans/operator surface is canonical, compact, and low-drift.

### Program Order

#### Program A: Shared Observability Boundary

**Plan:** [02_shared-observability-boundary.md](./02_shared-observability-boundary.md)  
**Status:** Complete

**Success criteria:**

- prompt_eval emits authoritative shared runs/items/aggregates into
  `llm_client`
- `load_result_from_observability()` reconstructs prompt_eval result families
- corpus-level metrics survive the shared backend round-trip

#### Program B: Prompt Asset And Scope Boundary

**Plan:** [03_prompt-asset-and-scope-boundary.md](./03_prompt-asset-and-scope-boundary.md)  
**Status:** Complete

**Success criteria:**

- explicit prompt assets are the preferred documented path
- inline message compatibility policy is intentionally retained and documented
- the package scope boundary for non-prompt optimization is explicit and
  prompt-centric

#### Program C: Documentation Surface Hardening

**Plan:** [04_documentation-surface-hardening.md](./04_documentation-surface-hardening.md)  
**Status:** Complete

**Success criteria:**

- README exists and is current
- AGENTS/CLAUDE no longer duplicate large stale content
- plan/index surface is real, not placeholder scaffolding
- legacy meta-pattern notes are archived out of the canonical docs surface

#### Program D: Explicit Experiment Semantics And Model Governance Alignment

**Plan:** [05_model-governance-alignment.md](./05_model-governance-alignment.md)
**Status:** Complete

**Success criteria:**

- experiment-semantic choices are documented as explicit caller decisions rather
  than hidden package defaults
- subject-model defaults are removed from public experiment and optimizer
  surfaces where the model choice affects what is being evaluated
- explicit raw model IDs remain available as deliberate overrides for
  model-comparison experiments
- judge helpers may still use documented `llm_client` judging defaults when the
  judge model is not itself under study
- the remaining package-level model/default surfaces are tracked in a concrete
  plan rather than left as undocumented drift

#### Program E: Prompts As Data Cleanup

**Plan:** [06_prompts-as-data-cleanup.md](./06_prompts-as-data-cleanup.md)
**Status:** Complete

**Success criteria:**

- production LLM prompts in `prompt_eval` are stored as YAML/Jinja templates
  rather than embedded Python strings
- judge and optimizer helper calls use explicit local template-path helpers
- no prompt examples are added without explicit review
- focused tests prove the template-backed behavior remains intact

#### Program F: Statistical Engine Modernization

**Plan:** [07_statistical-engine-modernization.md](./07_statistical-engine-modernization.md)
**Status:** Complete

**Success criteria:**

- inferential methods in the current `compare_variants()` API use an
  off-the-shelf stats library rather than hand-rolled math
- public docs stay honest that current comparison remains a lightweight
  IID-style helper
- the paired/clustered comparison question is tracked explicitly as a separate
  design concern

#### Program G: Explicit Paired-By-Input Comparison Mode

**Plan:** [08_paired-by-input-comparison-mode.md](./08_paired-by-input-comparison-mode.md)
**Status:** Complete

**Success criteria:**

- `compare_variants()` supports an explicit paired-by-input mode keyed by
  `input_id`
- pooled comparison remains available and backward compatible
- paired mode uses off-the-shelf paired inference over matched per-input means
- paired mode fails loudly when the repeated-measures contract is not met

### Current Default Next Step

There is no active unblocked program on the current roadmap. Future work now
needs new evidence, a new product goal, or a newly discovered defect rather
than another pending cleanup slice from the current architecture program.
