# Prompt Eval Integration Uncertainties

This document tracks the unresolved questions specific to `prompt_eval` after
ADR 0001 established the package boundary against `llm_client`.

Canonical execution order lives in
[`docs/plans/01_master-roadmap.md`](plans/01_master-roadmap.md). This file is
only for open or resolved architecture questions, not for sequencing work.

## Current Assumptions

1. `prompt_eval` keeps owning prompt-specific evaluation and optimization
   semantics.
2. `llm_client` remains the authoritative shared observability backend.
3. Local JSON result files can continue to exist as artifacts during migration.
4. Prompt identity should move toward explicit asset references when available.

## Open Questions

### U1: Shared Run Schema Mapping

**Status:** ✅ Resolved  
**Raised:** 2026-03-17  
**Resolved:** 2026-03-17  
**Resolution:** ADR 0002 defines the canonical mapping. A `prompt_eval`
experiment invocation maps to a family of `llm_client` runs, with one shared
run per `PromptVariant` x `replicate`, one shared item per `ExperimentInput`,
and family-level prompt-eval summaries derived across those runs.  
**Verified in:** `docs/adr/0002-prompt-eval-run-family-mapping.md`

### U2: Dual Persistence Strategy

**Status:** ✅ Resolved  
**Raised:** 2026-03-17  
**Resolved:** 2026-03-17  
**Resolution:** `prompt_eval` now dual-writes by default. `run_experiment()`
and the optimization entry points emit authoritative shared runs and items into
`llm_client` observability while local JSON result files remain available as
compatibility artifacts. `load_result_from_observability()` reconstructs a
prompt-eval result family from the shared backend by `execution_id`.  
**Verified in:** code in `prompt_eval.runner`, `prompt_eval.query`, and tests
covering shared-run emission plus read-side reconstruction

### U3: Backward Compatibility for Inline Prompt Messages

**Status:** ❓ Open  
**Raised:** 2026-03-17  
**Context:** Current `PromptVariant` instances carry inline message lists, while
the architecture direction prefers explicit prompt assets as data.  
**Current assumption:** support inline messages as a compatibility input for now,
but treat explicit prompt asset references as the preferred long-term model.

### U4: Scope Boundary for Non-Prompt Optimization

**Status:** ⏸️ Deferred  
**Raised:** 2026-03-17  
**Context:** Some runs will optimize code, retrieval, or workflow behavior
instead of prompt text.  
**Current assumption:** `prompt_eval` remains focused on prompt-centric
evaluation and optimization. Other evaluators can share the same observability
backend without forcing them into this package.

### U5: Corpus-Level Evaluator Representation

**Status:** ✅ Resolved  
**Raised:** 2026-03-17  
**Resolved:** 2026-03-17  
**Resolution:** corpus-level evaluator outputs now persist as first-class shared
`llm_client` experiment aggregates keyed by `family_id=execution_id` and
`condition_id=variant.name`. `load_result_from_observability()` hydrates those
rows back into `VariantSummary.corpus_score` and
`VariantSummary.corpus_dimension_scores`.  
**Verified in:** `llm_client` experiment aggregate APIs, `prompt_eval.runner`,
`prompt_eval.query`, and round-trip tests covering corpus metrics

### U6: Should `prompt_eval` Hide Experiment-Semantic Choices Behind Defaults?

**Status:** ✅ Resolved
**Raised:** 2026-03-19
**Resolved:** 2026-03-19
**Resolution:** `prompt_eval` should prefer explicit declaration of
experiment-semantic choices rather than silently choosing them through package
defaults. In particular, the subject model for an experiment or optimization
helper should be caller-declared, not invented by convenience behavior.
`llm_client` task-selection buckets remain useful as internal helper vocabulary
and policy/reporting categories, but they are not the main public
experiment-design abstraction for this package. Operational defaults may remain
for plumbing concerns and for judge helpers when the judge model is not itself
under study.
**Verified in:** `docs/adr/0003-explicit-experiment-semantics.md`,
`docs/plans/05_model-governance-alignment.md`
