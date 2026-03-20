# ADR 0005: Statistical Inference Boundary And Off-The-Shelf Engine

Status: Accepted
Date: 2026-03-19

## Context

`prompt_eval.compare_variants()` currently provides lightweight statistical
comparison over per-trial scores. The original implementation hand-rolled both
the bootstrap confidence interval path and the Welch path.

That creates two different concerns:

1. **Engine concern**: inferential math should use an off-the-shelf numerical
   library rather than bespoke formulas and hardcoded approximations.
2. **Design concern**: prompt-eval trials are typically structured by
   `variant x input x replicate`, so stronger external claims may eventually
   require paired or clustered comparison keyed by `input_id`, not just better
   scalar math over pooled trial scores.

Those concerns should not be conflated. Replacing the engine is unblocked. The
paired/clustered design question is real, but it is a separate architectural
step.

## Decision

1. `prompt_eval` will use an off-the-shelf statistics library for inferential
   methods in the current `compare_variants()` API.
2. The current `compare_variants()` contract remains a **lightweight
   IID-style comparison helper** over collected trial scores. It is useful for
   internal experiment triage and ranking, but it is not positioned as
   publication-grade or externally definitive inference.
3. The first modernization slice replaces the current hand-rolled Welch path
   with a SciPy-backed unequal-variance implementation while preserving the
   public API shape.
4. The bootstrap path should also move to SciPy-backed machinery in the same
   modernization program when that can be done without widening the API.
5. The repeated-measures question remains explicit: if `prompt_eval` needs
   externally defensible comparison, it should gain a paired/clustered mode
   keyed by `input_id` rather than overclaiming on pooled-trial inference.

## Consequences

Positive:

1. Numerical inference no longer depends on custom ad hoc approximations.
2. The public comparison API stays small and usable for internal work.
3. The docs become more honest about what `compare_variants()` does and does
   not imply.

Negative:

1. Adding SciPy increases the runtime dependency surface.
2. Off-the-shelf inference does not by itself make the current API
   publication-grade.
3. A later paired/clustered design may require either a new mode or a new API.

## Testing Contract

1. Statistical-engine tests must still cover:
   - clearly separated variants,
   - equal-score edge cases,
   - minimum-sample failures,
   - dimension-specific comparison.
2. The modernization slice should add focused tests that prove the Welch path
   is SciPy-backed and exposes the richer inferential detail now available.
3. The docs and issues surface must say explicitly that stronger external
   claims still require a paired/clustered design decision.
