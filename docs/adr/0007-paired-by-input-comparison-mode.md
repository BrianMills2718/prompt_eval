# ADR 0007: Explicit Paired-By-Input Comparison Mode

Status: Accepted
Date: 2026-03-19

## Context

`prompt_eval` experiments usually produce trials shaped like:

- `variant`
- `input_id`
- `replicate`

The existing `compare_variants()` helper compared pooled trial scores. That is
useful for lightweight internal ranking, but it ignores the repeated-measures
structure of typical experiments. Once the same inputs are evaluated across
variants, a stronger comparison should be able to treat `input_id` as the
matched unit of analysis.

At the same time, the package should not silently change its statistical
behavior. Existing pooled comparison is still useful, and callers should choose
the stronger mode explicitly.

## Decision

1. `compare_variants()` keeps its existing pooled comparison behavior as the
   default through `comparison_mode="pooled"`.
2. `compare_variants()` gains an explicit
   `comparison_mode="paired_by_input"` option.
3. In `paired_by_input` mode, the unit of analysis is the `input_id`:
   - scored replicates for each `(variant, input_id)` are aggregated to a mean,
   - the two variants are compared on matched per-input means,
   - the comparison fails loudly if the set of scored `input_id`s does not
     match across the two variants.
4. Method availability becomes mode-specific:
   - pooled mode supports `method="bootstrap"` and `method="welch"`,
   - paired-by-input mode supports `method="bootstrap"` and
     `method="paired_t"`.
5. This mode is the supported answer for stronger within-experiment comparison
   claims inside the current API. More advanced hierarchical or mixed-effects
   inference remains out of scope unless a future program requires it.

## Consequences

Positive:

1. `prompt_eval` now has an explicit comparison contract that matches the
   common `variant x input x replicate` structure.
2. Stronger comparison no longer depends on callers pretending pooled trials are
   the right unit.
3. Backward compatibility is preserved because pooled comparison stays the
   default.

Negative:

1. The API surface is slightly larger.
2. Callers need to understand when pooled versus paired-by-input comparison is
   appropriate.
3. This still stops short of full cluster-robust or mixed-effects modeling.

## Testing Contract

1. Pooled-mode tests must continue to pass unchanged.
2. New focused tests must cover:
   - clearly separated paired-by-input comparisons,
   - dimension-specific paired comparison,
   - invalid paired mode when scored `input_id` coverage does not match,
   - invalid method/mode combinations.
3. MCP-facing comparison helpers must surface the new comparison mode without
   silently changing defaults.
