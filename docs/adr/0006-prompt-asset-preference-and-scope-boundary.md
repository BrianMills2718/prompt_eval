# ADR 0006: Prompt Asset Preference And Prompt-Centric Scope Boundary

Status: Accepted
Date: 2026-03-19

## Context

`prompt_eval` already supports two real ways to define a `PromptVariant`:

1. inline message lists supplied directly by the caller, and
2. explicit shared prompt assets referenced through `prompt_ref`.

That flexibility created an unresolved boundary question: are inline messages a
temporary compatibility path that should eventually be deprecated, or a
permanent supported input alongside prompt assets?

The package scope was also still open. `prompt_eval` already owns prompt-centric
experiments, evaluators, and optimization loops, but the surrounding ecosystem
also has adjacent needs like code optimization, retrieval tuning, and workflow
experiments. Without a decision, the package could drift into generic
non-prompt optimization.

Real downstream evidence clarifies the boundary:

- `onto-canon6` uses prompt templates and explicit `prompt_ref` provenance as
  the preferred path for maintained extraction experiments.
- `qualitative_coding` still uses inline `PromptVariant(messages=...)` for
  ad hoc prompt iteration where no shared prompt asset exists.

Both patterns are legitimate. Treating inline messages as a deprecation path
would misdescribe real usage. Treating `prompt_eval` as a generic optimization
package would weaken its architecture and blur the `llm_client` boundary.

## Decision

1. Explicit prompt assets are the preferred path when a shared prompt already
   exists or should be promoted into shared provenance.
2. Inline message lists remain a permanent supported input, not a deprecated
   compatibility path. They are appropriate for ad hoc experiments, local
   iteration, tests, and project-local prompts that are not yet shared assets.
3. `prompt_eval` remains prompt-centric. It owns prompt-variant experiments,
   prompt evaluators, prompt optimization helpers, and prompt-specific result
   semantics on top of `llm_client`.
4. Generic non-prompt optimization for code, retrieval, workflow behavior, or
   broader agent systems does not belong in `prompt_eval`. Those concerns may
   use the same `llm_client` substrate, but they should live in sibling tools
   or packages rather than widening this one.
5. The docs should teach both supported input styles honestly:
   - prompt assets are preferred when available,
   - inline messages are intentionally retained and supported.

## Consequences

Positive:

1. The public contract now matches real downstream usage instead of pretending
   that prompt assets have already displaced all inline prompts.
2. `prompt_eval` stays focused and easier to reason about.
3. Downstream projects can adopt prompt assets incrementally without being
   forced through a fake deprecation cycle.

Negative:

1. The package now intentionally supports two prompt-definition styles, which
   means docs and examples must stay clear about which one is preferred.
2. Some future optimization ideas will need a different home instead of being
   added directly to `prompt_eval`.

## Testing Contract

1. Existing prompt-asset coverage must continue to prove that
   `build_prompt_variant_from_ref()` remains a first-class path.
2. Existing runner and experiment tests must continue to prove that inline
   message variants remain valid.
3. This ADR does not require deprecation warnings or runtime behavior changes.
   The required change is an explicit documented boundary.
