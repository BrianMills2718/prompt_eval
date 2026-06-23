# ADR 0008: Present Prompt Eval Through Applied Experiment Evidence

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Status

Accepted.

## Context

`prompt_eval` is a shared evaluation layer. Its API surface is useful, but
reviewers will understand its value best when it is attached to an applied
project decision: a prompt, schema, model, rubric, or evaluator changed because
frozen-case evidence supported the change.

The repo also has strong adjacent boundaries. `llm_client` owns runtime and
shared observability. Applied projects own substantive task validity. The
truth-surface pilot is historical consumer evidence unless a future plan
explicitly reopens package ownership.

## Decision

Portfolio surfaces should present `prompt_eval` through applied experiment
evidence:

- require a frozen case set when making external claims;
- report baseline, candidate, metric, uncertainty, and decision;
- keep statistical caveats visible;
- describe package APIs as enabling infrastructure, not the main achievement;
- leave runtime execution and shared observability in `llm_client`;
- leave downstream analytic truth to consuming projects.

## Consequences

Benefits:

- makes prompt-eval work understandable to non-infrastructure reviewers;
- reduces manual-spot-check claims;
- keeps boundaries aligned with existing ADRs;
- creates a clear evidence template for applied projects.

Costs:

- strongest evidence depends on downstream experiments;
- case-set design becomes a first-class review surface;
- the package's standalone page must stay modest about claims.

## Controls

- [docs/PORTFOLIO_EXPERIMENT_SUMMARY.md](../PORTFOLIO_EXPERIMENT_SUMMARY.md)
  defines the applied experiment note shape.
- [docs/ops/CAPABILITY_DECOMPOSITION.md](../ops/CAPABILITY_DECOMPOSITION.md)
  defines package ownership boundaries.
- [docs/VALIDATION.md](../VALIDATION.md) separates experiment evidence from
  downstream truth.
- [docs/CONCERNS.md](../CONCERNS.md) tracks portfolio and boundary risks.
