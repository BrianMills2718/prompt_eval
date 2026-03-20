# ADR 0004: Growing Acceptable Set Evaluator

Status: Accepted
Date: 2026-03-20

## Context

Some `prompt_eval` use cases have more than one reasonable answer for the same
input. Exact-match or narrow deterministic evaluators can therefore mark a
valid alternative as wrong even when a human or judge model would accept it.

This is already a real downstream need. `onto-canon6`'s ancestor-aware
evaluation design depends on a persistent acceptable-alternatives layer so that
reasonable non-golden labels can be reviewed once and then reused across future
runs instead of paying repeated judge cost forever.

The current local prototype proves the core idea:

1. run a cheap primary evaluator first,
2. if it misses, check a persistent acceptable-alternatives cache,
3. if the pair is unknown, route to a judge,
4. persist the judge decision, and
5. reuse that decision on future runs.

The architectural gap is that the prototype is not yet aligned with
`prompt_eval`'s actual evaluator contract:

- `run_experiment()` supports sync or async evaluators,
- the prototype currently exposes a synchronous manager method,
- the fallback judge contract is an ad hoc `dict`,
- and the persistence boundary is not yet explicitly documented.

## Decision

1. `prompt_eval` should support a growing acceptable-set evaluator pattern as a
   first-class prompt-centric evaluation helper.
2. The v1 unit of comparison is intentionally narrow: string-like output versus
   string-like reference value, with dataset/dimension scoping.
3. The evaluator wrapper should match `prompt_eval`'s real execution model:
   it must support async judge calls and be usable directly as a trial
   evaluator inside `run_experiment()`.
4. Judge decisions should use a typed contract rather than an unstructured
   `dict`.
5. V1 persistence should stay a local SQLite sidecar, separate from
   `llm_client` shared observability. This cache is an evaluator aid, not the
   authoritative experiment record.
6. The package should fail loudly on malformed judge decisions, database
   failures, or ambiguous cache semantics. No silent fallback to “just score
   zero” once the acceptable-set path is chosen.
7. This pattern is prompt-centric evaluation infrastructure. It belongs in
   `prompt_eval`, not in `llm_client`.

## Consequences

Positive:

1. Multi-correct-answer evaluation becomes cheaper over time because accepted
   alternatives are reused.
2. Downstream packages like `onto-canon6` get a reusable evaluator primitive
   instead of each building a bespoke acceptable-alternatives cache.
3. The architecture matches `prompt_eval`'s actual async evaluator contract.

Negative:

1. `prompt_eval` gains another persistence surface to maintain.
2. V1 is intentionally scoped to string-like comparisons, not arbitrary
   structured outputs.
3. A bad judge ruling can pollute the cache until manually reviewed or
   overridden.

## Non-Goals

1. Replacing `llm_client` observability as the authoritative experiment record.
2. Designing a distributed or shared acceptable-alternatives service.
3. Solving arbitrary structured semantic equivalence in the first slice.

## Testing Contract

1. Focused tests must prove:
   - primary-hit behavior skips cache/judge,
   - cache hits skip judge calls,
   - unknown alternatives route to the judge once,
   - accepted and rejected alternatives persist distinctly,
   - dataset/dimension scoping works,
   - malformed judge decisions fail loudly,
   - async integration works with `run_experiment()`.
2. The implementation plan should prove the evaluator through the real
   `prompt_eval` runner path, not only through direct unit calls.
