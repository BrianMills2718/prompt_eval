# ADR 0003: Experiment-Semantic Choices Must Be Explicit

Status: Accepted  
Date: 2026-03-19

## Context

`prompt_eval` grew several convenience defaults intended to reduce friction,
especially around model choice. Examples include hidden subject-model defaults
in `PromptVariant`, `SearchSpace`, `FewShotPool`, and instruction-search
helpers.

That convenience has a real cost:

1. it hides what the experiment is actually testing,
2. it makes agents less likely to exercise the full `llm_client`/`prompt_eval`
   capability surface deliberately,
3. it pushes architectural complexity into package defaults and task-bucket
   helpers instead of keeping experiment intent in the caller.

At the same time, not every default is equally problematic. Some defaults are
just operational plumbing. Others decide the semantics of the experiment.

The key architectural distinction is therefore:

- **experiment-semantic choices**: parameters that change what is being
  evaluated or optimized,
- **operational defaults**: parameters that control runtime plumbing without
  redefining the experiment itself.

## Decision

1. `prompt_eval` should prefer explicit caller declaration for
   experiment-semantic choices.
2. The package should not silently choose the **subject model** for public
   experiment and optimization helpers.
3. Explicit raw model IDs remain valid and first-class when model comparison is
   itself the purpose of the experiment.
4. `llm_client` task-selection buckets remain useful, but mainly as:
   - internal convenience for helper defaults such as judge selection,
   - inspectable policy/reporting vocabulary,
   - optional caller tooling outside `prompt_eval`.
5. `llm_client` task-selection buckets are **not** the primary public
   experiment-design abstraction for `prompt_eval`.
6. Operational defaults may remain where they do not hide experiment meaning.
   Examples include timeout, retry behavior, trace/task naming conventions,
   cache behavior, and similar runtime plumbing.
7. Judge helpers may keep a documented convenience default through
   `llm_client.get_model("judging")` when the judge model is not itself under
   study, while still preserving explicit `judge_model`/`judge_models`
   overrides.

## Examples

Experiment-semantic choices that should be caller-declared include:

- subject model for a prompt variant,
- candidate models in optimization search spaces,
- few-shot selection model,
- instruction-search evaluation model and rewrite model,
- prompt source (`messages` or explicit `prompt_ref`),
- evaluator/rubric choice,
- structured-output schema when the experiment depends on it.

Operational defaults that may remain package-managed include:

- timeout,
- retry count/policy,
- cache behavior,
- observability phase naming,
- default trace/task naming conventions.

## Consequences

Positive:
1. Experiments become easier to interpret because the important choices are
   declared by the caller.
2. Agents are more likely to use `llm_client` and `prompt_eval` deliberately
   instead of inheriting hidden package choices.
3. The architecture becomes simpler: less meaning is hidden in defaults and
   selection buckets.

Negative:
1. Public APIs become a bit more verbose.
2. Some convenience helpers will need migration work and clearer docs.
3. Existing callers that relied on hidden subject-model defaults will need to
   pass models explicitly once the fail-loud slices land.

## Testing Contract

1. Public experiment and optimization helpers must raise loudly when required
   experiment-semantic parameters are omitted.
2. Explicit override paths must remain covered by tests.
3. Judge helper defaults must continue to prove both:
   - default resolution through `llm_client` judging policy,
   - explicit override behavior when a judge model is specified.
