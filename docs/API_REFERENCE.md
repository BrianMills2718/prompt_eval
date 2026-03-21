# API Reference

This is a high-level reference for the public `prompt_eval` surface. It is not
generated API documentation; it exists to show the stable concepts and where to
look next.

## Core Models

- `Experiment`
  Defines one experiment family: variants, inputs, number of replicates, and
  optional structured response model.
- `PromptVariant`
  One variant under test. Carries `messages`, optional `prompt_ref`, explicit
  subject-model choice, temperature, and per-call `kwargs`.
- `ExperimentInput`
  One dataset item under evaluation, with optional `expected` value for
  scoring.
- `Trial`
  One executed `(variant, input, replicate)` result, including output,
  score/error fields, cost, latency, and optional shared `trace_id`.
- `EvalResult`
  Aggregated result family for one experiment invocation.
- `VariantSummary`
  Per-variant summary metrics across trials, including optional corpus-level
  scores.

Defined in [prompt_eval/experiment.py](../prompt_eval/experiment.py).

## Execution

- `run_experiment(experiment, *, evaluator=None, corpus_evaluator=None, observability=True)`
  Runs one experiment family and returns `EvalResult`.
- `PromptEvalObservabilityConfig`
  Controls shared `llm_client` observability emission: `project`, `dataset`,
  `phase`, `scenario_id`, explicit `experiment_execution_id`, and extra
  provenance.
- `load_result_from_observability(execution_id, *, project=None, dataset=None, limit=1000)`
  Reconstructs an `EvalResult` from shared `llm_client` runs/items/aggregates.

Defined in:

- [prompt_eval/runner.py](../prompt_eval/runner.py)
- [prompt_eval/observability.py](../prompt_eval/observability.py)
- [prompt_eval/query.py](../prompt_eval/query.py)

## Evaluators

- `exact_match_evaluator()`
- `contains_evaluator()`
- `kappa_evaluator(extractor)`
- `llm_judge_evaluator(...)`
- `llm_judge_dimensional_evaluator(...)`
- `EvalScore`
- `RubricDimension`
- `GoldenSetManager`
- `JudgeDecision`

These functions build per-trial evaluator callables. The LLM judge variants use
`llm_client`, render their internal prompts from local YAML templates, and fail
loudly when no judge produces a valid score.

`GoldenSetManager` is the acceptable-set helper for string-like
multi-correct-answer scenarios. It provides:

- `aevaluate(output, expected, *, experiment_context="")`
- `build_evaluator(*, experiment_context="")`
- `get_alternatives(reference)`
- `override(reference, alternative, status)`

It keeps a local SQLite cache of accepted/rejected alternatives and validates
fallback judge decisions through the typed `JudgeDecision` model. This cache is
an evaluator aid, not the authoritative experiment record.

Defined in [prompt_eval/evaluators.py](../prompt_eval/evaluators.py).
The acceptable-set helper is defined in [prompt_eval/golden_set.py](../prompt_eval/golden_set.py).

## Statistics

- `compare_variants(result, variant_a, variant_b, *, method="bootstrap", comparison_mode="pooled", ...)`

Compares two variants using SciPy-backed bootstrap confidence intervals or
Welch's test over the current pooled-trial comparison contract, or using a
matched `input_id` comparison through `comparison_mode="paired_by_input"` with
paired bootstrap or paired t-test. Corpus-level scores are not statistically
compared because they are single aggregated values per variant.

Defined in [prompt_eval/stats.py](../prompt_eval/stats.py).

## Optimization

- `optimize(...)`
- `grid_search(...)`
- `few_shot_selection(...)`
- `instruction_search(...)`
- `SearchSpace`
- `FewShotPool`
- `OptimizeResult`

These are prompt-centric optimization helpers layered on top of the normal
experiment runner. Subject-model choice is part of experiment semantics and
should be declared deliberately by the caller. `instruction_search()` also uses
an internal YAML rewrite prompt rendered through `llm_client`.

Defined in [prompt_eval/optimize.py](../prompt_eval/optimize.py).

## Prompt Assets

- `build_prompt_variant_from_ref(name, prompt_ref, *, model, render_context=None, ...)`

Turns an explicit shared `prompt_ref` into a normal `PromptVariant` while
preserving prompt identity in observability provenance. Prompt assets are the
preferred path when they already exist, but inline message variants remain a
supported input elsewhere in the package.

Defined in [prompt_eval/prompt_assets.py](../prompt_eval/prompt_assets.py).

## Persistence

- `save_result(...)`
- `load_result(...)`
- `save_experiment(...)`
- `load_experiment(...)`
- `list_results(...)`

These JSON artifact helpers remain supported as a compatibility/export path even
though shared observability is the authoritative backend.

Defined in [prompt_eval/store.py](../prompt_eval/store.py).

## MCP Server

Optional entrypoint:

- `prompt-eval-mcp`

Implements agent-facing tools for experiment execution and result inspection.
Requires `pip install -e .[mcp]`.

Defined in [prompt_eval/mcp_server.py](../prompt_eval/mcp_server.py).
