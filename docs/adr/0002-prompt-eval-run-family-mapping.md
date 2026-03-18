# ADR 0002: `prompt_eval` Experiment Families Map to `llm_client` Runs

Status: Accepted  
Date: 2026-03-17

## Context

ADR 0001 established that `prompt_eval` depends on `llm_client` for runtime and
shared observability, but it did not define the exact mapping from
`prompt_eval`'s current data model into the shared `llm_client` run schema.

That ambiguity blocked the first proving slice because the following questions
were still open:

1. whether one `Experiment` should map to one `llm_client` run or many,
2. where `condition_id`, `scenario_id`, `phase`, `metrics_schema`, `config`,
   `provenance`, and item-level `extra` should come from,
3. how to preserve repeated trial semantics from `n_runs`,
4. how to make `compare_runs()` and `compare_cohorts()` useful for prompt
   experiments.

The current `prompt_eval` model is:

- `Experiment`: an evaluation campaign with multiple variants, inputs, and
  repeats,
- `PromptVariant`: a single prompt condition,
- `ExperimentInput`: one evaluation case,
- `Trial`: one variant/input/run result,
- `VariantSummary`: an aggregate over all trials for one variant.

The current shared `llm_client` experiment schema is:

- one run with one `model`,
- one `condition_id`,
- optional `seed` and `replicate`,
- many per-item rows,
- run-level aggregate summary metrics.

## Decision

### 1. Mapping Unit

1. A `prompt_eval` `Experiment` invocation maps to a **run family**, not a
   single `llm_client` run.
2. One `PromptVariant` plus one repeat index (`run_idx`) maps to one
   `llm_client` run.
3. Therefore an experiment with `len(variants) = V` and `n_runs = R` produces
   `V * R` shared runs.

This is the only mapping that preserves:

- `condition_id = one prompt variant`,
- `model = one variant model`,
- `replicate = one repeat index`,
- stable item IDs across conditions for comparison.

### 2. Structural Mapping

| `prompt_eval` concept | `llm_client` concept | Notes |
| --- | --- | --- |
| `Experiment` | Run family | Grouped by shared provenance and defaults |
| `PromptVariant` | Run condition | Maps to `condition_id` |
| `run_idx` | `replicate` | Zero-based replicate index |
| `ExperimentInput` | Item | One item row per input per run |
| `Trial` | Item result | One `log_item()` call |
| `VariantSummary` | Derived cohort view | Aggregated over a family of runs, not a single run |

### 3. Run-Level Field Mapping

For each `PromptVariant` and `run_idx`, the shared run fields map as follows:

- `project`
  - explicit integration override if supplied,
  - otherwise let `llm_client` derive the current project.
- `dataset`
  - explicit dataset override if supplied,
  - otherwise compatibility fallback to `Experiment.name`.
  - The fallback is transitional and should not be treated as the ideal long-term
    dataset contract.
- `model`
  - `PromptVariant.model`
- `condition_id`
  - `PromptVariant.name`
- `seed`
  - explicit caller-supplied evaluation or optimizer seed if known,
  - otherwise `None`
- `replicate`
  - `run_idx` from `0` to `Experiment.n_runs - 1`
- `scenario_id`
  - default to `Experiment.name`
  - if the caller supplies a more specific scenario override, that override wins
    and `Experiment.name` remains in provenance
- `phase`
  - default `"evaluation"` for `run_experiment()`
  - default `"grid_search"`, `"few_shot_selection"`, or
    `"instruction_search"` for the corresponding optimizer entry points
  - finer-grained optimizer iteration metadata belongs in provenance, not in
    `phase`
- `metrics_schema`
  - `None` when no evaluator is used
  - `["score"]` when a scalar evaluator is used
  - `["score", *sorted(dimension_names)]` when an `EvalScore` evaluator is used
  - corpus-level metrics do not belong in `metrics_schema` because they are not
    item-level values
- `config`
  - machine-queryable execution knobs only:
    - `temperature`
    - `variant_kwargs`
    - `structured_output`
    - `response_model_name`
  - prompt content, evaluator identity, and optimizer lineage do not belong in
    `config`
- `provenance`
  - must carry the run-family metadata needed to reconstruct the larger
    experiment:
    - `source_package: "prompt_eval"`
    - `experiment_name`
    - `experiment_execution_id` shared by all runs emitted from one invocation
    - `variant_name`
    - `variant_count`
    - `input_count`
    - `n_runs`
    - `evaluator_name`
    - `corpus_evaluator_name` when present
    - prompt identity metadata:
      - `prompt_ref` when an explicit prompt asset is used
      - otherwise a stable hash of the variant message template
    - optimizer metadata when the run came from a search strategy

### 4. Item-Level Field Mapping

For each `Trial`, the shared item fields map as follows:

- `item_id`
  - `ExperimentInput.id`
  - must stay identical across conditions and replicates so run comparisons can
    align items correctly
- `metrics`
  - `{}` when no evaluator is used
  - `{"score": Trial.score}` when a scalar evaluator is used
  - `{"score": Trial.score, **Trial.dimension_scores}` when an `EvalScore`
    evaluator is used
- `predicted`
  - raw text output for string completions
  - canonical JSON string for structured outputs
  - fallback to `str(output)` only when no more precise serialization exists
- `gold`
  - canonical string or JSON serialization of `ExperimentInput.expected`
  - `None` when no expected value exists
- `latency_s`
  - `Trial.latency_ms / 1000`
- `cost`
  - `Trial.cost`
- `error`
  - `Trial.error`
- `trace_id`
  - must be unique per trial and hierarchical:
    `prompt_eval/{experiment_execution_id}/{condition_id}/r{replicate}/{item_id}`
  - the same trace ID should be used for the underlying LLM call and the
    resulting `log_item()` record
- `extra`
  - item-scoped metadata only:
    - `tokens_used`
    - `reasoning` from dimensional evaluators when present
    - serialization format markers for `predicted` and `gold`
    - `input_content_sha256`
    - `expected_sha256` when expected data exists
    - optional `artifact_id` links when richer payload capture is stored outside
      the observability DB
  - raw input content should not be duplicated into item metadata by default

### 5. Aggregate Metrics

1. The default integration should call `finish_run(summary_metrics=None)` and let
   `llm_client` auto-compute `avg_score` and any `avg_{dimension}` fields from
   `metrics_schema`.
2. `prompt_eval` `VariantSummary` values are family-level aggregates and should
   be derived across the set of runs sharing:
   - `experiment_execution_id`,
   - `condition_id`,
   - `dataset`,
   - `phase`
3. These `VariantSummary` fields do **not** map one-to-one onto a single shared
   run:
   - `mean_score`
   - `std_score`
   - `mean_cost`
   - `mean_latency_ms`
   - `total_tokens`
4. `corpus_score` and `corpus_dimension_scores` map to shared family-level
   aggregate rows, not to individual run rows, because current corpus
   evaluators run across all successful outputs for a variant rather than per
   replicate run.

### 6. Compatibility Rules

1. Inline `PromptVariant.messages` remain a compatibility input.
2. The preferred long-term path is explicit prompt asset references with
   identity recorded in provenance.
3. When explicit dataset metadata is absent, using `Experiment.name` as
   `dataset` is acceptable for compatibility but should be treated as migration
   debt, not as the final conceptual model.

## Consequences

Positive:
1. `condition_id` and `replicate` now have precise meanings for prompt
   experiments.
2. `compare_runs()` can compare matched item IDs across prompt variants.
3. `compare_cohorts()` can compare repeated prompt conditions across replicates.
4. The shared run schema stays generic while still supporting prompt-eval
   workflows.

Negative:
1. One logical prompt experiment now produces many shared runs.
2. `VariantSummary` remains a higher-level derived view rather than a direct
   shared row type.
3. Family-level aggregate rows add a second shared record type alongside runs
   and items, so readers must reconstruct prompt-eval summaries from both.

## Testing Contract

1. Dual-write integration tests must prove that `V * R` shared runs are emitted
   for an experiment with `V` variants and `R` repeats.
2. Item IDs must match across conditions for the same `ExperimentInput.id`.
3. Replicate indices must be zero-based and stable.
4. Trace IDs must include the replicate and item identity so repeated runs do
   not collide.
5. Shared run queries must be able to reconstruct variant cohorts using
   `condition_id`, `replicate`, `scenario_id`, and provenance family metadata.
