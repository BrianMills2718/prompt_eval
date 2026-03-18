# ADR 0001: `prompt_eval` Depends on `llm_client` for Runtime and Observability

Status: Accepted  
Date: 2026-03-17

## Context

`prompt_eval` provides prompt comparison, evaluators, optimization strategies,
and agent-facing experiment tooling. It already depends on `llm_client` for
actual LLM execution, but the architectural boundary has been easy to misread.

The main confusion has been whether:

1. `prompt_eval` and `llm_client` should share one observability stack or keep
   separate ones,
2. `prompt_eval` should own generic experiment metadata such as
   `condition_id`/`scenario_id`,
3. the two packages should collapse into one package because they are tightly
   related.

## Decision

1. `prompt_eval` remains a distinct package because it owns prompt-evaluation
   semantics, not the generic runtime substrate.
2. `prompt_eval` uses `llm_client` for model and embedding execution and should
   converge on the shared `llm_client` observability backend for authoritative
   run analytics.
3. `prompt_eval` owns prompt-specific concepts such as:
   - `Experiment`,
   - `PromptVariant`,
   - evaluator and judge strategies,
   - optimization and search strategies,
   - prompt-eval-specific MCP tooling.
4. Generic experiment envelope fields such as `condition_id`, `scenario_id`,
   `phase`, `metrics_schema`, `config`, `provenance`, and item-level `extra`
   belong to the shared `llm_client` experiment model. `prompt_eval` populates
   them for prompt experiments, but it does not monopolize their meaning.
5. `prompt_eval.store.py` is a secondary persistence layer for local artifacts
   and exports during the transition. It is not the long-term authoritative
   cross-project analytics backend.
6. Repo topology is secondary to package boundaries. `prompt_eval` and
   `llm_client` may remain separate repos or later move into a monorepo without
   changing this contract.
7. Prompt assets should be referenced explicitly when possible. `prompt_eval`
   evaluates prompts; it does not become the canonical prompt registry.

## Consequences

Positive:
1. Shared runtime and shared observability stay centralized in one substrate.
2. `prompt_eval` can stay focused on evaluation and optimization logic.
3. Cross-project experiment analysis does not require one analytics backend per
   package.

Negative:
1. Migration work is needed because some current `prompt_eval` persistence is
   still local-file oriented.
2. Contributors must separate prompt-specific features from generic experiment
   infrastructure instead of merging them by convenience.

## Testing Contract

1. `prompt_eval` integration tests must continue to prove that real experiment
   runs execute through `llm_client`.
2. Migration work should add tests that prove prompt-eval runs can be emitted
   into shared `llm_client` observability records without losing local artifact
   exports.
3. New optimization features must not assume that prompt_eval owns the global
   experiment schema or the only observability sink.
