# prompt_eval

`prompt_eval` is the prompt-evaluation and optimization layer for the Brian
projects ecosystem. It defines experiment models, runs prompt variants against
datasets, scores outputs, compares variants statistically, and records
prompt-eval-specific semantics on top of `llm_client`'s shared observability
backend.

## Boundary

`prompt_eval` is not the shared runtime substrate.

- `llm_client` owns actual LLM execution, prompt asset resolution, required call
  metadata (`task=`, `trace_id=`, `max_budget=`), and shared observability.
- `prompt_eval` owns prompt-centric experiment semantics: variants, replicates,
  evaluators, statistical comparison, and optimization loops.
- `enforced-planning` owns truth-surface deterministic and semantic review
  tooling. `prompt_eval` has served as a governed consumer pilot for that
  tooling, but that pilot does not widen `prompt_eval` into the owner of
  truth-surface review by default.
- Repo-local dead-code review now follows the same consumer pattern:
  `prompt_eval` uses governed dead-code audit tooling and a reviewed
  `dead_code_audit.json`, but that does not make dead-code policy or framework
  ownership part of the package's product boundary.
- Local JSON persistence in `prompt_eval.store` remains available as a
  compatibility/export path, but the authoritative shared record is
  `llm_client` observability.

## Current State

Implemented and verified:

- experiment execution across `PromptVariant x ExperimentInput x replicate`
- per-trial evaluators and corpus-level evaluators
- statistical comparison via SciPy-backed bootstrap confidence intervals or
  Welch's test
- optimization helpers for grid search, few-shot selection, and instruction
  search
- internal judge and instruction-search prompts stored as local YAML/Jinja
  templates rendered through `llm_client`
- dual-write into `llm_client` shared runs/items/aggregates
- read-side reconstruction via `load_result_from_observability()`
- prompt asset compatibility via `build_prompt_variant_from_ref()`
- growing acceptable-set evaluation for multi-correct-answer string outputs,
  with async judge support and local SQLite caching
- optional MCP server for agent workflows

Resolved boundary decisions:

- prompt assets are preferred when available, but inline message variants remain
  a permanent supported input for ad hoc and project-local experiments
- `prompt_eval` stays prompt-centric rather than broadening into generic code,
  retrieval, or workflow optimization
- `compare_variants()` now supports both explicit pooled comparison and an
  explicit `paired_by_input` mode keyed by `input_id`

The model-governance direction itself is now decided: experiment-semantic
choices such as the subject model should be declared explicitly rather than
silently chosen by package defaults. Internal convenience defaults may remain
for operational plumbing, including judge helpers when the judge model is not
itself under study. That contract is recorded in
[docs/adr/0003-explicit-experiment-semantics.md](docs/adr/0003-explicit-experiment-semantics.md)
and implemented through the public experiment and optimization surfaces. The
remaining prompt-policy drift is tracked separately in
[docs/plans/06_prompts-as-data-cleanup.md](docs/plans/06_prompts-as-data-cleanup.md).

The comparison boundary is also now explicit: `compare_variants()` keeps pooled
comparison as a lightweight default and also supports
`comparison_mode="paired_by_input"` when the same inputs are scored across
variants. Those decisions are recorded in
[docs/adr/0005-statistical-inference-boundary.md](docs/adr/0005-statistical-inference-boundary.md)
and
[docs/adr/0007-paired-by-input-comparison-mode.md](docs/adr/0007-paired-by-input-comparison-mode.md).

The acceptable-set evaluator boundary is now explicit too: `prompt_eval`
supports a growing acceptable-set helper for string-like alternative answers,
with typed judge decisions and a local SQLite sidecar cache rather than shared
observability state. That contract is recorded in
[docs/adr/0004-growing-acceptable-set-evaluator.md](docs/adr/0004-growing-acceptable-set-evaluator.md).

Plan 12 repaired the governed baseline and opted `prompt_eval` into sanctioned
worktree coordination. Plan 13 then replaced bootstrap-only linkage with
actionable governance and declared the repo-local capability ownership source
of record. The current governance follow-on is
[Plan 15: Semantic Truth-Surface Review Pilot](docs/plans/15_semantic-truth-surface-review-pilot.md),
which adopts the scoped truth-surface validator and optional semantic review
layer from `enforced-planning` and renders the current operator view into
[docs/ops/TRUTH_SURFACE_STATUS.md](docs/ops/TRUTH_SURFACE_STATUS.md). The most
recent product capability lane remains
[Plan 11: Precomputed Variant Comparison](docs/plans/11_precomputed_variant_comparison.md),
which added the shared-eval path for comparing frozen outputs from external
systems such as `grounded-research` without reviving alternate runtime modes.
Current architecture decisions remain tracked in
[docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md) and
[docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md).

The evaluator timeout ambiguity is now closed too: judge-evaluator `timeout`
parameters are forwarded through the scoring layer into `llm_client` rather
than being retained as inert compatibility-only arguments.

One deliberate non-goal now follows from the cross-project coordination
cleanup: the semantic truth-surface review pilot remains historical consumer
evidence, not a canonical `prompt_eval` package lane. If a future roadmap
slice wants repo-owned truth-surface functionality here, it must reopen that
boundary explicitly rather than inheriting the pilot by accident.

## Quick Start

```python
import asyncio

from llm_client import get_model
from prompt_eval import (
    Experiment,
    ExperimentInput,
    GoldenSetManager,
    JudgeDecision,
    PromptEvalObservabilityConfig,
    build_prompt_variant_from_ref,
    compare_variants,
    exact_match_evaluator,
    load_result_from_observability,
    run_experiment,
)

subject_model = get_model("synthesis")

experiment = Experiment(
    name="tone_eval",
    variants=[
        build_prompt_variant_from_ref(
            name="concise",
            prompt_ref="shared.summarize.concise@1",
            model=subject_model,
            render_context={"style": "executive"},
            kwargs={"task": "prompt_eval.summary", "max_budget": 1.0},
        ),
        build_prompt_variant_from_ref(
            name="bullet",
            prompt_ref="shared.summarize.bullet@1",
            model=subject_model,
            render_context={"bullet_count": 3},
            kwargs={"task": "prompt_eval.summary", "max_budget": 1.0},
        ),
    ],
    inputs=[
        ExperimentInput(id="doc-1", content="Quarterly revenue increased 18%."),
        ExperimentInput(id="doc-2", content="The committee delayed the vote by two weeks."),
    ],
    n_runs=3,
)

result = asyncio.run(
    run_experiment(
        experiment,
        evaluator=exact_match_evaluator(),
        observability=PromptEvalObservabilityConfig(
            project="prompt_eval_examples",
            dataset="tone_eval",
        ),
    )
)

comparison = compare_variants(
    result,
    "concise",
    "bullet",
    comparison_mode="paired_by_input",
)
reloaded = load_result_from_observability(
    result.execution_id,
    project="prompt_eval_examples",
    dataset="tone_eval",
)

acceptable_set = GoldenSetManager(
    primary_evaluator=exact_match_evaluator(),
    fallback_judge=lambda output, expected: JudgeDecision(
        reasonable=output == "Quarterly revenue rose 18%.",
        reasoning="Semantic paraphrase accepted",
        judge_model="example-judge",
    ),
)
acceptable_evaluator = acceptable_set.build_evaluator(experiment_context="tone_eval")
```

Notes:

- The preferred long-term path is explicit prompt assets. Use
  `build_prompt_variant_from_ref()` when a shared `prompt_ref` already exists.
- Choose the subject model explicitly. The example uses `get_model(...)` as an
  explicit caller decision, but raw model overrides are also valid when model
  comparison is itself the point of the experiment.
- Inline message lists still work for compatibility or one-off experiments.
- Use `comparison_mode="paired_by_input"` when the same `input_id`s are scored
  across variants and you want the matched-input comparison.
- Use `GoldenSetManager.build_evaluator()` when an evaluator needs to reuse
  accepted alternative answers across runs. The acceptable-set cache is a local
  SQLite sidecar, not the authoritative experiment record.
- `kwargs["task"]` and `kwargs["max_budget"]` override the default
  `prompt_eval.run` call metadata and flow through to `llm_client`.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ~/projects/llm_client
pip install -e .
```

Optional MCP support:

```bash
pip install -e .[mcp]
```

## Running Tests

```bash
pytest tests/ -v
python scripts/meta/sync_plan_status.py --check
python scripts/check_markdown_links.py README.md CLAUDE.md docs/plans/CLAUDE.md
```

This repo now carries the shared governed-repo linkage and validation surfaces
installed during Plan 12, including the relationship graph and doc-coupling
helpers. Plan 15 adds repo-local truth-surface tooling:
`python scripts/check_truth_surface_drift.py --config truth_surface_drift.yaml`
for deterministic drift checks,
`python scripts/review_truth_surface_semantic.py --config truth_surface_drift.yaml ...`
for optional advisory semantic review, and
`python scripts/render_truth_surface_status.py --config truth_surface_drift.yaml`
for the compact operator status view. It still does not maintain a formal
`tests/e2e/` hierarchy like `llm_client`; the active test suite lives flat under
[`tests/`](tests).

## Docs Map

- [docs/API_REFERENCE.md](docs/API_REFERENCE.md): public API overview
- [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md): canonical
  execution roadmap
- [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md): plan index and statuses
- [docs/ops/CAPABILITY_DECOMPOSITION.md](docs/ops/CAPABILITY_DECOMPOSITION.md):
  repo-local capability ownership source of record
- [docs/ops/TRUTH_SURFACE_STATUS.md](docs/ops/TRUTH_SURFACE_STATUS.md): current
  scoped truth-surface status view for this repo
- [docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md): unresolved architecture and
  scope questions
- [docs/adr/README.md](docs/adr/README.md): architecture decision record index
- [docs/archive/README.md](docs/archive/README.md): archived historical notes

<!-- Governance refresh: 2026-04-05 — enforced_planning contract current -->
