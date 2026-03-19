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
- Local JSON persistence in `prompt_eval.store` remains available as a
  compatibility/export path, but the authoritative shared record is
  `llm_client` observability.

## Current State

Implemented and verified:

- experiment execution across `PromptVariant x ExperimentInput x replicate`
- per-trial evaluators and corpus-level evaluators
- statistical comparison via bootstrap confidence intervals or Welch's test
- optimization helpers for grid search, few-shot selection, and instruction
  search
- dual-write into `llm_client` shared runs/items/aggregates
- read-side reconstruction via `load_result_from_observability()`
- prompt asset compatibility via `build_prompt_variant_from_ref()`
- optional MCP server for agent workflows

Open boundary decisions:

- whether inline message lists remain indefinite compatibility input or become a
  deprecated path once prompt assets are broadly adopted
- whether `prompt_eval` stays strictly prompt-centric or grows into a broader
  non-prompt optimization package

Those decisions are tracked in [docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md)
and [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md).

## Quick Start

```python
import asyncio

from prompt_eval import (
    Experiment,
    ExperimentInput,
    PromptEvalObservabilityConfig,
    build_prompt_variant_from_ref,
    compare_variants,
    exact_match_evaluator,
    load_result_from_observability,
    run_experiment,
)

experiment = Experiment(
    name="tone_eval",
    variants=[
        build_prompt_variant_from_ref(
            name="concise",
            prompt_ref="shared.summarize.concise@1",
            render_context={"style": "executive"},
            kwargs={"task": "prompt_eval.summary", "max_budget": 1.0},
        ),
        build_prompt_variant_from_ref(
            name="bullet",
            prompt_ref="shared.summarize.bullet@1",
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

comparison = compare_variants(result, "concise", "structured")
reloaded = load_result_from_observability(
    result.execution_id,
    project="prompt_eval_examples",
    dataset="tone_eval",
)
```

Notes:

- The preferred long-term path is explicit prompt assets. Use
  `build_prompt_variant_from_ref()` when a shared `prompt_ref` already exists.
- Inline message lists still work for compatibility or one-off experiments.
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
```

This repo does not currently maintain a relationship validator or a formal E2E
test hierarchy like `llm_client`. The active test suite lives flat under
[`tests/`](tests).

## Docs Map

- [docs/API_REFERENCE.md](docs/API_REFERENCE.md): public API overview
- [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md): canonical
  execution roadmap
- [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md): plan index and statuses
- [docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md): unresolved architecture and
  scope questions
- [docs/adr/README.md](docs/adr/README.md): architecture decision record index
- [docs/archive/README.md](docs/archive/README.md): archived historical notes
