#!/usr/bin/env python3
"""Register prompt_eval data contract boundaries in the shared registry.

Manually registers the Pydantic models and boundary functions that other
projects depend on. prompt_eval's models are regular BaseModels (not
BoundaryModels), so we register them via BoundaryInfo rather than the
@boundary decorator.

Run with prompt_eval's venv:
    ~/projects/prompt_eval/.venv/bin/python scripts/register_schemas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import prompt_eval models
# ---------------------------------------------------------------------------
try:
    from prompt_eval.experiment import (
        EvalResult,
        Experiment,
        ExperimentInput,
        PromptVariant,
        Trial,
        VariantSummary,
    )
    from prompt_eval.scoring import (
        CriterionScore,
        Rubric,
        RubricCriterion,
        ScoreResult,
    )
    from prompt_eval.optimize import (
        FewShotPool,
        OptimizeResult,
        SearchSpace,
    )
    from prompt_eval.evaluators import (
        DimensionScore,
        JudgeVerdict,
    )
    from prompt_eval.golden_set import JudgeDecision
    print("Imported prompt_eval models")
except ImportError as e:
    print(f"prompt_eval import failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import data_contracts registry
# ---------------------------------------------------------------------------
try:
    from data_contracts.registry import BoundaryInfo, registry
except ImportError as e:
    print(f"data_contracts import failed: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper to build schema from a Pydantic model
# ---------------------------------------------------------------------------
def _schema(model: type) -> dict | None:
    """Extract JSON schema from a Pydantic model class, or None."""
    if model is None:
        return None
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return None


# ---------------------------------------------------------------------------
# Register boundaries
# ---------------------------------------------------------------------------

BOUNDARIES = [
    # -- Experiment execution --
    BoundaryInfo(
        name="prompt_eval.run_experiment",
        version="1.2.0",
        producer_project="prompt_eval",
        consumer_projects=["onto-canon6", "research_v3", "grounded-research"],
        input_schema=_schema(Experiment),
        output_schema=_schema(EvalResult),
        description=(
            "Run all prompt variants against all inputs, collect trials, "
            "compute per-variant summaries. The primary experiment execution "
            "entry point."
        ),
    ),

    # -- Statistical comparison --
    BoundaryInfo(
        name="prompt_eval.compare_variants",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(EvalResult),
        output_schema={
            "title": "ComparisonResult",
            "type": "object",
            "properties": {
                "variant_a": {"type": "string"},
                "variant_b": {"type": "string"},
                "mean_a": {"type": "number"},
                "mean_b": {"type": "number"},
                "difference": {"type": "number"},
                "ci_lower": {"type": "number"},
                "ci_upper": {"type": "number"},
                "significant": {"type": "boolean"},
                "method": {"type": "string", "enum": ["bootstrap", "welch", "paired_t"]},
                "comparison_mode": {"type": "string", "enum": ["pooled", "paired_by_input"]},
                "n_units": {"type": "integer"},
                "detail": {"type": "string"},
            },
            "required": [
                "variant_a", "variant_b", "mean_a", "mean_b",
                "difference", "ci_lower", "ci_upper", "significant",
                "method", "comparison_mode", "n_units", "detail",
            ],
        },
        description=(
            "Compare two prompt variants with bootstrap CI, Welch's t-test, "
            "or paired t-test. Supports pooled and paired-by-input modes."
        ),
    ),

    # -- Rubric scoring (async) --
    BoundaryInfo(
        name="prompt_eval.ascore_output",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=["onto-canon6", "research_v3"],
        input_schema=_schema(Rubric),
        output_schema=_schema(ScoreResult),
        description=(
            "Score a task output against a rubric via LLM-as-judge. "
            "Async entry point. Logs to llm_client observability."
        ),
    ),

    # -- Rubric scoring (multi-judge) --
    BoundaryInfo(
        name="prompt_eval.ascore_output_multi_judge",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(Rubric),
        output_schema=_schema(ScoreResult),
        description=(
            "Score output using multiple judge models and average their "
            "scores. Each judge independently scores the output against "
            "the rubric."
        ),
    ),

    # -- Rubric scoring (sync wrapper) --
    BoundaryInfo(
        name="prompt_eval.score_output",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(Rubric),
        output_schema=_schema(ScoreResult),
        description="Sync wrapper for ascore_output.",
    ),

    # -- Optimization: grid search --
    BoundaryInfo(
        name="prompt_eval.grid_search",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(SearchSpace),
        output_schema=_schema(OptimizeResult),
        description=(
            "Exhaustive grid search over prompt template, model, "
            "temperature, and kwarg combinations."
        ),
    ),

    # -- Optimization: few-shot selection --
    BoundaryInfo(
        name="prompt_eval.few_shot_selection",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(FewShotPool),
        output_schema=_schema(OptimizeResult),
        description=(
            "Search over C(n,k) combinations of examples from a pool, "
            "injected into a base prompt template."
        ),
    ),

    # -- Optimization: unified entry point --
    BoundaryInfo(
        name="prompt_eval.optimize",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(SearchSpace),
        output_schema=_schema(OptimizeResult),
        description=(
            "Run an optimization strategy (grid_search, few_shot_selection, "
            "or instruction_search) over the search space."
        ),
    ),

    # -- Persistence: save/load results --
    BoundaryInfo(
        name="prompt_eval.save_result",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(EvalResult),
        output_schema=None,
        description="Save an EvalResult to JSON on disk.",
    ),
    BoundaryInfo(
        name="prompt_eval.load_result",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=None,
        output_schema=_schema(EvalResult),
        description="Load an EvalResult from a JSON file on disk.",
    ),

    # -- Persistence: save/load experiments --
    BoundaryInfo(
        name="prompt_eval.save_experiment",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=_schema(Experiment),
        output_schema=None,
        description="Save an Experiment definition to JSON on disk.",
    ),
    BoundaryInfo(
        name="prompt_eval.load_experiment",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=None,
        output_schema=_schema(Experiment),
        description="Load an Experiment definition from a JSON file on disk.",
    ),

    # -- Observability reconstruction --
    BoundaryInfo(
        name="prompt_eval.load_result_from_observability",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=None,
        output_schema=_schema(EvalResult),
        description=(
            "Reconstruct a prompt_eval EvalResult from shared llm_client "
            "observability runs by execution_id."
        ),
    ),

    # -- Rubric loading --
    BoundaryInfo(
        name="prompt_eval.load_rubric",
        version="1.0.0",
        producer_project="prompt_eval",
        consumer_projects=[],
        input_schema=None,
        output_schema=_schema(Rubric),
        description=(
            "Load a scoring rubric by name or path. Searches project-local "
            "rubrics/ then built-in rubrics/."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Register and persist
# ---------------------------------------------------------------------------

for info in BOUNDARIES:
    registry.register(info)

boundaries = registry.list_by_project("prompt_eval")
print(f"\nRegistered prompt_eval boundaries: {len(boundaries)}")
for b in boundaries:
    in_label = "in" if b.input_schema else "no-in"
    out_label = "out" if b.output_schema else "no-out"
    print(f"  {b.name} v{b.version} [{in_label}/{out_label}]")

if not boundaries:
    print("ERROR: No prompt_eval boundaries registered")
    sys.exit(1)

all_boundaries = registry.list_all()
print(f"\nTotal boundaries in registry: {len(all_boundaries)}")

registry.save()
target = Path.home() / "projects" / "data" / "contract_registry.json"
print(f"Registry saved to {target}")
