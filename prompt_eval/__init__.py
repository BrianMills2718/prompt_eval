"""Public package surface for prompt-centric experiment execution and analysis.

This module exposes the stable concepts a caller needs to define experiments,
run prompt variants, score outputs, compare variants, and use optional helpers
such as acceptable-set evaluation. It intentionally re-exports the compact
public surface rather than requiring callers to know the internal module
layout.
"""

from prompt_eval.evaluators import (
    EvalScore,
    RubricDimension,
    contains_evaluator,
    exact_match_evaluator,
    kappa_evaluator,
    llm_judge_dimensional_evaluator,
    llm_judge_evaluator,
)
from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    VariantSummary,
)
from prompt_eval.observability import PromptEvalObservabilityConfig
from prompt_eval.optimize import (
    FewShotPool,
    OptimizeResult,
    SearchSpace,
    few_shot_selection,
    grid_search,
    instruction_search,
    optimize,
)
from prompt_eval.prompt_assets import build_prompt_variant_from_ref
from prompt_eval.query import load_result_from_observability
from prompt_eval.runner import run_experiment
from prompt_eval.stats import compare_variants
from prompt_eval.golden_set import AlternativeRecord, GoldenSetManager, JudgeDecision
from prompt_eval.store import (
    list_results,
    load_experiment,
    load_result,
    save_experiment,
    save_result,
)

__all__ = [
    # Models
    "Experiment",
    "ExperimentInput",
    "EvalResult",
    "PromptVariant",
    "PromptEvalObservabilityConfig",
    "Trial",
    "VariantSummary",
    "build_prompt_variant_from_ref",
    "load_result_from_observability",
    # Runner
    "run_experiment",
    # Stats
    "compare_variants",
    # Persistence
    "save_result",
    "load_result",
    "save_experiment",
    "load_experiment",
    "list_results",
    # Golden Set
    "GoldenSetManager",
    "AlternativeRecord",
    "JudgeDecision",
    # Evaluators
    "kappa_evaluator",
    "exact_match_evaluator",
    "contains_evaluator",
    "llm_judge_evaluator",
    "llm_judge_dimensional_evaluator",
    "EvalScore",
    "RubricDimension",
    # Optimization
    "optimize",
    "grid_search",
    "few_shot_selection",
    "instruction_search",
    "SearchSpace",
    "FewShotPool",
    "OptimizeResult",
]
