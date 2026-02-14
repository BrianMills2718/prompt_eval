"""prompt_eval — A/B prompt testing and evaluation."""

from prompt_eval.evaluators import (
    contains_evaluator,
    exact_match_evaluator,
    kappa_evaluator,
)
from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    VariantSummary,
)
from prompt_eval.optimize import OptimizeResult, SearchSpace, grid_search, optimize
from prompt_eval.runner import run_experiment
from prompt_eval.stats import compare_variants
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
    "Trial",
    "VariantSummary",
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
    # Evaluators
    "kappa_evaluator",
    "exact_match_evaluator",
    "contains_evaluator",
    # Optimization
    "optimize",
    "grid_search",
    "SearchSpace",
    "OptimizeResult",
]
