"""Optimization — grid search over prompt/model/temperature combinations."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
)
from prompt_eval.runner import run_experiment

logger = logging.getLogger(__name__)


class SearchSpace(BaseModel):
    """Defines the space of prompt/model/temperature combinations to search."""

    prompt_templates: list[list[dict[str, str]]] = Field(
        description="Message lists to try (each is an OpenAI chat messages list)"
    )
    models: list[str] = Field(default=["gpt-5-mini"])
    temperatures: list[float] = Field(default=[1.0])
    kwargs_variants: list[dict[str, Any]] = Field(default=[{}])


class OptimizeResult(BaseModel):
    """Result of an optimization search."""

    best_variant: str
    best_score: float
    all_results: list[EvalResult] = Field(default_factory=list)
    n_trials: int = 0
    strategy: str


async def grid_search(
    search_space: SearchSpace,
    inputs: list[ExperimentInput],
    evaluator: Callable[[Any, Optional[Any]], float],
    n_runs: int = 3,
    response_model: Any | None = None,
) -> OptimizeResult:
    """Exhaustive grid search over all combinations in the search space.

    Creates a single-variant Experiment for each combination, runs it,
    and picks the winner by highest mean_score.
    """
    combos = list(
        itertools.product(
            enumerate(search_space.prompt_templates),
            search_space.models,
            search_space.temperatures,
            enumerate(search_space.kwargs_variants),
        )
    )

    all_results: list[EvalResult] = []
    best_name: str | None = None
    best_score: float = float("-inf")

    for (tmpl_idx, messages), model, temp, (kw_idx, kwargs) in combos:
        name = f"tmpl{tmpl_idx}__{model}__{temp}__{kw_idx}"
        logger.info("Grid search: trying %s", name)

        variant = PromptVariant(
            name=name,
            messages=messages,
            model=model,
            temperature=temp,
            kwargs=kwargs,
        )
        experiment = Experiment(
            name=name,
            variants=[variant],
            inputs=inputs,
            n_runs=n_runs,
            response_model=response_model,
        )

        result = await run_experiment(experiment, evaluator=evaluator)
        all_results.append(result)

        summary = result.summary.get(name)
        score = summary.mean_score if summary and summary.mean_score is not None else float("-inf")
        if score > best_score:
            best_score = score
            best_name = name

    return OptimizeResult(
        best_variant=best_name or "",
        best_score=best_score if best_score != float("-inf") else 0.0,
        all_results=all_results,
        n_trials=len(combos),
        strategy="grid_search",
    )


async def optimize(
    search_space: SearchSpace,
    inputs: list[ExperimentInput],
    evaluator: Callable[[Any, Optional[Any]], float],
    strategy: str = "grid_search",
    n_runs: int = 3,
    budget: int | None = None,
    response_model: Any | None = None,
) -> OptimizeResult:
    """Run an optimization strategy over the search space.

    Args:
        search_space: Combinations to search.
        inputs: Test inputs.
        evaluator: Scoring function(output, expected) -> float.
        strategy: "grid_search" (default). "few_shot_selection" and
            "instruction_search" are planned but not yet implemented.
        n_runs: Runs per combination.
        budget: Max combinations to try (accepted but ignored by grid_search).
        response_model: Optional Pydantic model for structured output.

    Returns:
        OptimizeResult with best variant and all results.
    """
    if strategy == "grid_search":
        return await grid_search(
            search_space, inputs, evaluator, n_runs=n_runs, response_model=response_model,
        )
    elif strategy in ("few_shot_selection", "instruction_search"):
        raise NotImplementedError(f"Strategy '{strategy}' is not yet implemented.")
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'grid_search'.")
