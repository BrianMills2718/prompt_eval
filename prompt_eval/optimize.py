"""Optimization — search strategies for prompt/model/temperature combinations."""

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
from llm_client import acall_llm
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


class FewShotPool(BaseModel):
    """Pool of examples for few-shot selection search."""

    examples: list[str] = Field(description="Individual example strings to combine")
    k: int = Field(default=2, description="Number of examples per combination")
    base_messages: list[dict[str, str]] = Field(
        description="Base message template. Use {examples} placeholder where examples should be injected."
    )
    separator: str = Field(default="\n\n", description="Separator between examples")
    model: str = Field(default="gpt-5-mini")
    temperature: float = Field(default=1.0)


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


def _inject_examples(
    base_messages: list[dict[str, str]],
    examples: list[str],
    separator: str,
) -> list[dict[str, str]]:
    """Replace {examples} placeholder in messages with joined example strings."""
    joined = separator.join(examples)
    return [
        {**msg, "content": msg["content"].replace("{examples}", joined)}
        for msg in base_messages
    ]


async def few_shot_selection(
    pool: FewShotPool,
    inputs: list[ExperimentInput],
    evaluator: Callable[[Any, Optional[Any]], float],
    n_runs: int = 3,
    budget: int | None = None,
    response_model: Any | None = None,
) -> OptimizeResult:
    """Search over combinations of K examples from the pool.

    Tries all C(n, k) combinations of examples, injecting each into the
    base prompt template via the {examples} placeholder. Picks the combination
    with the highest mean evaluator score.

    Args:
        pool: FewShotPool with examples, k, and base_messages.
        inputs: Test inputs.
        evaluator: Scoring function(output, expected) -> float.
        n_runs: Runs per combination.
        budget: Max combinations to try. If None, tries all C(n, k).
        response_model: Optional Pydantic model for structured output.
    """
    combos = list(itertools.combinations(pool.examples, pool.k))
    if budget is not None and len(combos) > budget:
        import random
        combos = random.sample(combos, budget)

    all_results: list[EvalResult] = []
    best_name: str | None = None
    best_score: float = float("-inf")

    for idx, example_combo in enumerate(combos):
        name = f"fewshot_{idx}"
        messages = _inject_examples(pool.base_messages, list(example_combo), pool.separator)
        logger.info("Few-shot selection: trying %s (%d examples)", name, len(example_combo))

        variant = PromptVariant(
            name=name,
            messages=messages,
            model=pool.model,
            temperature=pool.temperature,
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
        strategy="few_shot_selection",
    )


async def instruction_search(
    base_instruction: str,
    inputs: list[ExperimentInput],
    evaluator: Callable[[Any, Optional[Any]], float],
    n_iterations: int = 5,
    n_rewrites: int = 3,
    n_runs: int = 3,
    model: str = "gpt-5-mini",
    rewrite_model: str = "gpt-5-mini",
    response_model: Any | None = None,
) -> OptimizeResult:
    """Hill-climbing instruction search via LLM-generated rewrites.

    Starts with a base instruction, asks the LLM to generate n_rewrites
    alternative instructions, evaluates each, and keeps the best as the
    starting point for the next iteration.

    Args:
        base_instruction: Starting instruction text (user message content).
        inputs: Test inputs.
        evaluator: Scoring function(output, expected) -> float.
        n_iterations: Number of hill-climbing iterations.
        n_rewrites: Number of rewrites to generate per iteration.
        n_runs: Runs per candidate instruction.
        model: Model to evaluate instructions with.
        rewrite_model: Model to generate rewrites with.
        response_model: Optional Pydantic model for structured output.
    """
    all_results: list[EvalResult] = []
    current_best_instruction = base_instruction
    current_best_score: float = float("-inf")
    best_name: str = "iter0_base"

    # Evaluate the base instruction first
    base_variant = PromptVariant(
        name="iter0_base",
        messages=[{"role": "user", "content": base_instruction}],
        model=model,
    )
    base_experiment = Experiment(
        name="iter0_base",
        variants=[base_variant],
        inputs=inputs,
        n_runs=n_runs,
        response_model=response_model,
    )
    base_result = await run_experiment(base_experiment, evaluator=evaluator)
    all_results.append(base_result)

    summary = base_result.summary.get("iter0_base")
    if summary and summary.mean_score is not None:
        current_best_score = summary.mean_score
    logger.info("Instruction search: base score = %.4f", current_best_score)

    for iteration in range(n_iterations):
        # Generate rewrites
        rewrite_prompt = (
            f"Rewrite this instruction to be more effective. "
            f"Generate exactly {n_rewrites} alternative versions, "
            f"each on its own line, separated by '---'. "
            f"Keep the {{input}} placeholder if present.\n\n"
            f"Original instruction:\n{current_best_instruction}"
        )
        rewrite_response, _ = await acall_llm(
            rewrite_model,
            [{"role": "user", "content": rewrite_prompt}],
        )

        # Parse rewrites (split on ---)
        rewrites = [r.strip() for r in str(rewrite_response).split("---") if r.strip()]
        rewrites = rewrites[:n_rewrites]  # cap at n_rewrites

        if not rewrites:
            logger.warning("Instruction search iter %d: no rewrites generated", iteration)
            continue

        # Evaluate each rewrite
        for rewrite_idx, rewrite in enumerate(rewrites):
            name = f"iter{iteration + 1}_rewrite{rewrite_idx}"
            variant = PromptVariant(
                name=name,
                messages=[{"role": "user", "content": rewrite}],
                model=model,
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
            if score > current_best_score:
                current_best_score = score
                current_best_instruction = rewrite
                best_name = name
                logger.info(
                    "Instruction search iter %d: new best %.4f from %s",
                    iteration + 1, score, name,
                )

    return OptimizeResult(
        best_variant=best_name,
        best_score=current_best_score if current_best_score != float("-inf") else 0.0,
        all_results=all_results,
        n_trials=len(all_results),
        strategy="instruction_search",
    )


async def optimize(
    search_space: SearchSpace,
    inputs: list[ExperimentInput],
    evaluator: Callable[[Any, Optional[Any]], float],
    strategy: str = "grid_search",
    n_runs: int = 3,
    budget: int | None = None,
    response_model: Any | None = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Run an optimization strategy over the search space.

    Args:
        search_space: Combinations to search (used by grid_search).
        inputs: Test inputs.
        evaluator: Scoring function(output, expected) -> float.
        strategy: "grid_search" (default), "few_shot_selection", or "instruction_search".
        n_runs: Runs per combination.
        budget: Max combinations to try (few_shot_selection) or ignored (grid_search).
        response_model: Optional Pydantic model for structured output.
        **kwargs: Strategy-specific arguments:
            few_shot_selection: pool (FewShotPool) — required.
            instruction_search: base_instruction (str) — required.
                Optional: n_iterations, n_rewrites, model, rewrite_model.

    Returns:
        OptimizeResult with best variant and all results.
    """
    if strategy == "grid_search":
        return await grid_search(
            search_space, inputs, evaluator, n_runs=n_runs, response_model=response_model,
        )
    elif strategy == "few_shot_selection":
        pool = kwargs.get("pool")
        if pool is None:
            raise ValueError("few_shot_selection requires 'pool' (FewShotPool) kwarg.")
        return await few_shot_selection(
            pool, inputs, evaluator, n_runs=n_runs, budget=budget, response_model=response_model,
        )
    elif strategy == "instruction_search":
        base_instruction = kwargs.get("base_instruction")
        if base_instruction is None:
            raise ValueError("instruction_search requires 'base_instruction' (str) kwarg.")
        return await instruction_search(
            base_instruction,
            inputs,
            evaluator,
            n_iterations=kwargs.get("n_iterations", 5),
            n_rewrites=kwargs.get("n_rewrites", 3),
            n_runs=n_runs,
            model=kwargs.get("model", "gpt-5-mini"),
            rewrite_model=kwargs.get("rewrite_model", "gpt-5-mini"),
            response_model=response_model,
        )
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'grid_search', 'few_shot_selection', or 'instruction_search'.")
