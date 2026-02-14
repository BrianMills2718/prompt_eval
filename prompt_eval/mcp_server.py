"""MCP server for prompt_eval — exposes experiment tools to agents.

Not imported in __init__.py to avoid ImportError when fastmcp is not installed.

Run: prompt-eval-mcp (entry point) or python -m prompt_eval.mcp_server
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from prompt_eval.evaluators import (
    contains_evaluator,
    exact_match_evaluator,
    kappa_evaluator,
    llm_judge_evaluator,
)
from prompt_eval.experiment import Experiment
from prompt_eval.runner import run_experiment as _run_experiment
from prompt_eval.stats import compare_variants
from prompt_eval.store import list_results, load_result, save_result

logger = logging.getLogger(__name__)

mcp = FastMCP(name="prompt-eval", instructions="A/B prompt testing and evaluation.")

def _identity_extractor(x: Any) -> list[str]:
    """Extract list of strings from output — identity if already a list, else wrap."""
    if isinstance(x, list):
        return [str(item) for item in x]
    return [str(x)]


_BUILTIN_EVALUATORS: dict[str, Any] = {
    "exact_match": exact_match_evaluator(),
    "contains": contains_evaluator(),
    "kappa": kappa_evaluator(_identity_extractor),
}


# --- Core logic (testable without MCP) ---


async def _run_experiment_impl(
    experiment_json: str,
    evaluator_name: str | None = None,
    rubric: str | None = None,
    judge_model: str | None = None,
) -> dict:
    """Run a prompt experiment and save the results."""
    experiment = Experiment.model_validate_json(experiment_json)

    evaluator = None
    if evaluator_name == "llm_judge":
        if rubric is None:
            return {"error": "llm_judge evaluator requires a 'rubric' parameter"}
        evaluator = llm_judge_evaluator(rubric, judge_model=judge_model or "gpt-5-mini")
    elif evaluator_name is not None:
        evaluator = _BUILTIN_EVALUATORS.get(evaluator_name)
        if evaluator is None:
            available = list(_BUILTIN_EVALUATORS) + ["llm_judge"]
            return {"error": f"Unknown evaluator: {evaluator_name}. Available: {available}"}

    result = await _run_experiment(experiment, evaluator=evaluator)
    path = save_result(result)

    return {
        "experiment_name": result.experiment_name,
        "variants": result.variants,
        "n_trials": len(result.trials),
        "summary": {k: v.model_dump() for k, v in result.summary.items()},
        "saved_to": str(path),
    }


async def _get_result_impl(
    experiment_name: str | None = None,
    path: str | None = None,
) -> dict:
    """Load experiment results — most recent by name, or a specific file."""
    if path is not None:
        result = load_result(Path(path))
        return result.model_dump()

    if experiment_name is not None:
        results = list_results(experiment_name)
        if not results:
            return {"error": f"No results found for '{experiment_name}'"}
        result = load_result(results[0])  # newest first
        return result.model_dump()

    return {"error": "Provide either experiment_name or path"}


async def _list_experiments_impl() -> dict:
    """List all experiment directories and result files."""
    all_results = list_results()
    experiments: dict[str, list[str]] = {}
    for p in all_results:
        exp_name = p.parent.name
        experiments.setdefault(exp_name, []).append(str(p))

    return {
        "experiments": experiments,
        "total_results": len(all_results),
    }


async def _compare_impl(
    path: str,
    variant_a: str,
    variant_b: str,
    method: str = "bootstrap",
) -> dict:
    """Compare two variants from a saved result."""
    result = load_result(Path(path))
    comparison = compare_variants(result, variant_a, variant_b, method=method)
    return {
        "variant_a": comparison.variant_a,
        "variant_b": comparison.variant_b,
        "mean_a": comparison.mean_a,
        "mean_b": comparison.mean_b,
        "difference": comparison.difference,
        "ci_lower": comparison.ci_lower,
        "ci_upper": comparison.ci_upper,
        "significant": comparison.significant,
        "method": comparison.method,
        "detail": comparison.detail,
    }


async def _evaluate_output_impl(
    output: str,
    evaluator_name: str,
    expected: str | None = None,
    rubric: str | None = None,
    judge_model: str | None = None,
) -> dict:
    """Score a single output using a built-in evaluator."""
    if evaluator_name == "llm_judge":
        if rubric is None:
            return {"error": "llm_judge evaluator requires a 'rubric' parameter"}
        evaluator = llm_judge_evaluator(rubric, judge_model=judge_model or "gpt-5-mini")
        score = await evaluator(output, expected)
    elif evaluator_name in _BUILTIN_EVALUATORS:
        evaluator = _BUILTIN_EVALUATORS[evaluator_name]
        score = evaluator(output, expected)
    else:
        available = list(_BUILTIN_EVALUATORS) + ["llm_judge"]
        return {"error": f"Unknown evaluator: {evaluator_name}. Available: {available}"}

    return {"score": score, "evaluator": evaluator_name}


# --- MCP tool registration ---


@mcp.tool
async def run_experiment_tool(
    experiment_json: str,
    evaluator_name: str | None = None,
    rubric: str | None = None,
    judge_model: str | None = None,
) -> dict:
    """Run a prompt experiment and save the results.

    Args:
        experiment_json: JSON string of an Experiment definition.
        evaluator_name: Optional evaluator ("exact_match", "contains", "kappa", "llm_judge").
        rubric: Scoring criteria for llm_judge evaluator (required if evaluator_name="llm_judge").
        judge_model: Model for llm_judge (default: gpt-5-mini).
    """
    return await _run_experiment_impl(experiment_json, evaluator_name, rubric, judge_model)


@mcp.tool
async def get_result(
    experiment_name: str | None = None,
    path: str | None = None,
) -> dict:
    """Load experiment results — most recent by name, or a specific file.

    Args:
        experiment_name: Load the most recent result for this experiment.
        path: Load a specific result file by path.
    """
    return await _get_result_impl(experiment_name, path)


@mcp.tool
async def list_experiments() -> dict:
    """List all experiment directories and result files."""
    return await _list_experiments_impl()


@mcp.tool
async def compare(
    path: str,
    variant_a: str,
    variant_b: str,
    method: str = "bootstrap",
) -> dict:
    """Compare two variants from a saved result.

    Args:
        path: Path to a saved result file.
        variant_a: Name of first variant.
        variant_b: Name of second variant.
        method: "bootstrap" (default) or "welch".
    """
    return await _compare_impl(path, variant_a, variant_b, method)


@mcp.tool
async def evaluate_output(
    output: str,
    evaluator_name: str,
    expected: str | None = None,
    rubric: str | None = None,
    judge_model: str | None = None,
) -> dict:
    """Score a single output using a built-in evaluator.

    Use this to evaluate outputs produced by external tools (e.g., a QC coding pipeline).

    Args:
        output: The output to evaluate (string or JSON string).
        evaluator_name: Evaluator to use ("exact_match", "contains", "kappa", "llm_judge").
        expected: Optional expected/reference output for comparison.
        rubric: Scoring criteria (required for llm_judge).
        judge_model: Model for llm_judge (default: gpt-5-mini).
    """
    return await _evaluate_output_impl(output, evaluator_name, expected, rubric, judge_model)


def main() -> None:
    """Entry point for prompt-eval-mcp script."""
    mcp.run()


if __name__ == "__main__":
    main()
