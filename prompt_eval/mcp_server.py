"""MCP server for prompt_eval — exposes experiment tools to agents.

Not imported in __init__.py to avoid ImportError when fastmcp is not installed.

Run: prompt-eval-mcp (entry point) or python -m prompt_eval.mcp_server
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from prompt_eval.evaluators import contains_evaluator, exact_match_evaluator
from prompt_eval.experiment import Experiment
from prompt_eval.runner import run_experiment
from prompt_eval.stats import compare_variants
from prompt_eval.store import list_results, load_result, save_result

logger = logging.getLogger(__name__)

mcp = FastMCP(name="prompt-eval", instructions="A/B prompt testing and evaluation.")

_BUILTIN_EVALUATORS: dict[str, Any] = {
    "exact_match": exact_match_evaluator(),
    "contains": contains_evaluator(),
}


@mcp.tool
async def run_experiment_tool(
    experiment_json: str,
    evaluator_name: str | None = None,
) -> dict:
    """Run a prompt experiment and save the results.

    Args:
        experiment_json: JSON string of an Experiment definition.
        evaluator_name: Optional built-in evaluator name ("exact_match" or "contains").
    """
    experiment = Experiment.model_validate_json(experiment_json)

    evaluator = None
    if evaluator_name is not None:
        evaluator = _BUILTIN_EVALUATORS.get(evaluator_name)
        if evaluator is None:
            return {"error": f"Unknown evaluator: {evaluator_name}. Available: {list(_BUILTIN_EVALUATORS)}"}

    result = await run_experiment(experiment, evaluator=evaluator)
    path = save_result(result)

    return {
        "experiment_name": result.experiment_name,
        "variants": result.variants,
        "n_trials": len(result.trials),
        "summary": {k: v.model_dump() for k, v in result.summary.items()},
        "saved_to": str(path),
    }


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


@mcp.tool
async def list_experiments() -> dict:
    """List all experiment directories and result files."""
    all_results = list_results()
    # Group by experiment (parent directory name)
    experiments: dict[str, list[str]] = {}
    for p in all_results:
        exp_name = p.parent.name
        experiments.setdefault(exp_name, []).append(str(p))

    return {
        "experiments": experiments,
        "total_results": len(all_results),
    }


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


def main() -> None:
    """Entry point for prompt-eval-mcp script."""
    mcp.run()


if __name__ == "__main__":
    main()
