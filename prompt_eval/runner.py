"""Experiment runner — executes prompt variants and collects trials."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Optional

from llm_client import acall_llm, acall_llm_structured

from prompt_eval.evaluators import EvalScore
from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    VariantSummary,
)

logger = logging.getLogger(__name__)


async def run_experiment(
    experiment: Experiment,
    evaluator: Optional[Callable[[Any, Optional[Any]], float]] = None,
    corpus_evaluator: Optional[Callable[[list[Any]], float]] = None,
) -> EvalResult:
    """Run all variants against all inputs, collect trials, compute summaries.

    Args:
        experiment: The experiment definition.
        evaluator: Optional function(output, expected) -> float score.
            If None, trials are collected without scores.
        corpus_evaluator: Optional function(list[outputs]) -> float | EvalScore.
            Runs once per variant on all its successful outputs.
            Returns a single aggregate score per variant.

    Returns:
        EvalResult with all trials and per-variant summaries.
    """
    trials: list[Trial] = []

    for variant in experiment.variants:
        for inp in experiment.inputs:
            for run_idx in range(experiment.n_runs):
                logger.info(
                    "Running %s / %s / run %d",
                    variant.name, inp.id, run_idx + 1,
                )
                trial = await _run_single_trial(
                    variant, inp, experiment.response_model, evaluator,
                )
                trials.append(trial)

    # Warn about unreliable variants
    for variant in experiment.variants:
        expected = len(experiment.inputs) * experiment.n_runs
        errors = sum(1 for t in trials if t.variant_name == variant.name and t.error is not None)
        if errors > expected * 0.5:
            logger.warning(
                "Variant '%s' has %d/%d failed trials — results are unreliable",
                variant.name, errors, expected,
            )

    # Corpus-level evaluation (one call per variant)
    corpus_results: dict[str, tuple[Optional[float], Optional[dict]]] = {}
    if corpus_evaluator is not None:
        for variant in experiment.variants:
            outputs = [
                t.output for t in trials
                if t.variant_name == variant.name and t.output is not None
            ]
            if not outputs:
                logger.warning("No outputs for variant '%s' — skipping corpus eval", variant.name)
                corpus_results[variant.name] = (None, None)
                continue
            try:
                if inspect.iscoroutinefunction(corpus_evaluator):
                    result = await corpus_evaluator(outputs)
                else:
                    result = corpus_evaluator(outputs)

                if isinstance(result, EvalScore):
                    corpus_results[variant.name] = (result.score, result.dimension_scores)
                elif isinstance(result, (int, float)):
                    corpus_results[variant.name] = (float(result), None)
            except Exception as e:
                logger.warning("Corpus evaluator failed for '%s': %s", variant.name, e)
                corpus_results[variant.name] = (None, None)

    # Build summaries
    summary = _build_summaries(trials, [v.name for v in experiment.variants], corpus_results or None)

    return EvalResult(
        experiment_name=experiment.name,
        variants=[v.name for v in experiment.variants],
        trials=trials,
        summary=summary,
    )


async def _run_single_trial(
    variant: PromptVariant,
    inp: ExperimentInput,
    response_model: Optional[Any],
    evaluator: Optional[Callable],
) -> Trial:
    """Execute one LLM call and return a Trial."""
    # Substitute {input} placeholder in user messages
    messages = _substitute_input(variant.messages, inp.content)

    start = time.monotonic()
    try:
        trace_id = f"prompt_eval.run.{variant.name}.{inp.id}"
        if response_model is not None:
            result, meta = await acall_llm_structured(
                variant.model,
                messages,
                response_model=response_model,
                temperature=variant.temperature,
                task="prompt_eval.run",
                trace_id=trace_id,
                max_budget=0,
                **variant.kwargs,
            )
        else:
            meta = await acall_llm(
                variant.model,
                messages,
                temperature=variant.temperature,
                task="prompt_eval.run",
                trace_id=trace_id,
                max_budget=0,
                **variant.kwargs,
            )
            result = meta.content

        latency_ms = (time.monotonic() - start) * 1000

        score = None
        dimension_scores = None
        reasoning = None
        if evaluator is not None:
            try:
                if inspect.iscoroutinefunction(evaluator):
                    eval_result = await evaluator(result, inp.expected)
                else:
                    eval_result = evaluator(result, inp.expected)

                if isinstance(eval_result, EvalScore):
                    score = eval_result.score
                    dimension_scores = eval_result.dimension_scores
                    reasoning = eval_result.reasoning
                elif isinstance(eval_result, (int, float)):
                    score = float(eval_result)
            except Exception as e:
                logger.warning("Evaluator failed for %s/%s: %s", variant.name, inp.id, e)

        return Trial(
            variant_name=variant.name,
            input_id=inp.id,
            output=result,
            score=score,
            dimension_scores=dimension_scores,
            reasoning=reasoning,
            cost=meta.cost,
            latency_ms=latency_ms,
            tokens_used=meta.usage.get("total_tokens", 0) if meta.usage else 0,
        )

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        logger.error("Trial failed for %s/%s: %s", variant.name, inp.id, e)
        return Trial(
            variant_name=variant.name,
            input_id=inp.id,
            output=None,
            error=str(e),
            latency_ms=latency_ms,
        )


def _substitute_input(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    """Replace {input} placeholder in message content."""
    return [
        {**msg, "content": msg["content"].replace("{input}", content)}
        for msg in messages
    ]


def _build_summaries(
    trials: list[Trial],
    variant_names: list[str],
    corpus_results: Optional[dict[str, tuple[Optional[float], Optional[dict]]]] = None,
) -> dict[str, VariantSummary]:
    """Compute per-variant aggregate stats."""
    import statistics

    summaries = {}
    for name in variant_names:
        variant_trials = [t for t in trials if t.variant_name == name]
        scores = [t.score for t in variant_trials if t.score is not None]
        errors = [t for t in variant_trials if t.error is not None]

        # Aggregate dimension scores
        dimension_means = None
        dim_trials = [t for t in variant_trials if t.dimension_scores is not None]
        if dim_trials:
            all_dims: dict[str, list[float]] = {}
            for t in dim_trials:
                for dim_name, dim_score in t.dimension_scores.items():
                    all_dims.setdefault(dim_name, []).append(dim_score)
            dimension_means = {
                dim_name: statistics.mean(dim_scores)
                for dim_name, dim_scores in all_dims.items()
            }

        # Corpus-level scores
        corpus_score = None
        corpus_dimension_scores = None
        if corpus_results and name in corpus_results:
            corpus_score, corpus_dimension_scores = corpus_results[name]

        summaries[name] = VariantSummary(
            variant_name=name,
            n_trials=len(variant_trials),
            n_errors=len(errors),
            mean_score=statistics.mean(scores) if scores else None,
            std_score=statistics.stdev(scores) if len(scores) >= 2 else None,
            dimension_means=dimension_means,
            mean_cost=(
                statistics.mean(t.cost for t in variant_trials if t.error is None)
                if any(t.error is None for t in variant_trials)
                else 0.0
            ),
            mean_latency_ms=statistics.mean(t.latency_ms for t in variant_trials),
            total_tokens=sum(t.tokens_used for t in variant_trials),
            corpus_score=corpus_score,
            corpus_dimension_scores=corpus_dimension_scores,
        )

    return summaries
