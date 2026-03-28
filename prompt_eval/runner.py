"""Experiment runner — executes prompt variants and collects trials."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from llm_client import acall_llm, acall_llm_structured
from llm_client.observability import (
    activate_experiment_run,
    activate_feature_profile,
    finish_run as finish_observability_run,
    log_experiment_aggregate,
    log_item as log_observability_item,
    start_run as start_observability_run,
)

from prompt_eval.evaluators import EvalScore
from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PrecomputedOutput,
    PromptVariant,
    Trial,
    VariantSummary,
)
from prompt_eval.observability import (
    PromptEvalObservabilityConfig,
    _corpus_aggregate_metrics,
    _corpus_aggregate_provenance,
    _experiment_execution_id,
    _log_item_payload,
    _resolve_observability_config,
    _run_config_payload,
    _run_provenance_payload,
    _summary_metrics_for_run,
    _trace_id,
    _variant_call_kwargs,
    _variant_max_budget,
    _variant_task,
)

logger = logging.getLogger(__name__)
_PROMPT_EVAL_FEATURE_PROFILE = {
    "name": "prompt_eval_default",
    "features": {"experiment_context": True},
}
TrialEvaluator = Callable[[Any, Optional[Any]], float | EvalScore | Awaitable[float | EvalScore]]
CorpusEvaluator = Callable[[list[Any]], float | EvalScore | Awaitable[float | EvalScore]]


async def run_experiment(
    experiment: Experiment,
    evaluator: Optional[TrialEvaluator] = None,
    corpus_evaluator: Optional[CorpusEvaluator] = None,
    observability: bool | PromptEvalObservabilityConfig | None = True,
) -> EvalResult:
    """Run all variants against all inputs, collect trials, compute summaries.

    Args:
        experiment: The experiment definition.
        evaluator: Optional per-trial evaluator. It may be synchronous or
            asynchronous and must resolve to either a numeric score or an
            ``EvalScore``. If None, trials are collected without scores.
            Evaluator failures are logged and leave the affected trial
            unscored rather than aborting the whole experiment.
        corpus_evaluator: Optional corpus-level evaluator. It may be
            synchronous or asynchronous and must resolve to either a numeric
            score or an ``EvalScore``. Runs once per variant on all its
            successful outputs. Corpus-evaluator failures are logged and leave
            corpus summary fields unset rather than aborting the whole
            experiment.
        observability: Whether to emit shared llm_client observability records.
            Pass ``False`` to disable emission, ``True`` for default shared
            logging, or ``PromptEvalObservabilityConfig`` to override project,
            dataset, scenario, phase, seed, or provenance metadata.

    Returns:
        EvalResult with all trials and per-variant summaries.
    """
    trials: list[Trial] = []
    run_ids_by_variant: dict[str, list[str]] = {}
    observability_config = _resolve_observability_config(
        observability,
        default_phase="evaluation",
    )
    execution_id = _experiment_execution_id(observability_config)

    for variant in experiment.variants:
        for run_idx in range(experiment.n_runs):
            run_trials: list[tuple[Trial, ExperimentInput]] = []
            run_id: str | None = None
            active_run_context = None
            active_profile_context = None
            if observability_config is not None:
                run_id = start_observability_run(
                    dataset=observability_config.dataset or experiment.name,
                    model=variant.model,
                    task=_variant_task(variant),
                    config=_run_config_payload(variant=variant, experiment=experiment),
                    condition_id=variant.name,
                    seed=observability_config.seed,
                    replicate=run_idx,
                    scenario_id=observability_config.scenario_id or experiment.name,
                    phase=observability_config.phase,
                    metrics_schema=["score"] if evaluator is not None else None,
                    provenance=_run_provenance_payload(
                        experiment=experiment,
                        variant=variant,
                        execution_id=execution_id,
                        evaluator=evaluator,
                        corpus_evaluator=corpus_evaluator,
                        observability=observability_config,
                    ),
                    feature_profile=_PROMPT_EVAL_FEATURE_PROFILE,
                    allow_missing_agent_spec=True,
                    missing_agent_spec_reason=(
                        "prompt_eval experiment runs are evaluation workloads, "
                        "not agent-spec governed agent tasks"
                    ),
                    project=observability_config.project,
                )
                run_ids_by_variant.setdefault(variant.name, []).append(run_id)
                active_run_context = activate_experiment_run(run_id)
                active_profile_context = activate_feature_profile(_PROMPT_EVAL_FEATURE_PROFILE)
                active_run_context.__enter__()
                active_profile_context.__enter__()

            try:
                for inp in experiment.inputs:
                    trace_id = _trace_id(
                        execution_id=execution_id,
                        condition_id=variant.name,
                        replicate=run_idx,
                        item_id=inp.id,
                    )
                    logger.info(
                        "Running %s / %s / run %d",
                        variant.name, inp.id, run_idx + 1,
                    )
                    trial = await _run_single_trial(
                        variant,
                        inp,
                        experiment.response_model,
                        evaluator,
                        run_idx=run_idx,
                        trace_id=trace_id,
                    )
                    trials.append(trial)
                    run_trials.append((trial, inp))
                    if run_id is not None:
                        log_observability_item(
                            run_id=run_id,
                            **_log_item_payload(trial=trial, inp=inp),
                        )
                if run_id is not None:
                    finish_observability_run(
                        run_id=run_id,
                        summary_metrics=_summary_metrics_for_run(
                            [trial for trial, _inp in run_trials]
                        ),
                    )
            finally:
                if active_profile_context is not None:
                    active_profile_context.__exit__(None, None, None)
                if active_run_context is not None:
                    active_run_context.__exit__(None, None, None)

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
                result = corpus_evaluator(outputs)
                if inspect.isawaitable(result):
                    result = await result
                corpus_results[variant.name] = _coerce_corpus_evaluator_result(result)
            except Exception as e:
                logger.warning("Corpus evaluator failed for '%s': %s", variant.name, e)
                corpus_results[variant.name] = (None, None)

    if observability_config is not None and corpus_evaluator is not None:
        for variant in experiment.variants:
            corpus_score, corpus_dimension_scores = corpus_results.get(variant.name, (None, None))
            aggregate_metrics = _corpus_aggregate_metrics(
                score=corpus_score,
                dimension_scores=corpus_dimension_scores,
            )
            if aggregate_metrics is None:
                continue
            log_experiment_aggregate(
                dataset=observability_config.dataset or experiment.name,
                family_id=execution_id,
                aggregate_type="prompt_eval.corpus_evaluator",
                condition_id=variant.name,
                scenario_id=observability_config.scenario_id or experiment.name,
                phase=observability_config.phase,
                metrics=aggregate_metrics,
                provenance=_corpus_aggregate_provenance(
                    experiment=experiment,
                    variant=variant,
                    execution_id=execution_id,
                    corpus_evaluator=corpus_evaluator,
                ),
                source_run_ids=run_ids_by_variant.get(variant.name, []),
                project=observability_config.project,
            )

    # Build summaries
    summary = _build_summaries(trials, [v.name for v in experiment.variants], corpus_results or None)
    return EvalResult(
        experiment_name=experiment.name,
        execution_id=execution_id,
        variants=[v.name for v in experiment.variants],
        trials=trials,
        summary=summary,
    )


async def evaluate_precomputed_variants(
    *,
    experiment_name: str,
    inputs: list[ExperimentInput],
    outputs: list[PrecomputedOutput],
    evaluator: Optional[TrialEvaluator] = None,
    corpus_evaluator: Optional[CorpusEvaluator] = None,
    observability: bool | PromptEvalObservabilityConfig | None = True,
) -> EvalResult:
    """Score and compare externally produced outputs through the normal result contract.

    This path is for frozen outputs produced outside `prompt_eval`, such as
    archived reports from another repo. It deliberately does not execute the
    subject system; it only evaluates and summarizes the supplied outputs.
    """
    input_by_id = _validate_input_cases(inputs)
    variant_names, replicate_values = _validate_precomputed_outputs(
        outputs,
        input_by_id=input_by_id,
    )
    observability_config = _resolve_observability_config(
        observability,
        default_phase="evaluation",
    )
    execution_id = _experiment_execution_id(observability_config)

    trials: list[Trial] = []
    trials_by_variant_replicate: dict[tuple[str, int], list[tuple[Trial, ExperimentInput]]] = {}
    for output in outputs:
        inp = input_by_id[output.input_id]
        trace_id = _trace_id(
            execution_id=execution_id,
            condition_id=output.variant_name,
            replicate=output.replicate,
            item_id=inp.id,
        )
        trial = await _trial_from_precomputed_output(
            output,
            inp,
            evaluator,
            trace_id=trace_id,
        )
        trials.append(trial)
        trials_by_variant_replicate.setdefault((output.variant_name, output.replicate), []).append((trial, inp))

    run_ids_by_variant: dict[str, list[str]] = {}
    if observability_config is not None:
        for variant_name in variant_names:
            for replicate in replicate_values:
                run_trials = trials_by_variant_replicate[(variant_name, replicate)]
                run_id = start_observability_run(
                    dataset=observability_config.dataset or experiment_name,
                    model="precomputed",
                    task="prompt_eval.precomputed",
                    config={
                        "precomputed": True,
                        "variant_name": variant_name,
                        "input_count": len(inputs),
                    },
                    condition_id=variant_name,
                    seed=observability_config.seed,
                    replicate=replicate,
                    scenario_id=observability_config.scenario_id or experiment_name,
                    phase=observability_config.phase,
                    metrics_schema=["score"] if evaluator is not None else None,
                    provenance={
                        "source_package": "prompt_eval",
                        "experiment_name": experiment_name,
                        "experiment_execution_id": execution_id,
                        "variant_name": variant_name,
                        "variant_count": len(variant_names),
                        "input_count": len(inputs),
                        "n_runs": len(replicate_values),
                        "prompt_source": "precomputed_outputs",
                        "task": "prompt_eval.precomputed",
                        "llm_task": "prompt_eval.precomputed",
                        "precomputed": True,
                    },
                    feature_profile=_PROMPT_EVAL_FEATURE_PROFILE,
                    allow_missing_agent_spec=True,
                    missing_agent_spec_reason=(
                        "prompt_eval precomputed comparison runs are evaluation workloads, "
                        "not agent-spec governed agent tasks"
                    ),
                    project=observability_config.project,
                )
                run_ids_by_variant.setdefault(variant_name, []).append(run_id)
                active_run_context = activate_experiment_run(run_id)
                active_profile_context = activate_feature_profile(_PROMPT_EVAL_FEATURE_PROFILE)
                active_run_context.__enter__()
                active_profile_context.__enter__()
                try:
                    for trial, inp in run_trials:
                        log_observability_item(
                            run_id=run_id,
                            **_log_item_payload(trial=trial, inp=inp),
                        )
                    finish_observability_run(
                        run_id=run_id,
                        summary_metrics=_summary_metrics_for_run(
                            [trial for trial, _inp in run_trials]
                        ),
                    )
                finally:
                    active_profile_context.__exit__(None, None, None)
                    active_run_context.__exit__(None, None, None)

    corpus_results: dict[str, tuple[Optional[float], Optional[dict]]] = {}
    if corpus_evaluator is not None:
        for variant_name in variant_names:
            variant_outputs = [
                trial.output
                for trial in trials
                if trial.variant_name == variant_name and trial.output is not None
            ]
            if not variant_outputs:
                logger.warning("No outputs for variant '%s' — skipping corpus eval", variant_name)
                corpus_results[variant_name] = (None, None)
                continue
            try:
                result = corpus_evaluator(variant_outputs)
                if inspect.isawaitable(result):
                    result = await result
                corpus_results[variant_name] = _coerce_corpus_evaluator_result(result)
            except Exception as e:
                logger.warning("Corpus evaluator failed for '%s': %s", variant_name, e)
                corpus_results[variant_name] = (None, None)

    if observability_config is not None and corpus_evaluator is not None:
        for variant_name in variant_names:
            corpus_score, corpus_dimension_scores = corpus_results.get(variant_name, (None, None))
            aggregate_metrics = _corpus_aggregate_metrics(
                score=corpus_score,
                dimension_scores=corpus_dimension_scores,
            )
            if aggregate_metrics is None:
                continue
            log_experiment_aggregate(
                dataset=observability_config.dataset or experiment_name,
                family_id=execution_id,
                aggregate_type="prompt_eval.corpus_evaluator",
                condition_id=variant_name,
                scenario_id=observability_config.scenario_id or experiment_name,
                phase=observability_config.phase,
                metrics=aggregate_metrics,
                provenance={
                    "source_package": "prompt_eval",
                    "experiment_name": experiment_name,
                    "experiment_execution_id": execution_id,
                    "variant_name": variant_name,
                    "prompt_source": "precomputed_outputs",
                    "precomputed": True,
                },
                source_run_ids=run_ids_by_variant.get(variant_name, []),
                project=observability_config.project,
            )

    summary = _build_summaries(trials, variant_names, corpus_results or None)
    return EvalResult(
        experiment_name=experiment_name,
        execution_id=execution_id,
        variants=variant_names,
        trials=trials,
        summary=summary,
    )


async def _run_single_trial(
    variant: PromptVariant,
    inp: ExperimentInput,
    response_model: Optional[Any],
    evaluator: Optional[TrialEvaluator],
    *,
    run_idx: int,
    trace_id: str,
) -> Trial:
    """Execute one LLM call, apply optional evaluation, and return a Trial."""
    # Substitute {input} placeholder in user messages
    messages = _substitute_input(variant.messages, inp.content)

    start = time.monotonic()
    try:
        if response_model is not None:
            result, meta = await acall_llm_structured(
                variant.model,
                messages,
                response_model=response_model,
                temperature=variant.temperature,
                task=_variant_task(variant),
                trace_id=trace_id,
                max_budget=_variant_max_budget(variant),
                **_variant_call_kwargs(variant),
            )
        else:
            meta = await acall_llm(
                variant.model,
                messages,
                temperature=variant.temperature,
                task=_variant_task(variant),
                trace_id=trace_id,
                max_budget=_variant_max_budget(variant),
                **_variant_call_kwargs(variant),
            )
            result = meta.content

        latency_ms = (time.monotonic() - start) * 1000

        score = None
        dimension_scores = None
        reasoning = None
        if evaluator is not None:
            try:
                eval_result = evaluator(result, inp.expected)
                if inspect.isawaitable(eval_result):
                    eval_result = await eval_result
                score, dimension_scores, reasoning = _coerce_trial_evaluator_result(eval_result)
            except Exception as e:
                logger.warning("Evaluator failed for %s/%s: %s", variant.name, inp.id, e)

        return Trial(
            variant_name=variant.name,
            input_id=inp.id,
            replicate=run_idx,
            output=result,
            score=score,
            dimension_scores=dimension_scores,
            reasoning=reasoning,
            cost=meta.cost,
            latency_ms=latency_ms,
            tokens_used=meta.usage.get("total_tokens", 0) if meta.usage else 0,
            trace_id=trace_id,
        )

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        logger.error("Trial failed for %s/%s: %s", variant.name, inp.id, e)
        return Trial(
            variant_name=variant.name,
            input_id=inp.id,
            replicate=run_idx,
            output=None,
            error=str(e),
            latency_ms=latency_ms,
            trace_id=trace_id,
        )


async def _trial_from_precomputed_output(
    output: PrecomputedOutput,
    inp: ExperimentInput,
    evaluator: Optional[TrialEvaluator],
    *,
    trace_id: str,
) -> Trial:
    """Apply the normal evaluator contract to one externally produced output."""
    if output.error is not None:
        return Trial(
            variant_name=output.variant_name,
            input_id=inp.id,
            replicate=output.replicate,
            output=None,
            error=output.error,
            trace_id=trace_id,
        )

    score, dimension_scores, reasoning = await _evaluate_output(
        output.output,
        inp,
        evaluator,
        variant_name=output.variant_name,
    )
    return Trial(
        variant_name=output.variant_name,
        input_id=inp.id,
        replicate=output.replicate,
        output=output.output,
        score=score,
        dimension_scores=dimension_scores,
        reasoning=reasoning,
        trace_id=trace_id,
    )


async def _evaluate_output(
    output: Any,
    inp: ExperimentInput,
    evaluator: Optional[TrialEvaluator],
    *,
    variant_name: str,
) -> tuple[float | None, dict[str, float] | None, str | None]:
    """Apply the per-trial evaluator contract to an already-available output."""
    if evaluator is None:
        return None, None, None
    try:
        eval_result = evaluator(output, inp.expected)
        if inspect.isawaitable(eval_result):
            eval_result = await eval_result
        return _coerce_trial_evaluator_result(eval_result)
    except Exception as e:
        logger.warning("Evaluator failed for %s/%s: %s", variant_name, inp.id, e)
        return None, None, None


def _validate_input_cases(inputs: list[ExperimentInput]) -> dict[str, ExperimentInput]:
    """Fail loudly if the caller supplies duplicate or empty input IDs."""
    input_by_id: dict[str, ExperimentInput] = {}
    for inp in inputs:
        if not inp.id:
            raise ValueError("ExperimentInput.id must be non-empty for precomputed evaluation.")
        if inp.id in input_by_id:
            raise ValueError(f"Duplicate ExperimentInput.id for precomputed evaluation: {inp.id!r}.")
        input_by_id[inp.id] = inp
    return input_by_id


def _validate_precomputed_outputs(
    outputs: list[PrecomputedOutput],
    *,
    input_by_id: dict[str, ExperimentInput],
) -> tuple[list[str], list[int]]:
    """Validate rectangular coverage across variants, inputs, and replicates."""
    if not outputs:
        raise ValueError("Precomputed evaluation requires at least one output.")
    seen_keys: set[tuple[str, str, int]] = set()
    variant_names: list[str] = []
    seen_variants: set[str] = set()
    replicate_values: set[int] = set()
    coverage: dict[tuple[str, int], set[str]] = {}

    for output in outputs:
        if output.input_id not in input_by_id:
            raise ValueError(
                f"Precomputed output references unknown input_id={output.input_id!r}."
            )
        key = (output.variant_name, output.input_id, output.replicate)
        if key in seen_keys:
            raise ValueError(f"Duplicate precomputed output for {key!r}.")
        seen_keys.add(key)
        if output.variant_name not in seen_variants:
            variant_names.append(output.variant_name)
            seen_variants.add(output.variant_name)
        replicate_values.add(output.replicate)
        coverage.setdefault((output.variant_name, output.replicate), set()).add(output.input_id)

    expected_inputs = set(input_by_id)
    ordered_replicates = sorted(replicate_values)
    if ordered_replicates != list(range(len(ordered_replicates))):
        raise ValueError(
            "Precomputed outputs must use contiguous replicate indices starting at 0."
        )
    for variant_name in variant_names:
        for replicate in ordered_replicates:
            seen_input_ids = coverage.get((variant_name, replicate))
            if seen_input_ids != expected_inputs:
                raise ValueError(
                    "Precomputed outputs must cover every input for each "
                    f"(variant, replicate). Missing coverage for {(variant_name, replicate)!r}."
                )
    return variant_names, ordered_replicates


def _substitute_input(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    """Replace {input} placeholder in message content."""
    return [
        {**msg, "content": msg["content"].replace("{input}", content)}
        for msg in messages
    ]


def _coerce_trial_evaluator_result(
    eval_result: float | EvalScore,
) -> tuple[float, dict[str, float] | None, str | None]:
    """Normalize one per-trial evaluator result or raise on contract drift."""
    if isinstance(eval_result, EvalScore):
        return eval_result.score, eval_result.dimension_scores, eval_result.reasoning
    if isinstance(eval_result, (int, float)):
        return float(eval_result), None, None
    raise TypeError(
        "Trial evaluator must return float, int, or EvalScore, got "
        f"{type(eval_result).__name__}."
    )


def _coerce_corpus_evaluator_result(
    eval_result: float | EvalScore,
) -> tuple[float, dict[str, float] | None]:
    """Normalize one corpus evaluator result or raise on contract drift."""
    if isinstance(eval_result, EvalScore):
        return eval_result.score, eval_result.dimension_scores
    if isinstance(eval_result, (int, float)):
        return float(eval_result), None
    raise TypeError(
        "Corpus evaluator must return float, int, or EvalScore, got "
        f"{type(eval_result).__name__}."
    )


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
                dimension_scores = t.dimension_scores
                if dimension_scores is None:
                    continue
                for dim_name, dim_score in dimension_scores.items():
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
            mean_latency_ms=statistics.mean(t.latency_ms for t in variant_trials) if variant_trials else 0.0,
            total_tokens=sum(t.tokens_used for t in variant_trials),
            corpus_score=corpus_score,
            corpus_dimension_scores=corpus_dimension_scores,
        )

    return summaries
