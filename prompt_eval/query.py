"""Read-side helpers for reconstructing prompt_eval views from shared runs.

This module provides the smallest useful query adapter over `llm_client`
observability so prompt_eval can reload experiment families without depending
on its legacy JSON result files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llm_client.observability import (
    get_experiment_aggregates,
    get_run_items,
    get_runs,
)

from prompt_eval.experiment import EvalResult, Trial
from prompt_eval.runner import _build_summaries


def load_result_from_observability(
    execution_id: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
    limit: int = 1000,
) -> EvalResult:
    """Reconstruct one prompt_eval result family from shared llm_client runs.

    Args:
        execution_id: Shared run-family identifier emitted by
            `prompt_eval.run_experiment()`.
        project: Optional project filter passed through to `llm_client`.
        dataset: Optional dataset filter passed through to `llm_client`.
        limit: Maximum number of candidate runs to scan before filtering by
            provenance execution ID.

    Returns:
        An `EvalResult` reconstructed from the matching shared runs and items.

    Raises:
        ValueError: If the execution family is missing, incomplete, or
            internally inconsistent.

    Notes:
        Structured outputs are reconstructed into plain Python values when their
        serialized format is known, but the original Pydantic model class is not
        recreated on load.
    """
    family_runs = _load_family_runs(
        execution_id,
        project=project,
        dataset=dataset,
        limit=limit,
    )
    variant_names = _variant_names_in_order(family_runs)

    trials: list[Trial] = []
    for run in family_runs:
        items = get_run_items(run["run_id"])
        expected_input_count = _expected_input_count(run)
        if len(items) != expected_input_count:
            raise ValueError(
                "Observed run family is incomplete: "
                f"run {run['run_id']} expected {expected_input_count} items "
                f"from provenance but found {len(items)}."
            )
        for item in items:
            trials.append(_trial_from_item(run=run, item=item))

    experiment_name = _family_experiment_name(family_runs)
    summary = _build_summaries(trials, variant_names)
    _apply_corpus_aggregates(
        summary=summary,
        execution_id=execution_id,
        project=project,
        dataset=dataset,
        limit=limit,
    )
    return EvalResult(
        experiment_name=experiment_name,
        execution_id=execution_id,
        variants=variant_names,
        trials=trials,
        summary=summary,
    )


def _load_family_runs(
    execution_id: str,
    *,
    project: str | None,
    dataset: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Load and validate the shared runs belonging to one execution family."""
    candidate_runs = get_runs(project=project, dataset=dataset, limit=limit)
    family_runs = [
        run
        for run in candidate_runs
        if _run_execution_id(run) == execution_id
    ]
    if not family_runs:
        raise ValueError(
            "No shared prompt_eval run family found for "
            f"execution_id={execution_id!r}."
        )
    _validate_family_runs(
        family_runs,
        execution_id=execution_id,
        dataset=dataset,
        project=project,
        limit=limit,
    )
    return sorted(
        family_runs,
        key=lambda run: (
            str(run.get("timestamp") or ""),
            str(run.get("condition_id") or ""),
            int(run.get("replicate") or 0),
        ),
    )


def _validate_family_runs(
    family_runs: list[dict[str, Any]],
    *,
    execution_id: str,
    dataset: str | None,
    project: str | None,
    limit: int,
) -> None:
    """Fail loudly when a shared run family is incomplete or contradictory."""
    experiment_names = set()
    variant_counts = set()
    n_runs_values = set()
    condition_pairs: set[tuple[str, int]] = set()
    condition_names: set[str] = set()
    replicate_values: set[int] = set()

    for run in family_runs:
        provenance = _provenance(run)
        experiment_name = provenance.get("experiment_name")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError(
                "Shared prompt_eval run family is missing provenance.experiment_name "
                f"for execution_id={execution_id!r}."
            )
        experiment_names.add(experiment_name)

        variant_count = provenance.get("variant_count")
        if not isinstance(variant_count, int) or variant_count <= 0:
            raise ValueError(
                "Shared prompt_eval run family has invalid provenance.variant_count "
                f"for execution_id={execution_id!r}: {variant_count!r}."
            )
        variant_counts.add(variant_count)

        n_runs = provenance.get("n_runs")
        if not isinstance(n_runs, int) or n_runs <= 0:
            raise ValueError(
                "Shared prompt_eval run family has invalid provenance.n_runs "
                f"for execution_id={execution_id!r}: {n_runs!r}."
            )
        n_runs_values.add(n_runs)

        condition_id = run.get("condition_id")
        replicate = run.get("replicate")
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError(
                "Shared prompt_eval run family is missing condition_id for "
                f"execution_id={execution_id!r}."
            )
        if not isinstance(replicate, int):
            raise ValueError(
                "Shared prompt_eval run family is missing replicate for "
                f"execution_id={execution_id!r}."
            )

        pair = (condition_id, replicate)
        if pair in condition_pairs:
            raise ValueError(
                "Shared prompt_eval run family contains duplicate "
                f"(condition_id, replicate) pairs for execution_id={execution_id!r}: "
                f"{pair!r}."
            )
        condition_pairs.add(pair)
        condition_names.add(condition_id)
        replicate_values.add(replicate)

    if len(experiment_names) != 1:
        raise ValueError(
            "Shared prompt_eval run family has inconsistent experiment names for "
            f"execution_id={execution_id!r}: {sorted(experiment_names)!r}."
        )
    if len(variant_counts) != 1 or len(n_runs_values) != 1:
        raise ValueError(
            "Shared prompt_eval run family has inconsistent provenance counts for "
            f"execution_id={execution_id!r}."
        )

    expected_run_count = next(iter(variant_counts)) * next(iter(n_runs_values))
    if len(family_runs) != expected_run_count:
        filters = []
        if project is not None:
            filters.append(f"project={project!r}")
        if dataset is not None:
            filters.append(f"dataset={dataset!r}")
        filter_clause = ""
        if filters:
            filter_clause = f" with filters ({', '.join(filters)})"
        raise ValueError(
            "Shared prompt_eval run family is incomplete or truncated: "
            f"execution_id={execution_id!r} expected {expected_run_count} runs "
            f"from provenance but found {len(family_runs)} within limit={limit}"
            f"{filter_clause}."
        )

    if len(condition_names) != next(iter(variant_counts)):
        raise ValueError(
            "Shared prompt_eval run family has inconsistent condition coverage for "
            f"execution_id={execution_id!r}: expected {next(iter(variant_counts))} "
            f"unique conditions but found {len(condition_names)}."
        )
    if replicate_values != set(range(next(iter(n_runs_values)))):
        raise ValueError(
            "Shared prompt_eval run family has inconsistent replicate coverage for "
            f"execution_id={execution_id!r}: expected "
            f"{sorted(set(range(next(iter(n_runs_values)))))!r} but found "
            f"{sorted(replicate_values)!r}."
        )


def _variant_names_in_order(family_runs: list[dict[str, Any]]) -> list[str]:
    """Preserve first-seen variant order from chronological shared runs."""
    ordered: list[str] = []
    seen: set[str] = set()
    for run in family_runs:
        condition_id = str(run["condition_id"])
        if condition_id not in seen:
            ordered.append(condition_id)
            seen.add(condition_id)
    return ordered


def _family_experiment_name(family_runs: list[dict[str, Any]]) -> str:
    """Return the one validated experiment name for the family."""
    experiment_names = {
        str(_provenance(run)["experiment_name"])
        for run in family_runs
    }
    if len(experiment_names) != 1:
        raise ValueError(
            "Shared prompt_eval run family has inconsistent experiment names."
        )
    return next(iter(experiment_names))


def _run_execution_id(run: Mapping[str, Any]) -> str | None:
    """Extract the prompt_eval family execution ID from run provenance."""
    execution_id = _provenance(run).get("experiment_execution_id")
    if isinstance(execution_id, str) and execution_id.strip():
        return execution_id
    return None


def _expected_input_count(run: Mapping[str, Any]) -> int:
    """Return the expected number of items in one reconstructed shared run."""
    input_count = _provenance(run).get("input_count")
    if not isinstance(input_count, int) or input_count < 0:
        raise ValueError(
            "Shared prompt_eval run family has invalid provenance.input_count for "
            f"run_id={run.get('run_id')!r}: {input_count!r}."
        )
    return input_count


def _trial_from_item(
    *,
    run: Mapping[str, Any],
    item: Mapping[str, Any],
) -> Trial:
    """Convert one shared run item row back into a prompt_eval Trial."""
    metrics = item.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            "Shared prompt_eval item is missing metrics for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )
    score = _numeric_metric(metrics.get("score"))
    dimension_scores = {
        str(name): value
        for name, value in (
            (str(metric_name), _numeric_metric(metric_value))
            for metric_name, metric_value in metrics.items()
            if metric_name != "score"
        )
        if value is not None
    } or None

    extra = item.get("extra")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError(
            "Shared prompt_eval item has invalid extra metadata for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )
    extra_map: Mapping[str, Any] = extra if isinstance(extra, Mapping) else {}
    predicted_format = extra_map.get("predicted_format")
    if predicted_format is not None and not isinstance(predicted_format, str):
        raise ValueError(
            "Shared prompt_eval item has invalid predicted_format for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )

    latency_s = item.get("latency_s")
    if latency_s is None:
        latency_ms = 0.0
    elif isinstance(latency_s, (int, float)):
        latency_ms = float(latency_s) * 1000.0
    else:
        raise ValueError(
            "Shared prompt_eval item has invalid latency_s for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )

    cost = item.get("cost")
    if cost is None:
        numeric_cost = 0.0
    elif isinstance(cost, (int, float)):
        numeric_cost = float(cost)
    else:
        raise ValueError(
            "Shared prompt_eval item has invalid cost for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )

    tokens_used = extra_map.get("tokens_used", 0)
    if not isinstance(tokens_used, int):
        raise ValueError(
            "Shared prompt_eval item has invalid tokens_used for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )
    reasoning = extra_map.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError(
            "Shared prompt_eval item has invalid reasoning for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )

    trace_id = item.get("trace_id")
    if trace_id is not None and not isinstance(trace_id, str):
        raise ValueError(
            "Shared prompt_eval item has invalid trace_id for "
            f"run_id={run.get('run_id')!r}, item_id={item.get('item_id')!r}."
        )

    item_id = item.get("item_id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError(
            "Shared prompt_eval item is missing item_id for "
            f"run_id={run.get('run_id')!r}."
        )

    condition_id = run.get("condition_id")
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError(
            "Shared prompt_eval run is missing condition_id for "
            f"run_id={run.get('run_id')!r}."
        )
    replicate = run.get("replicate")
    if not isinstance(replicate, int):
        raise ValueError(
            "Shared prompt_eval run is missing replicate for "
            f"run_id={run.get('run_id')!r}."
        )

    error = item.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError(
            "Shared prompt_eval item has invalid error payload for "
            f"run_id={run.get('run_id')!r}, item_id={item_id!r}."
        )

    predicted = item.get("predicted")
    if predicted is not None and not isinstance(predicted, str):
        raise ValueError(
            "Shared prompt_eval item has invalid predicted payload for "
            f"run_id={run.get('run_id')!r}, item_id={item_id!r}."
        )

    return Trial(
        variant_name=condition_id,
        input_id=item_id,
        replicate=replicate,
        output=_deserialize_payload(predicted, predicted_format),
        score=score,
        dimension_scores=dimension_scores,
        reasoning=reasoning,
        cost=numeric_cost,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        error=error,
        trace_id=trace_id,
    )


def _apply_corpus_aggregates(
    *,
    summary: dict[str, Any],
    execution_id: str,
    project: str | None,
    dataset: str | None,
    limit: int,
) -> None:
    """Hydrate corpus-level prompt_eval aggregates from the shared backend."""
    aggregate_rows = get_experiment_aggregates(
        family_id=execution_id,
        aggregate_type="prompt_eval.corpus_evaluator",
        project=project,
        dataset=dataset,
        limit=limit,
    )
    seen_conditions: set[str] = set()
    for row in aggregate_rows:
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError(
                "Shared prompt_eval corpus aggregate is missing condition_id for "
                f"aggregate_id={row.get('aggregate_id')!r}."
            )
        if condition_id not in summary:
            raise ValueError(
                "Shared prompt_eval corpus aggregate references an unknown condition "
                f"{condition_id!r} for execution_id={execution_id!r}."
            )
        if condition_id in seen_conditions:
            raise ValueError(
                "Shared prompt_eval corpus aggregate duplicated condition "
                f"{condition_id!r} for execution_id={execution_id!r}."
            )
        seen_conditions.add(condition_id)

        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(
                "Shared prompt_eval corpus aggregate has invalid metrics for "
                f"aggregate_id={row.get('aggregate_id')!r}."
            )
        summary_row = summary[condition_id]
        summary_row.corpus_score = _numeric_metric(metrics.get("score"))
        dimension_scores = {
            str(name): value
            for name, value in (
                (str(metric_name), _numeric_metric(metric_value))
                for metric_name, metric_value in metrics.items()
                if metric_name != "score"
            )
            if value is not None
        }
        summary_row.corpus_dimension_scores = dimension_scores or None


def _numeric_metric(value: Any) -> float | None:
    """Convert one stored metric value back into a numeric score."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected numeric metric, got boolean {value!r}.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric metric, got {type(value).__name__}: {value!r}.")


def _deserialize_payload(serialized: str | None, payload_format: str | None) -> Any:
    """Reconstruct one stored payload or fail loudly on format drift."""
    if serialized is None:
        return None
    if payload_format is None or payload_format == "text":
        return serialized
    if payload_format == "json" or payload_format.startswith("pydantic:"):
        try:
            return json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Shared prompt_eval item payload claimed JSON serialization but "
                f"could not be decoded for format {payload_format!r}."
            ) from exc
    return serialized


def _provenance(run: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return run provenance as a mapping or fail loudly on invalid payloads."""
    provenance = run.get("provenance")
    if provenance is None:
        return {}
    if isinstance(provenance, Mapping):
        return provenance
    raise ValueError(
        "Shared prompt_eval run has non-mapping provenance for "
        f"run_id={run.get('run_id')!r}."
    )
