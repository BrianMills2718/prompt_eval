"""Adapters from prompt_eval experiments to llm_client observability records.

This module keeps prompt-level mapping logic out of the runner so the boundary
between prompt-eval semantics and shared run telemetry stays explicit.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import uuid
from dataclasses import dataclass, replace
from typing import Any, Mapping

from pydantic import BaseModel

from prompt_eval.experiment import Experiment, ExperimentInput, PromptVariant, Trial


@dataclass(frozen=True)
class PromptEvalObservabilityConfig:
    """Control how prompt_eval emits shared llm_client observability records.

    Attributes:
        enabled: Whether to emit shared run and item records.
        project: Optional project override for shared observability.
        dataset: Optional dataset override. When omitted, the experiment name is
            used as a compatibility fallback.
        scenario_id: Optional scenario label shared across the run family.
        phase: Optional lifecycle phase label. When omitted, the caller's
            default phase is used.
        seed: Optional seed for cohort matching across repeated runs.
        experiment_execution_id: Optional stable family identifier. When
            omitted, one is generated per run_experiment invocation.
        provenance: Optional extra provenance fields merged into each emitted
            run record.
    """

    enabled: bool = True
    project: str | None = None
    dataset: str | None = None
    scenario_id: str | None = None
    phase: str | None = None
    seed: int | None = None
    experiment_execution_id: str | None = None
    provenance: Mapping[str, Any] | None = None


def _resolve_observability_config(
    observability: bool | PromptEvalObservabilityConfig | None,
    *,
    default_phase: str,
) -> PromptEvalObservabilityConfig | None:
    """Normalize public observability options into a concrete config."""
    if observability is False:
        return None
    if observability is None or observability is True:
        return PromptEvalObservabilityConfig(phase=default_phase)
    if not observability.enabled:
        return None
    if observability.phase is not None:
        return observability
    return replace(observability, phase=default_phase)


def _with_default_phase(
    observability: bool | PromptEvalObservabilityConfig | None,
    *,
    phase: str,
) -> bool | PromptEvalObservabilityConfig | None:
    """Apply a default phase without overriding an explicit caller-provided one."""
    if isinstance(observability, PromptEvalObservabilityConfig):
        if observability.phase is not None:
            return observability
        return replace(observability, phase=phase)
    if observability in {None, True}:
        return PromptEvalObservabilityConfig(phase=phase)
    return observability


def _experiment_execution_id(config: PromptEvalObservabilityConfig | None) -> str:
    """Return the shared execution-family identifier for one experiment invocation."""
    if config is not None and config.experiment_execution_id is not None:
        return config.experiment_execution_id
    return uuid.uuid4().hex[:12]


def _message_template_sha256(messages: list[dict[str, str]]) -> str:
    """Hash inline prompt templates so provenance stays explicit."""
    encoded = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialization_payload(value: Any) -> tuple[str | None, str | None]:
    """Serialize values into the text fields used by llm_client observability."""
    if value is None:
        return None, None
    if isinstance(value, str):
        return value, "text"
    if isinstance(value, BaseModel):
        return value.model_dump_json(), f"pydantic:{type(value).__name__}"
    try:
        return json.dumps(value, sort_keys=True, default=str), "json"
    except TypeError:
        return str(value), "text"


def _text_sha256(value: str | None) -> str | None:
    """Hash serialized text payloads when stable identity is useful."""
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evaluator_name(evaluator: Any) -> str | None:
    """Return a readable evaluator label when possible."""
    if evaluator is None:
        return None
    name = getattr(evaluator, "__name__", None)
    if isinstance(name, str) and name.strip():
        return name
    return type(evaluator).__name__


def _run_config_payload(
    *,
    variant: PromptVariant,
    experiment: Experiment,
) -> dict[str, Any]:
    """Build machine-queryable run config metadata."""
    response_model_name: str | None = None
    if experiment.response_model is not None:
        response_model_name = getattr(experiment.response_model, "__name__", None)
        if response_model_name is None:
            response_model_name = str(experiment.response_model)
    return {
        "temperature": variant.temperature,
        "variant_kwargs": dict(variant.kwargs),
        "structured_output": experiment.response_model is not None,
        "response_model_name": response_model_name,
    }


def _run_provenance_payload(
    *,
    experiment: Experiment,
    variant: PromptVariant,
    execution_id: str,
    evaluator: Any,
    corpus_evaluator: Any,
    observability: PromptEvalObservabilityConfig,
) -> dict[str, Any]:
    """Build run-family provenance metadata for prompt_eval-originated runs."""
    provenance: dict[str, Any] = {
        "source_package": "prompt_eval",
        "experiment_name": experiment.name,
        "experiment_execution_id": execution_id,
        "variant_name": variant.name,
        "variant_count": len(experiment.variants),
        "input_count": len(experiment.inputs),
        "n_runs": experiment.n_runs,
        "evaluator_name": _evaluator_name(evaluator),
        "corpus_evaluator_name": _evaluator_name(corpus_evaluator),
        "prompt_template_sha256": _message_template_sha256(variant.messages),
        "prompt_source": "prompt_ref" if variant.prompt_ref else "inline_messages",
    }
    if variant.prompt_ref is not None:
        provenance["prompt_ref"] = variant.prompt_ref
    if observability.provenance is not None:
        provenance.update(dict(observability.provenance))
    return provenance


def _trace_id(
    *,
    execution_id: str,
    condition_id: str,
    replicate: int,
    item_id: str,
) -> str:
    """Build a hierarchical trace identifier shared by the call and item row."""
    return f"prompt_eval/{execution_id}/{condition_id}/r{replicate}/{item_id}"


def _metrics_schema_for_run(run_trials: list[Trial]) -> list[str] | None:
    """Derive item-level metric keys present in one shared run."""
    metric_names: set[str] = set()
    for trial in run_trials:
        if trial.score is not None:
            metric_names.add("score")
        if trial.dimension_scores:
            metric_names.update(str(name) for name in trial.dimension_scores)
    if not metric_names:
        return None
    if "score" in metric_names:
        return ["score", *sorted(name for name in metric_names if name != "score")]
    return sorted(metric_names)


def _summary_metrics_for_run(run_trials: list[Trial]) -> dict[str, Any] | None:
    """Compute run-level aggregate metrics in llm_client-compatible units."""
    summary: dict[str, Any] = {}

    scores = [trial.score for trial in run_trials if trial.score is not None]
    if scores:
        summary["avg_score"] = round(100.0 * statistics.mean(scores), 2)

    dimension_values: dict[str, list[float]] = {}
    for trial in run_trials:
        if trial.dimension_scores is None:
            continue
        for name, value in trial.dimension_scores.items():
            dimension_values.setdefault(name, []).append(float(value))
    for name, values in dimension_values.items():
        summary[f"avg_{name}"] = round(100.0 * statistics.mean(values), 2)

    successful = [trial for trial in run_trials if trial.error is None]
    if successful:
        summary["mean_cost"] = round(statistics.mean(trial.cost for trial in successful), 6)
    if run_trials:
        summary["mean_latency_ms"] = round(
            statistics.mean(trial.latency_ms for trial in run_trials),
            3,
        )
        summary["total_tokens"] = sum(trial.tokens_used for trial in run_trials)

    return summary or None


def _corpus_aggregate_metrics(
    *,
    score: float | None,
    dimension_scores: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert corpus-evaluator outputs into shared aggregate metrics."""
    metrics: dict[str, Any] = {}
    if score is not None:
        metrics["score"] = float(score)
    if dimension_scores:
        for name, value in dimension_scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "Corpus aggregate dimension scores must be numeric. "
                    f"Got {name!r}={value!r}."
                )
            metrics[str(name)] = float(value)
    return metrics or None


def _corpus_aggregate_provenance(
    *,
    experiment: Experiment,
    variant: PromptVariant,
    execution_id: str,
    corpus_evaluator: Any,
) -> dict[str, Any]:
    """Build provenance for one prompt-eval corpus aggregate record."""
    provenance: dict[str, Any] = {
        "source_package": "prompt_eval",
        "aggregate_semantics": "corpus_evaluator",
        "experiment_name": experiment.name,
        "experiment_execution_id": execution_id,
        "variant_name": variant.name,
        "corpus_evaluator_name": _evaluator_name(corpus_evaluator),
    }
    if variant.prompt_ref is not None:
        provenance["prompt_ref"] = variant.prompt_ref
    else:
        provenance["prompt_template_sha256"] = _message_template_sha256(variant.messages)
    return provenance


def _item_metrics(trial: Trial) -> dict[str, Any]:
    """Convert a Trial into the item-level metric payload stored in llm_client."""
    metrics: dict[str, Any] = {}
    if trial.score is not None:
        metrics["score"] = trial.score
    if trial.dimension_scores:
        metrics.update(trial.dimension_scores)
    return metrics


def _item_extra(
    *,
    trial: Trial,
    inp: ExperimentInput,
) -> dict[str, Any] | None:
    """Build item-scoped metadata without duplicating large raw payloads."""
    _predicted_text, predicted_format = _serialization_payload(trial.output)
    expected_text, expected_format = _serialization_payload(inp.expected)

    extra: dict[str, Any] = {
        "tokens_used": trial.tokens_used,
        "predicted_format": predicted_format,
        "input_content_sha256": _text_sha256(inp.content),
    }
    if trial.reasoning:
        extra["reasoning"] = trial.reasoning
    if expected_format is not None:
        extra["expected_format"] = expected_format
    expected_hash = _text_sha256(expected_text)
    if expected_hash is not None:
        extra["expected_sha256"] = expected_hash
    return extra or None


def _log_item_payload(
    *,
    trial: Trial,
    inp: ExperimentInput,
) -> dict[str, Any]:
    """Build the keyword payload passed into llm_client.log_item()."""
    predicted_text, _predicted_format = _serialization_payload(trial.output)
    expected_text, _expected_format = _serialization_payload(inp.expected)
    return {
        "item_id": inp.id,
        "metrics": _item_metrics(trial),
        "predicted": predicted_text,
        "gold": expected_text,
        "latency_s": trial.latency_ms / 1000.0,
        "cost": trial.cost,
        "error": trial.error,
        "extra": _item_extra(trial=trial, inp=inp),
        "trace_id": trial.trace_id,
    }
