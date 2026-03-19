"""Core data models for prompt experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptVariant(BaseModel):
    """A single prompt variant to test."""

    name: str = Field(description="Human-readable variant name (e.g. 'concise_v2')")
    messages: List[Dict[str, str]] = Field(
        description="Chat messages in OpenAI format [{'role': ..., 'content': ...}]"
    )
    prompt_ref: Optional[str] = Field(
        default=None,
        description="Explicit prompt asset reference when this variant comes from a shared prompt library.",
    )
    model: str = Field(
        description=(
            "Explicit subject model for this variant. This is part of the "
            "experiment semantics and must be caller-declared."
        )
    )
    temperature: float = Field(default=1.0)
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra kwargs passed to llm_client (max_tokens, etc.). Reserved "
            "keys `task` and `max_budget` override the default "
            "`prompt_eval.run` call metadata."
        ),
    )


class Trial(BaseModel):
    """Result of a single LLM call for one variant on one input."""

    variant_name: str
    input_id: str
    replicate: int = Field(default=0, description="Zero-based repeat index for this trial.")
    output: Any = Field(description="Raw LLM output (string or parsed model)")
    score: Optional[float] = Field(default=None, description="Metric score if evaluator provided")
    dimension_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension scores from dimensional evaluator"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Judge reasoning from dimensional evaluator"
    )
    cost: float = Field(default=0.0)
    latency_ms: float = Field(default=0.0)
    tokens_used: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    trace_id: Optional[str] = Field(
        default=None,
        description="Shared observability trace identifier for this trial when recorded.",
    )


class EvalResult(BaseModel):
    """Aggregated results for an experiment."""

    experiment_name: str
    execution_id: Optional[str] = Field(
        default=None,
        description="Shared execution-family identifier for this experiment invocation.",
    )
    variants: List[str] = Field(default_factory=list)
    trials: List[Trial] = Field(default_factory=list)
    summary: Dict[str, VariantSummary] = Field(default_factory=dict)


class VariantSummary(BaseModel):
    """Aggregated stats for one variant."""

    variant_name: str
    n_trials: int = 0
    n_errors: int = 0
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    dimension_means: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension mean scores across trials"
    )
    mean_cost: float = 0.0
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    corpus_score: Optional[float] = Field(
        default=None, description="Score from corpus-level evaluator (runs on all outputs for this variant)"
    )
    corpus_dimension_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension scores from corpus-level evaluator"
    )


# Fix forward reference
EvalResult.model_rebuild()


class Experiment(BaseModel):
    """Definition of a prompt experiment."""

    name: str = Field(description="Experiment name")
    variants: List[PromptVariant] = Field(description="Prompt variants to compare")
    inputs: List[ExperimentInput] = Field(
        default_factory=list,
        description="Inputs to test each variant against",
    )
    n_runs: int = Field(default=3, description="Number of runs per variant per input")
    response_model: Optional[Any] = Field(
        default=None,
        description="Pydantic model for structured output (uses acall_llm_structured if set)",
    )


class ExperimentInput(BaseModel):
    """A single input case for the experiment."""

    id: str = Field(description="Unique identifier for this input")
    content: str = Field(description="The input content to substitute into the prompt")
    expected: Optional[Any] = Field(
        default=None,
        description="Expected output for evaluation (ground truth)",
    )


# Fix forward reference
Experiment.model_rebuild()
