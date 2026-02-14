"""Core data models for prompt experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptVariant(BaseModel):
    """A single prompt variant to test."""

    name: str = Field(description="Human-readable variant name (e.g. 'concise_v2')")
    messages: List[Dict[str, str]] = Field(
        description="Chat messages in OpenAI format [{'role': ..., 'content': ...}]"
    )
    model: str = Field(default="gpt-5-mini", description="Model to use for this variant")
    temperature: float = Field(default=1.0)
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs passed to llm_client (max_tokens, etc.)",
    )


class Trial(BaseModel):
    """Result of a single LLM call for one variant on one input."""

    variant_name: str
    input_id: str
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


class EvalResult(BaseModel):
    """Aggregated results for an experiment."""

    experiment_name: str
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
