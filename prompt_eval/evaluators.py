"""Evaluator factories for scoring LLM outputs.

All LLM judge evaluators delegate to ``scoring.ascore_output`` so that:
- Judge prompts come from YAML templates (Prompts-as-Data)
- Judge model selection uses ``get_model("judging")`` from the registry
- Every evaluation is logged to the observability DB
- Rubric definitions are reusable across evaluators and experiments

Non-LLM evaluators (kappa, exact_match, contains) remain self-contained.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from llm_client import get_model
from prompt_eval.scoring import (
    Rubric,
    ScoreResult,
    ascore_output,
    ascore_output_multi_judge,
)

logger = logging.getLogger(__name__)


_JUDGE_SELECTION_TASK = "judging"


@dataclass
class EvalScore:
    """Rich evaluation result with per-dimension scores and reasoning."""

    score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class RubricDimension:
    """One dimension of a scoring rubric."""

    name: str
    description: str
    weight: float = 1.0
    anchors: dict[str, str] = field(default_factory=dict)


class DimensionScore(BaseModel):
    """A single dimension score from the judge (0-100 integer scale).

    Kept for backward compatibility with code that references this type.
    The actual scoring now goes through scoring.ascore_output.
    """

    dimension: str
    score: int = Field(ge=0, le=100)


class JudgeVerdict(BaseModel):
    """Structured judge output with chain-of-thought reasoning.

    Kept for backward compatibility with code that references this type.
    The actual scoring now goes through scoring.ascore_output.
    """

    reasoning: str
    scores: list[DimensionScore]
    overall_score: int = Field(ge=0, le=100)


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _build_presence_vectors(
    list_a: list[str], list_b: list[str]
) -> tuple[list[int], list[int]]:
    """Build binary presence vectors over the union of normalized names."""
    norm_a = {_normalize(n) for n in list_a}
    norm_b = {_normalize(n) for n in list_b}
    universe = sorted(norm_a | norm_b)
    vec_a = [1 if name in norm_a else 0 for name in universe]
    vec_b = [1 if name in norm_b else 0 for name in universe]
    return vec_a, vec_b


def _cohens_kappa(vec_a: list[int], vec_b: list[int]) -> float:
    """Compute Cohen's kappa from two binary presence vectors.

    Returns 0.0 for empty vectors or when chance agreement = 1.0.
    """
    n = len(vec_a)
    if n == 0:
        return 0.0

    agree = sum(a == b for a, b in zip(vec_a, vec_b))
    po = agree / n

    p_yes_a = sum(vec_a) / n
    p_yes_b = sum(vec_b) / n
    pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)

    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def kappa_evaluator(
    code_extractor: Callable[[Any], list[str]],
) -> Callable[[Any, Optional[Any]], float]:
    """Create an evaluator that computes Cohen's kappa between output and expected.

    Args:
        code_extractor: Function that pulls a list of code names from LLM output.
            Applied to both `output` and `expected`.

    Returns:
        Evaluator function(output, expected) -> float (kappa score).
    """

    def evaluate(output: Any, expected: Any = None) -> float:
        if expected is None:
            return 0.0
        codes_output = code_extractor(output)
        codes_expected = code_extractor(expected)
        vec_a, vec_b = _build_presence_vectors(codes_output, codes_expected)
        return _cohens_kappa(vec_a, vec_b)

    return evaluate


def exact_match_evaluator() -> Callable[[Any, Optional[Any]], float]:
    """Create an evaluator that returns 1.0 if str(output) == str(expected), else 0.0."""

    def evaluate(output: Any, expected: Any = None) -> float:
        return 1.0 if str(output) == str(expected) else 0.0

    return evaluate


def contains_evaluator(
    key_extractor: Callable[[Any], str] | None = None,
) -> Callable[[Any, Optional[Any]], float]:
    """Create an evaluator that returns 1.0 if expected is found in output.

    Args:
        key_extractor: Optional function to extract a string from output/expected
            before comparison. If None, uses str().
    """

    def evaluate(output: Any, expected: Any = None) -> float:
        if expected is None:
            return 0.0
        out_str = key_extractor(output) if key_extractor else str(output)
        exp_str = key_extractor(expected) if key_extractor else str(expected)
        return 1.0 if exp_str in out_str else 0.0

    return evaluate


def _score_result_to_float(result: ScoreResult) -> float:
    """Convert a ScoreResult to a 0.0-1.0 float score."""
    return max(0.0, min(1.0, result.overall_score))


def _score_result_to_eval_score(result: ScoreResult) -> EvalScore:
    """Convert a ScoreResult to an EvalScore with per-dimension scores.

    Normalizes raw dimension scores (on the rubric's 1-5 scale) to 0.0-1.0.
    """
    dim_scores: dict[str, float] = {}
    for name, raw in result.dimensions.items():
        # Default scale is 5 (1-5). Normalize: (raw - 1) / (scale - 1)
        dim_scores[name] = max(0.0, min(1.0, (raw - 1) / 4.0))

    # Combine reasoning from all dimensions
    reasoning_parts = []
    for name, reason in result.reasoning.items():
        reasoning_parts.append(f"[{name}] {reason}")

    return EvalScore(
        score=max(0.0, min(1.0, result.overall_score)),
        dimension_scores=dim_scores,
        reasoning="\n\n".join(reasoning_parts),
    )


def _build_dimensions_text(dimensions: list[RubricDimension]) -> str:
    """Build the dimensions section of the judge prompt."""
    parts = []
    for dim in dimensions:
        section = f"### {dim.name}\n{dim.description}\n"
        if dim.anchors:
            for level, desc in dim.anchors.items():
                section += f"- **{level}**: {desc}\n"
        parts.append(section)
    return "\n".join(parts)


def llm_judge_evaluator(
    rubric: str,
    judge_model: str | None = None,
    output_formatter: Callable[[Any], str] | None = None,
    timeout: int = 120,
) -> Callable:
    """Create an async evaluator that uses an LLM to score output against a rubric.

    Delegates to ``scoring.ascore_output`` for the actual judge call, ensuring
    observability logging and consistent judge prompt templates.

    Args:
        rubric: Scoring criteria. Can be anything: research questions,
            quality standards, a grading rubric, methodology requirements.
        judge_model: Explicit model override for the judge. When omitted,
            resolves through ``llm_client.get_model("judging")``.
        output_formatter: Optional function to format the output before
            sending to the judge. If None, uses str().
        timeout: Timeout in seconds for the judge LLM call (default: 120).
            Accepted for backward compatibility; the underlying
            ``ascore_output`` manages its own timeouts.

    Returns:
        Async evaluator function(output, expected) -> float.
    """
    resolved_judge_model = judge_model or get_model(_JUDGE_SELECTION_TASK)

    # Build an inline rubric from the rubric string
    rubric_obj = Rubric.from_inline(rubric)

    async def evaluate(output: Any, expected: Any = None) -> float:
        formatted_output = output_formatter(output) if output_formatter else str(output)

        # Build context with expected if provided
        context = ""
        if expected is not None:
            expected_str = output_formatter(expected) if output_formatter else str(expected)
            context = f"Reference (expected): {expected_str}"

        result = await ascore_output(
            output=formatted_output,
            rubric=rubric_obj,
            context=context,
            task="prompt_eval.evaluate.judge",
            judge_model=resolved_judge_model,
        )

        return _score_result_to_float(result)

    return evaluate


def llm_judge_dimensional_evaluator(
    dimensions: list[RubricDimension],
    judge_models: list[str] | None = None,
    output_formatter: Callable[[Any], str] | None = None,
    timeout: int = 120,
) -> Callable:
    """Create an async evaluator that scores output on multiple rubric dimensions.

    Delegates to ``scoring.ascore_output`` (single judge) or
    ``ascore_output_multi_judge`` (multiple judges) for the actual judge
    call, ensuring observability logging and consistent prompt templates.

    Args:
        dimensions: Scoring dimensions with descriptions and optional anchors.
        judge_models: Explicit model overrides to use as judges (averaged if
            multiple). Defaults to ``[get_model("judging")]``.
        output_formatter: Optional function to format output before judging.
        timeout: Timeout in seconds for each judge LLM call (default: 120).
            Accepted for backward compatibility; the underlying
            ``ascore_output`` manages its own timeouts.

    Returns:
        Async evaluator function(output, expected) -> EvalScore.
    """
    if judge_models is None:
        judge_models = [get_model(_JUDGE_SELECTION_TASK)]

    # Build a Rubric from the dimension list
    rubric_obj = Rubric.from_dimensions(
        [
            {
                "name": dim.name,
                "description": dim.description,
                "weight": dim.weight,
                "anchors": dim.anchors if dim.anchors else None,
            }
            for dim in dimensions
        ]
    )

    async def evaluate(output: Any, expected: Any = None) -> EvalScore:
        formatted_output = output_formatter(output) if output_formatter else str(output)

        # Build context with expected if provided
        context = ""
        if expected is not None:
            expected_str = output_formatter(expected) if output_formatter else str(expected)
            context = f"Reference (expected): {expected_str}"

        if len(judge_models) == 1:
            result = await ascore_output(
                output=formatted_output,
                rubric=rubric_obj,
                context=context,
                task="prompt_eval.evaluate.dimensional_judge",
                judge_model=judge_models[0],
            )
        else:
            result = await ascore_output_multi_judge(
                output=formatted_output,
                rubric=rubric_obj,
                judge_models=judge_models,
                context=context,
                task="prompt_eval.evaluate.dimensional_judge",
            )

        return _score_result_to_eval_score(result)

    return evaluate
