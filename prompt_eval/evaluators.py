"""Evaluator factories for scoring LLM outputs."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from llm_client import acall_llm, acall_llm_structured, get_model
from prompt_eval.prompt_templates import (
    render_dimensional_judge_messages,
    render_scalar_judge_messages,
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
    """A single dimension score from the judge (0-100 integer scale)."""

    dimension: str
    score: int = Field(ge=0, le=100)


class JudgeVerdict(BaseModel):
    """Structured judge output with chain-of-thought reasoning."""

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


def llm_judge_evaluator(
    rubric: str,
    judge_model: str | None = None,
    output_formatter: Callable[[Any], str] | None = None,
    timeout: int = 120,
) -> Callable:
    """Create an async evaluator that uses an LLM to score output against a rubric.

    The judge LLM receives the rubric and the output, and returns a score
    between 0.0 and 1.0. If `expected` is provided, it's included for context.

    Args:
        rubric: Scoring criteria. Can be anything: research questions,
            quality standards, a grading rubric, methodology requirements.
        judge_model: Explicit model override for the judge. When omitted,
            resolves through ``llm_client.get_model("judging")``.
        output_formatter: Optional function to format the output before
            sending to the judge. If None, uses str().
        timeout: Timeout in seconds for the judge LLM call (default: 120).

    Returns:
        Async evaluator function(output, expected) -> float.
    """
    resolved_judge_model = judge_model or get_model(_JUDGE_SELECTION_TASK)

    async def evaluate(output: Any, expected: Any = None) -> float:
        formatted_output = output_formatter(output) if output_formatter else str(output)

        expected_section = ""
        if expected is not None:
            expected_str = output_formatter(expected) if output_formatter else str(expected)
            expected_section = f"## Reference (expected)\n{expected_str}"

        messages = render_scalar_judge_messages(
            rubric=rubric,
            output=formatted_output,
            expected_section=expected_section,
        )

        output_hash = hashlib.sha256(formatted_output.encode()).hexdigest()[:8]
        result = await acall_llm(
            resolved_judge_model,
            messages,
            timeout=timeout,
            task="prompt_eval.evaluate.judge",
            trace_id=f"prompt_eval.judge.{resolved_judge_model}.{output_hash}",
            max_budget=0,
        )

        # Parse integer 0-100 score, convert to 0.0-1.0
        text = result.content.strip()
        try:
            raw = float(text)
            if raw > 1.0:
                raw = raw / 100.0  # integer scale -> float
            return max(0.0, min(1.0, raw))
        except ValueError:
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                raw = float(match.group(1))
                if raw > 1.0:
                    raw = raw / 100.0
                return max(0.0, min(1.0, raw))
            logger.warning("Judge returned unparseable score: %r", text)
            raise RuntimeError(
                "Judge model "
                f"{resolved_judge_model!r} returned an unparseable score: {text!r}"
            )

    return evaluate


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


def llm_judge_dimensional_evaluator(
    dimensions: list[RubricDimension],
    judge_models: list[str] | None = None,
    output_formatter: Callable[[Any], str] | None = None,
    timeout: int = 120,
) -> Callable:
    """Create an async evaluator that scores output on multiple rubric dimensions.

    Uses structured output (JudgeVerdict) to get chain-of-thought reasoning
    and per-dimension scores from the judge LLM.

    Args:
        dimensions: Scoring dimensions with descriptions and optional anchors.
        judge_models: Explicit model overrides to use as judges (averaged if
            multiple). Defaults to ``[get_model("judging")]``.
        output_formatter: Optional function to format output before judging.
        timeout: Timeout in seconds for each judge LLM call (default: 120).

    Returns:
        Async evaluator function(output, expected) -> EvalScore.
    """
    if judge_models is None:
        judge_models = [get_model(_JUDGE_SELECTION_TASK)]

    dimensions_text = _build_dimensions_text(dimensions)
    weights = {dim.name: dim.weight for dim in dimensions}
    total_weight = sum(weights.values())

    async def evaluate(output: Any, expected: Any = None) -> EvalScore:
        formatted_output = output_formatter(output) if output_formatter else str(output)

        expected_section = ""
        if expected is not None:
            expected_str = output_formatter(expected) if output_formatter else str(expected)
            expected_section = f"## Reference (expected)\n{expected_str}"

        messages = render_dimensional_judge_messages(
            dimensions_text=dimensions_text,
            output=formatted_output,
            expected_section=expected_section,
        )

        all_dim_scores: dict[str, list[float]] = {dim.name: [] for dim in dimensions}
        all_reasoning: list[str] = []

        for model in judge_models:
            try:
                output_hash = hashlib.sha256(formatted_output.encode()).hexdigest()[:8]
                verdict, _meta = await acall_llm_structured(
                    model,
                    messages,
                    response_model=JudgeVerdict,
                    timeout=timeout,
                    task="prompt_eval.evaluate.dimensional_judge",
                    trace_id=f"prompt_eval.dimensional_judge.{model}.{output_hash}",
                    max_budget=0,
                )
                all_reasoning.append(f"[{model}] {verdict.reasoning}")
                for ds in verdict.scores:
                    if ds.dimension in all_dim_scores:
                        all_dim_scores[ds.dimension].append(ds.score / 100.0)
            except Exception as e:
                logger.warning("Judge model %s failed: %s", model, e)

        # Check that at least one judge produced valid scores
        has_scores = any(scores for scores in all_dim_scores.values())
        if not has_scores:
            raise RuntimeError(
                f"All {len(judge_models)} judge model(s) failed to produce scores"
            )

        # Average across judges
        avg_dims: dict[str, float] = {}
        for name, scores in all_dim_scores.items():
            avg_dims[name] = sum(scores) / len(scores) if scores else 0.0

        # Weighted average for overall score
        weighted_sum = sum(avg_dims.get(name, 0.0) * w for name, w in weights.items())
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return EvalScore(
            score=max(0.0, min(1.0, overall)),
            dimension_scores=avg_dims,
            reasoning="\n\n".join(all_reasoning),
        )

    return evaluate
