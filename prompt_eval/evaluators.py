"""Evaluator factories for scoring LLM outputs."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from llm_client import acall_llm

logger = logging.getLogger(__name__)


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


_JUDGE_PROMPT = """Score the following output on a scale of 0.0 to 1.0.

## Rubric
{rubric}

## Output to evaluate
{output}

{expected_section}

## Instructions
Respond with ONLY a single decimal number between 0.0 and 1.0.
Do not include any other text, explanation, or formatting."""


def llm_judge_evaluator(
    rubric: str,
    judge_model: str = "gpt-5-mini",
    output_formatter: Callable[[Any], str] | None = None,
) -> Callable:
    """Create an async evaluator that uses an LLM to score output against a rubric.

    The judge LLM receives the rubric and the output, and returns a score
    between 0.0 and 1.0. If `expected` is provided, it's included for context.

    Args:
        rubric: Scoring criteria. Can be anything: research questions,
            quality standards, a grading rubric, methodology requirements.
        judge_model: Model to use as judge (default: gpt-5-mini).
        output_formatter: Optional function to format the output before
            sending to the judge. If None, uses str().

    Returns:
        Async evaluator function(output, expected) -> float.
    """
    async def evaluate(output: Any, expected: Any = None) -> float:
        formatted_output = output_formatter(output) if output_formatter else str(output)

        expected_section = ""
        if expected is not None:
            expected_str = output_formatter(expected) if output_formatter else str(expected)
            expected_section = f"## Reference (expected)\n{expected_str}"

        prompt = _JUDGE_PROMPT.format(
            rubric=rubric,
            output=formatted_output,
            expected_section=expected_section,
        )

        response, _ = await acall_llm(
            judge_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        # Parse the score from the response
        text = str(response).strip()
        try:
            score = float(text)
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to extract a number from the response
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            logger.warning("Judge returned unparseable score: %r", text)
            return 0.0

    return evaluate
