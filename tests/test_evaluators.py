"""Tests for prompt_eval.evaluators."""

from unittest.mock import AsyncMock, patch

import pytest

from prompt_eval.evaluators import (
    _build_presence_vectors,
    _cohens_kappa,
    _normalize,
    contains_evaluator,
    exact_match_evaluator,
    kappa_evaluator,
    llm_judge_evaluator,
)


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_strip_punctuation(self) -> None:
        assert _normalize("self-care (emotional)") == "selfcare emotional"

    def test_collapse_whitespace(self) -> None:
        assert _normalize("  too   many   spaces  ") == "too many spaces"

    def test_empty(self) -> None:
        assert _normalize("") == ""


class TestPresenceVectors:
    def test_identical(self) -> None:
        vec_a, vec_b = _build_presence_vectors(["a", "b"], ["a", "b"])
        assert vec_a == vec_b == [1, 1]

    def test_disjoint(self) -> None:
        vec_a, vec_b = _build_presence_vectors(["a"], ["b"])
        assert vec_a == [1, 0]
        assert vec_b == [0, 1]

    def test_overlap(self) -> None:
        vec_a, vec_b = _build_presence_vectors(["a", "b"], ["b", "c"])
        # universe: a, b, c
        assert vec_a == [1, 1, 0]
        assert vec_b == [0, 1, 1]

    def test_normalization_applied(self) -> None:
        vec_a, vec_b = _build_presence_vectors(["Hello"], ["hello"])
        assert vec_a == [1]
        assert vec_b == [1]

    def test_empty(self) -> None:
        vec_a, vec_b = _build_presence_vectors([], [])
        assert vec_a == []
        assert vec_b == []


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        kappa = _cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
        assert kappa == 1.0

    def test_no_agreement_beyond_chance(self) -> None:
        # Two raters with opposite coding on balanced data
        kappa = _cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1])
        assert kappa < 0  # negative kappa = worse than chance

    def test_empty_vectors(self) -> None:
        assert _cohens_kappa([], []) == 0.0


class TestKappaEvaluator:
    def test_perfect_match(self) -> None:
        evaluator = kappa_evaluator(lambda x: x)
        score = evaluator(["code_a", "code_b"], ["code_a", "code_b"])
        assert score == 1.0

    def test_no_overlap(self) -> None:
        evaluator = kappa_evaluator(lambda x: x)
        score = evaluator(["a"], ["b"])
        assert score < 0  # worse than chance

    def test_none_expected(self) -> None:
        evaluator = kappa_evaluator(lambda x: x)
        assert evaluator(["a"], None) == 0.0

    def test_with_pydantic_extractor(self) -> None:
        """Simulate extracting codes from a Pydantic model."""
        from pydantic import BaseModel

        class CodingResult(BaseModel):
            codes: list[str]

        extractor = lambda r: r.codes if isinstance(r, CodingResult) else r
        evaluator = kappa_evaluator(extractor)

        # Identical codes → kappa = 1.0
        output = CodingResult(codes=["trust", "communication"])
        expected = CodingResult(codes=["trust", "communication"])
        assert evaluator(output, expected) == 1.0

        # Subset relationship → kappa = 0.0 (observed == chance agreement
        # because the universe has no true negatives when one is a subset)
        expected_superset = CodingResult(codes=["trust", "communication", "conflict"])
        assert evaluator(output, expected_superset) == 0.0


class TestExactMatchEvaluator:
    def test_match(self) -> None:
        ev = exact_match_evaluator()
        assert ev("hello", "hello") == 1.0

    def test_no_match(self) -> None:
        ev = exact_match_evaluator()
        assert ev("hello", "world") == 0.0

    def test_none_expected(self) -> None:
        ev = exact_match_evaluator()
        assert ev("hello", None) == 0.0

    def test_coerces_to_str(self) -> None:
        ev = exact_match_evaluator()
        assert ev(42, "42") == 1.0


class TestContainsEvaluator:
    def test_contains(self) -> None:
        ev = contains_evaluator()
        assert ev("the answer is 42", "42") == 1.0

    def test_not_contains(self) -> None:
        ev = contains_evaluator()
        assert ev("the answer is 41", "42") == 0.0

    def test_none_expected(self) -> None:
        ev = contains_evaluator()
        assert ev("hello", None) == 0.0

    def test_key_extractor(self) -> None:
        ev = contains_evaluator(key_extractor=lambda x: x.get("text", ""))
        assert ev({"text": "the answer is 42"}, {"text": "42"}) == 1.0


class TestLlmJudgeEvaluator:
    # mock-ok: testing evaluator orchestration, not actual LLM judge quality

    async def test_clean_score(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("0.85", AsyncMock())
            ev = llm_judge_evaluator(rubric="Is it good?")
            score = await ev("some output")
            assert score == 0.85

    async def test_clamps_to_range(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("1.5", AsyncMock())
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 1.0

        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("-0.3", AsyncMock())
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.0

    async def test_extracts_number_from_text(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("Score: 0.72", AsyncMock())
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.72

    async def test_unparseable_returns_zero(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("I can't score this", AsyncMock())
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.0

    async def test_includes_expected_in_prompt(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("0.9", AsyncMock())
            ev = llm_judge_evaluator(rubric="test")
            await ev("output", expected="reference answer")
            call_args = mock.call_args
            prompt = call_args[0][1][0]["content"]
            assert "reference answer" in prompt

    async def test_custom_output_formatter(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("0.7", AsyncMock())
            ev = llm_judge_evaluator(
                rubric="test",
                output_formatter=lambda x: f"FORMATTED: {x}",
            )
            await ev("raw output")
            prompt = mock.call_args[0][1][0]["content"]
            assert "FORMATTED: raw output" in prompt

    async def test_rubric_in_prompt(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = ("0.5", AsyncMock())
            ev = llm_judge_evaluator(rubric="Codes must be grounded in participant quotes")
            await ev("some codes")
            prompt = mock.call_args[0][1][0]["content"]
            assert "grounded in participant quotes" in prompt
