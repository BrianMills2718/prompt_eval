"""Tests for prompt_eval.evaluators."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult

from prompt_eval.evaluators import (
    EvalScore,
    JudgeVerdict,
    DimensionScore,
    RubricDimension,
    _build_dimensions_text,
    _build_presence_vectors,
    _cohens_kappa,
    _normalize,
    contains_evaluator,
    exact_match_evaluator,
    kappa_evaluator,
    llm_judge_dimensional_evaluator,
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
            mock.return_value = LLMCallResult(content="85", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="Is it good?")
            score = await ev("some output")
            assert score == 0.85

    async def test_clamps_to_range(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="150", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 1.0

        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="-30", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.0

    async def test_extracts_number_from_text(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="Score: 72", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.72

    async def test_unparseable_returns_zero(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="I can't score this", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.0

    async def test_includes_expected_in_prompt(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="90", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="test")
            await ev("output", expected="reference answer")
            call_args = mock.call_args
            prompt = call_args[0][1][0]["content"]
            assert "reference answer" in prompt

    async def test_custom_output_formatter(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="70", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(
                rubric="test",
                output_formatter=lambda x: f"FORMATTED: {x}",
            )
            await ev("raw output")
            prompt = mock.call_args[0][1][0]["content"]
            assert "FORMATTED: raw output" in prompt

    async def test_rubric_in_prompt(self) -> None:
        with patch("prompt_eval.evaluators.acall_llm") as mock:
            mock.return_value = LLMCallResult(content="50", usage={"total_tokens": 100}, cost=0.001, model="test")
            ev = llm_judge_evaluator(rubric="Codes must be grounded in participant quotes")
            await ev("some codes")
            prompt = mock.call_args[0][1][0]["content"]
            assert "grounded in participant quotes" in prompt


class TestBuildDimensionsText:
    def test_basic(self) -> None:
        dims = [RubricDimension(name="clarity", description="Is it clear?")]
        text = _build_dimensions_text(dims)
        assert "### clarity" in text
        assert "Is it clear?" in text

    def test_with_anchors(self) -> None:
        dims = [RubricDimension(
            name="depth",
            description="Analytical depth",
            anchors={"low": "Surface only", "high": "Deep insight"},
        )]
        text = _build_dimensions_text(dims)
        assert "**low**" in text
        assert "Surface only" in text
        assert "**high**" in text


class TestDimensionalEvaluator:
    # mock-ok: testing evaluator orchestration, not actual LLM judge quality

    @pytest.fixture
    def dimensions(self) -> list[RubricDimension]:
        return [
            RubricDimension(name="clarity", description="Is it clear?", weight=1.0),
            RubricDimension(name="depth", description="Is it deep?", weight=1.0),
        ]

    async def test_returns_eval_score(self, dimensions: list[RubricDimension]) -> None:
        verdict = JudgeVerdict(
            reasoning="Good output overall.",
            scores=[
                DimensionScore(dimension="clarity", score=80),
                DimensionScore(dimension="depth", score=60),
            ],
            overall_score=70,
        )
        mock_meta = LLMCallResult(content="", usage={"total_tokens": 200}, cost=0.002, model="test")
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.return_value = (verdict, mock_meta)
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            result = await ev("some output")

        assert isinstance(result, EvalScore)
        assert result.score == pytest.approx(0.7)
        assert result.dimension_scores["clarity"] == 0.8
        assert result.dimension_scores["depth"] == 0.6
        assert "Good output overall." in result.reasoning

    async def test_multi_judge_averages(self, dimensions: list[RubricDimension]) -> None:
        verdict_a = JudgeVerdict(
            reasoning="Judge A thinks it's okay.",
            scores=[
                DimensionScore(dimension="clarity", score=80),
                DimensionScore(dimension="depth", score=60),
            ],
            overall_score=70,
        )
        verdict_b = JudgeVerdict(
            reasoning="Judge B thinks it's great.",
            scores=[
                DimensionScore(dimension="clarity", score=100),
                DimensionScore(dimension="depth", score=80),
            ],
            overall_score=90,
        )
        mock_meta = LLMCallResult(content="", usage={"total_tokens": 200}, cost=0.002, model="test")
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.side_effect = [(verdict_a, mock_meta), (verdict_b, mock_meta)]
            ev = llm_judge_dimensional_evaluator(
                dimensions=dimensions,
                judge_models=["model-a", "model-b"],
            )
            result = await ev("some output")

        assert result.dimension_scores["clarity"] == pytest.approx(0.9)  # (0.8+1.0)/2
        assert result.dimension_scores["depth"] == pytest.approx(0.7)    # (0.6+0.8)/2
        assert result.score == pytest.approx(0.8)                        # (0.9+0.7)/2
        assert "Judge A" in result.reasoning
        assert "Judge B" in result.reasoning

    async def test_weighted_dimensions(self) -> None:
        dims = [
            RubricDimension(name="major", description="Important", weight=3.0),
            RubricDimension(name="minor", description="Less important", weight=1.0),
        ]
        verdict = JudgeVerdict(
            reasoning="Test",
            scores=[
                DimensionScore(dimension="major", score=80),
                DimensionScore(dimension="minor", score=40),
            ],
            overall_score=70,
        )
        mock_meta = LLMCallResult(content="", usage={"total_tokens": 200}, cost=0.002, model="test")
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.return_value = (verdict, mock_meta)
            ev = llm_judge_dimensional_evaluator(dimensions=dims)
            result = await ev("output")

        # weighted: (0.8*3 + 0.4*1) / 4 = 2.8/4 = 0.7
        assert result.score == pytest.approx(0.7)

    async def test_judge_failure_handled(self, dimensions: list[RubricDimension]) -> None:
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("API error")
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            result = await ev("output")

        assert isinstance(result, EvalScore)
        assert result.score == 0.0
        assert result.dimension_scores["clarity"] == 0.0

    async def test_includes_expected_in_prompt(self, dimensions: list[RubricDimension]) -> None:
        verdict = JudgeVerdict(
            reasoning="Ok",
            scores=[
                DimensionScore(dimension="clarity", score=50),
                DimensionScore(dimension="depth", score=50),
            ],
            overall_score=50,
        )
        mock_meta = LLMCallResult(content="", usage={"total_tokens": 200}, cost=0.002, model="test")
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.return_value = (verdict, mock_meta)
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            await ev("output", expected="reference answer")
            prompt = mock.call_args[0][1][0]["content"]
            assert "reference answer" in prompt

    async def test_output_formatter(self, dimensions: list[RubricDimension]) -> None:
        verdict = JudgeVerdict(
            reasoning="Ok",
            scores=[
                DimensionScore(dimension="clarity", score=50),
                DimensionScore(dimension="depth", score=50),
            ],
            overall_score=50,
        )
        mock_meta = LLMCallResult(content="", usage={"total_tokens": 200}, cost=0.002, model="test")
        with patch("prompt_eval.evaluators.acall_llm_structured", new_callable=AsyncMock) as mock:
            mock.return_value = (verdict, mock_meta)
            ev = llm_judge_dimensional_evaluator(
                dimensions=dimensions,
                output_formatter=lambda x: f"FORMATTED: {x}",
            )
            await ev("raw output")
            prompt = mock.call_args[0][1][0]["content"]
            assert "FORMATTED: raw output" in prompt
