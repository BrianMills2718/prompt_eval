"""Tests for prompt_eval.evaluators.

LLM judge evaluators now delegate to ``scoring.ascore_output``, so tests
mock at the scoring layer rather than at the raw LLM call layer. This
validates the evaluator-to-scoring wiring, return type conversion, and
backward-compatible interfaces.

Non-LLM evaluators (kappa, exact_match, contains) are tested directly.
"""

from unittest.mock import AsyncMock, patch

import pytest

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
from prompt_eval.scoring import ScoreResult


def _make_score_result(
    overall: float = 0.75,
    dimensions: dict[str, int] | None = None,
    reasoning: dict[str, str] | None = None,
    judge_model: str = "test-judge",
) -> ScoreResult:
    """Helper to build a ScoreResult for mocking ascore_output."""
    return ScoreResult(
        rubric="inline",
        overall_score=overall,
        dimensions=dimensions or {"quality": 4},
        reasoning=reasoning or {"quality": "Good work"},
        judge_model=judge_model,
        method="llm_judge",
        cost=0.001,
        latency_s=0.5,
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

        def extractor(result: CodingResult | list[str]) -> list[str]:
            return result.codes if isinstance(result, CodingResult) else result

        evaluator = kappa_evaluator(extractor)

        # Identical codes -> kappa = 1.0
        output = CodingResult(codes=["trust", "communication"])
        expected = CodingResult(codes=["trust", "communication"])
        assert evaluator(output, expected) == 1.0

        # Subset relationship -> kappa = 0.0 (observed == chance agreement
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
    # mock-ok: testing evaluator-to-scoring delegation, not actual LLM calls

    async def test_returns_float_score(self) -> None:
        """Evaluator returns a float between 0.0 and 1.0."""
        mock_result = _make_score_result(overall=0.85)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result):
            ev = llm_judge_evaluator(rubric="Is it good?")
            score = await ev("some output")
            assert score == pytest.approx(0.85)

    async def test_default_judge_model_uses_task_selection(self) -> None:
        """When no judge_model given, uses get_model('judging')."""
        mock_result = _make_score_result(overall=0.85)
        with (
            patch("prompt_eval.evaluators.get_model", return_value="selected-judge"),
            patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score,
        ):
            ev = llm_judge_evaluator(rubric="test")
            await ev("some output")
            # The resolved judge model should be passed to ascore_output
            assert mock_score.call_args.kwargs["judge_model"] == "selected-judge"

    async def test_clamps_to_unit_range(self) -> None:
        """Scores are clamped to [0.0, 1.0]."""
        # Score > 1.0 gets clamped
        mock_result = _make_score_result(overall=1.5)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result):
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 1.0

        # Score < 0.0 gets clamped
        mock_result = _make_score_result(overall=-0.3)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result):
            ev = llm_judge_evaluator(rubric="test")
            assert await ev("output") == 0.0

    async def test_passes_expected_as_context(self) -> None:
        """When expected is provided, it appears in the context arg."""
        mock_result = _make_score_result(overall=0.90)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_evaluator(rubric="test")
            await ev("output", expected="reference answer")
            context = mock_score.call_args.kwargs["context"]
            assert "reference answer" in context

    async def test_custom_output_formatter(self) -> None:
        """output_formatter transforms the output before passing to scoring."""
        mock_result = _make_score_result(overall=0.70)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_evaluator(
                rubric="test",
                output_formatter=lambda x: f"FORMATTED: {x}",
            )
            await ev("raw output")
            output_arg = mock_score.call_args.kwargs["output"]
            assert output_arg == "FORMATTED: raw output"

    async def test_rubric_text_in_inline_rubric(self) -> None:
        """The rubric text is passed through to the Rubric object."""
        mock_result = _make_score_result(overall=0.50)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_evaluator(rubric="Codes must be grounded in participant quotes")
            await ev("some codes")
            rubric_arg = mock_score.call_args.kwargs["rubric"]
            assert rubric_arg.description == "Codes must be grounded in participant quotes"

    async def test_explicit_judge_model(self) -> None:
        """Explicit judge_model overrides the registry default."""
        mock_result = _make_score_result(overall=0.75)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_evaluator(rubric="test", judge_model="custom-model")
            await ev("output")
            assert mock_score.call_args.kwargs["judge_model"] == "custom-model"

    async def test_timeout_is_forwarded_to_scoring(self) -> None:
        """Backward-compatible timeout is forwarded to the scoring layer."""
        mock_result = _make_score_result(overall=0.75)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_evaluator(rubric="test", timeout=180)
            await ev("output")
            assert mock_score.call_args.kwargs["timeout"] == 180

    async def test_scoring_error_propagates(self) -> None:
        """Errors from ascore_output propagate to the caller."""
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, side_effect=RuntimeError("judge failed")):
            ev = llm_judge_evaluator(rubric="test")
            with pytest.raises(RuntimeError, match="judge failed"):
                await ev("output")


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
    # mock-ok: testing evaluator-to-scoring delegation, not actual LLM calls

    @pytest.fixture
    def dimensions(self) -> list[RubricDimension]:
        return [
            RubricDimension(name="clarity", description="Is it clear?", weight=1.0),
            RubricDimension(name="depth", description="Is it deep?", weight=1.0),
        ]

    async def test_returns_eval_score(self, dimensions: list[RubricDimension]) -> None:
        """Dimensional evaluator returns EvalScore with per-dimension scores."""
        # ScoreResult with 1-5 scale raw scores: clarity=4, depth=3
        # Normalized to 0-1: clarity=(4-1)/4=0.75, depth=(3-1)/4=0.5
        # Overall: (0.75+0.5)/2=0.625
        mock_result = _make_score_result(
            overall=0.625,
            dimensions={"clarity": 4, "depth": 3},
            reasoning={"clarity": "Clear writing", "depth": "Some depth"},
        )
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result):
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            result = await ev("some output")

        assert isinstance(result, EvalScore)
        assert result.score == pytest.approx(0.625)
        assert result.dimension_scores["clarity"] == pytest.approx(0.75)
        assert result.dimension_scores["depth"] == pytest.approx(0.5)
        assert "clarity" in result.reasoning
        assert "depth" in result.reasoning

    async def test_default_dimensional_judge_uses_task_selection(
        self,
        dimensions: list[RubricDimension],
    ) -> None:
        """When no judge_models given, uses get_model('judging')."""
        mock_result = _make_score_result(overall=0.7)
        with (
            patch("prompt_eval.evaluators.get_model", return_value="selected-judge"),
            patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score,
        ):
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            await ev("some output")
            assert mock_score.call_args.kwargs["judge_model"] == "selected-judge"

    async def test_multi_judge_averages(self, dimensions: list[RubricDimension]) -> None:
        """Multiple judges have their scores averaged via ascore_output_multi_judge."""
        # Multi-judge returns averaged ScoreResult
        mock_result = _make_score_result(
            overall=0.625,  # average of two judges
            dimensions={"clarity": 4, "depth": 4},  # averaged raw scores
            reasoning={
                "clarity": "[model-a] Good | [model-b] Great",
                "depth": "[model-a] Ok | [model-b] Fine",
            },
            judge_model="model-a,model-b",
        )
        with patch("prompt_eval.evaluators.ascore_output_multi_judge", new_callable=AsyncMock, return_value=mock_result):
            ev = llm_judge_dimensional_evaluator(
                dimensions=dimensions,
                judge_models=["model-a", "model-b"],
            )
            result = await ev("some output")

        assert isinstance(result, EvalScore)
        assert result.score == pytest.approx(0.625)
        assert "model-a" in result.reasoning
        assert "model-b" in result.reasoning

    async def test_multi_judge_timeout_is_forwarded(self, dimensions: list[RubricDimension]) -> None:
        """Multi-judge dimensional evaluators forward timeout to the multi-judge scorer."""
        mock_result = _make_score_result(
            overall=0.625,
            dimensions={"clarity": 4, "depth": 4},
            reasoning={"clarity": "Good", "depth": "Good"},
            judge_model="model-a,model-b",
        )
        with patch("prompt_eval.evaluators.ascore_output_multi_judge", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(
                dimensions=dimensions,
                judge_models=["model-a", "model-b"],
                timeout=180,
            )
            await ev("some output")
            assert mock_score.call_args.kwargs["timeout"] == 180

    async def test_weighted_dimensions(self) -> None:
        """Weighted dimensions are passed through to the Rubric."""
        dims = [
            RubricDimension(name="major", description="Important", weight=3.0),
            RubricDimension(name="minor", description="Less important", weight=1.0),
        ]
        # Scoring produces weighted result: major=4 (w=3), minor=2 (w=1)
        # Normalized: major=(4-1)/4=0.75, minor=(2-1)/4=0.25
        # Weighted overall: (0.75*3 + 0.25*1)/4 = 2.5/4 = 0.625
        mock_result = _make_score_result(
            overall=0.625,
            dimensions={"major": 4, "minor": 2},
            reasoning={"major": "Good", "minor": "Weak"},
        )
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(dimensions=dims)
            await ev("output")

        # Verify the Rubric passed to ascore_output has correct weights
        rubric_arg = mock_score.call_args.kwargs["rubric"]
        weights = {d.name: d.weight for d in rubric_arg.dimensions}
        assert weights["major"] == 3.0
        assert weights["minor"] == 1.0

    async def test_judge_failure_raises(self, dimensions: list[RubricDimension]) -> None:
        """When scoring fails, the error propagates."""
        with patch(
            "prompt_eval.evaluators.ascore_output",
            new_callable=AsyncMock,
            side_effect=RuntimeError("All 1 judge model(s) failed"),
        ):
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            with pytest.raises(RuntimeError, match="All 1 judge model"):
                await ev("output")

    async def test_includes_expected_as_context(self, dimensions: list[RubricDimension]) -> None:
        """When expected is provided, it appears in the context."""
        mock_result = _make_score_result(overall=0.5)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            await ev("output", expected="reference answer")
            context = mock_score.call_args.kwargs["context"]
            assert "reference answer" in context

    async def test_output_formatter(self, dimensions: list[RubricDimension]) -> None:
        """output_formatter transforms output before scoring."""
        mock_result = _make_score_result(overall=0.5)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(
                dimensions=dimensions,
                output_formatter=lambda x: f"FORMATTED: {x}",
            )
            await ev("raw output")
            output_arg = mock_score.call_args.kwargs["output"]
            assert output_arg == "FORMATTED: raw output"

    async def test_rubric_has_correct_dimensions(self, dimensions: list[RubricDimension]) -> None:
        """The Rubric object passed to scoring has the right dimensions."""
        mock_result = _make_score_result(overall=0.5)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions)
            await ev("output")
            rubric_arg = mock_score.call_args.kwargs["rubric"]
            dim_names = [d.name for d in rubric_arg.dimensions]
            assert "clarity" in dim_names
            assert "depth" in dim_names

    async def test_dimensional_timeout_is_forwarded(self, dimensions: list[RubricDimension]) -> None:
        """Dimensional evaluators forward timeout to the underlying scorer."""
        mock_result = _make_score_result(overall=0.5)
        with patch("prompt_eval.evaluators.ascore_output", new_callable=AsyncMock, return_value=mock_result) as mock_score:
            ev = llm_judge_dimensional_evaluator(dimensions=dimensions, timeout=180)
            await ev("output")
            assert mock_score.call_args.kwargs["timeout"] == 180


class TestRubricFactoryMethods:
    """Test the Rubric.from_inline and Rubric.from_dimensions classmethods."""

    def test_from_inline_basic(self) -> None:
        from prompt_eval.scoring import Rubric

        rubric = Rubric.from_inline("Is the output accurate?")
        assert rubric.name == "inline"
        assert rubric.description == "Is the output accurate?"
        assert len(rubric.dimensions) == 1
        assert rubric.dimensions[0].name == "quality"
        assert rubric.dimensions[0].scale == 5

    def test_from_inline_custom_name(self) -> None:
        from prompt_eval.scoring import Rubric

        rubric = Rubric.from_inline("test", name="my_rubric", scale=10)
        assert rubric.name == "my_rubric"
        assert rubric.dimensions[0].scale == 10

    def test_from_dimensions_basic(self) -> None:
        from prompt_eval.scoring import Rubric

        dims = [
            {"name": "clarity", "description": "Is it clear?", "weight": 1.0},
            {"name": "depth", "description": "Is it deep?", "weight": 2.0},
        ]
        rubric = Rubric.from_dimensions(dims)
        assert rubric.name == "inline_dimensional"
        assert len(rubric.dimensions) == 2
        assert rubric.dimensions[0].name == "clarity"
        assert rubric.dimensions[1].weight == 2.0

    def test_from_dimensions_with_anchors(self) -> None:
        from prompt_eval.scoring import Rubric

        dims = [
            {
                "name": "quality",
                "description": "Overall quality",
                "weight": 1.0,
                "anchors": {"low": "Poor", "high": "Excellent"},
            }
        ]
        rubric = Rubric.from_dimensions(dims)
        desc = rubric.dimensions[0].description
        assert "Overall quality" in desc
        assert "low: Poor" in desc
        assert "high: Excellent" in desc


class TestBackwardCompatTypes:
    """Verify backward-compatible types are still importable and constructable."""

    def test_judge_verdict(self) -> None:
        v = JudgeVerdict(
            reasoning="test",
            scores=[DimensionScore(dimension="x", score=50)],
            overall_score=50,
        )
        assert v.reasoning == "test"
        assert v.scores[0].dimension == "x"

    def test_dimension_score(self) -> None:
        ds = DimensionScore(dimension="test", score=75)
        assert ds.dimension == "test"
        assert ds.score == 75
