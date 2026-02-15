"""Tests for experiment runner."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult

from prompt_eval.experiment import (
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
)
from prompt_eval.evaluators import EvalScore
from prompt_eval.runner import run_experiment, _substitute_input, _build_summaries


class TestSubstituteInput:

    def test_replaces_placeholder(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Analyze: {input}"},
        ]
        result = _substitute_input(messages, "some text")
        assert result[0]["content"] == "You are helpful."
        assert result[1]["content"] == "Analyze: some text"

    def test_no_placeholder(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _substitute_input(messages, "ignored")
        assert result[0]["content"] == "hello"

    def test_multiple_placeholders(self):
        messages = [{"role": "user", "content": "{input} and {input}"}]
        result = _substitute_input(messages, "X")
        assert result[0]["content"] == "X and X"


class TestBuildSummaries:

    def test_basic_summary(self):
        trials = [
            Trial(variant_name="a", input_id="i1", output="x", score=0.8, cost=0.01, latency_ms=100, tokens_used=50),
            Trial(variant_name="a", input_id="i2", output="y", score=0.6, cost=0.02, latency_ms=200, tokens_used=60),
            Trial(variant_name="b", input_id="i1", output="z", score=0.9, cost=0.01, latency_ms=150, tokens_used=40),
        ]
        summaries = _build_summaries(trials, ["a", "b"])
        assert summaries["a"].n_trials == 2
        assert summaries["a"].mean_score == pytest.approx(0.7)
        assert summaries["b"].n_trials == 1
        assert summaries["b"].mean_score == 0.9

    def test_with_errors(self):
        trials = [
            Trial(variant_name="a", input_id="i1", output="x", score=0.8, cost=0.01, latency_ms=100, tokens_used=50),
            Trial(variant_name="a", input_id="i2", output=None, error="timeout", latency_ms=5000),
        ]
        summaries = _build_summaries(trials, ["a"])
        assert summaries["a"].n_trials == 2
        assert summaries["a"].n_errors == 1
        assert summaries["a"].mean_score == 0.8

    def test_no_scores(self):
        trials = [
            Trial(variant_name="a", input_id="i1", output="x", cost=0.01, latency_ms=100, tokens_used=50),
        ]
        summaries = _build_summaries(trials, ["a"])
        assert summaries["a"].mean_score is None


class TestRunExperiment:

    @pytest.fixture
    def simple_experiment(self):
        return Experiment(
            name="test_exp",
            variants=[
                PromptVariant(
                    name="variant_a",
                    messages=[{"role": "user", "content": "Summarize: {input}"}],
                ),
                PromptVariant(
                    name="variant_b",
                    messages=[
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Summarize: {input}"},
                    ],
                ),
            ],
            inputs=[
                ExperimentInput(id="doc1", content="The quick brown fox."),
            ],
            n_runs=2,
        )

    @pytest.mark.asyncio
    async def test_runs_all_combinations(self, simple_experiment):
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="summary text", usage={"total_tokens": 50}, cost=0.001, model="test")

            result = await run_experiment(simple_experiment)

        # 2 variants * 1 input * 2 runs = 4 trials
        assert len(result.trials) == 4
        assert mock_llm.call_count == 4
        assert set(result.variants) == {"variant_a", "variant_b"}

    @pytest.mark.asyncio
    async def test_with_evaluator(self, simple_experiment):
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="summary text", usage={"total_tokens": 50}, cost=0.001, model="test")

            evaluator = lambda output, expected: 0.85
            result = await run_experiment(simple_experiment, evaluator=evaluator)

        for trial in result.trials:
            assert trial.score == 0.85
        assert result.summary["variant_a"].mean_score == 0.85

    @pytest.mark.asyncio
    async def test_structured_output(self):
        from pydantic import BaseModel

        class Summary(BaseModel):
            text: str

        exp = Experiment(
            name="structured",
            variants=[
                PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}]),
            ],
            inputs=[ExperimentInput(id="i1", content="test")],
            n_runs=1,
            response_model=Summary,
        )

        mock_meta = LLMCallResult(content="", usage={"total_tokens": 50}, cost=0.001, model="test")
        with patch("prompt_eval.runner.acall_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (Summary(text="result"), mock_meta)

            result = await run_experiment(exp)

        assert len(result.trials) == 1
        assert result.trials[0].output.text == "result"
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_error_captured(self, simple_experiment):
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")

            result = await run_experiment(simple_experiment)

        assert all(t.error == "API error" for t in result.trials)
        assert result.summary["variant_a"].n_errors == 2

    @pytest.mark.asyncio
    async def test_evaluator_error_doesnt_crash(self, simple_experiment):
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="text", usage={"total_tokens": 50}, cost=0.001, model="test")

            def bad_evaluator(output, expected):
                raise ValueError("eval broke")

            result = await run_experiment(simple_experiment, evaluator=bad_evaluator)

        # Should complete without crashing, scores are None
        assert all(t.score is None for t in result.trials)
        assert all(t.error is None for t in result.trials)

    @pytest.mark.asyncio
    async def test_async_evaluator(self, simple_experiment):
        """Async evaluators (like llm_judge_evaluator) are awaited correctly."""
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="summary text", usage={"total_tokens": 50}, cost=0.001, model="test")

            async def async_eval(output, expected):
                return 0.75

            result = await run_experiment(simple_experiment, evaluator=async_eval)

        for trial in result.trials:
            assert trial.score == 0.75

    @pytest.mark.asyncio
    async def test_eval_score_populates_dimensions(self, simple_experiment):
        """EvalScore from dimensional evaluator populates Trial fields."""
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="text", usage={"total_tokens": 50}, cost=0.001, model="test")

            async def dim_eval(output, expected):
                return EvalScore(
                    score=0.65,
                    dimension_scores={"clarity": 0.8, "depth": 0.5},
                    reasoning="Good clarity, weak depth.",
                )

            result = await run_experiment(simple_experiment, evaluator=dim_eval)

        for trial in result.trials:
            assert trial.score == 0.65
            assert trial.dimension_scores == {"clarity": 0.8, "depth": 0.5}
            assert trial.reasoning == "Good clarity, weak depth."

        # Check summary aggregates dimension means
        for summary in result.summary.values():
            assert summary.dimension_means is not None
            assert summary.dimension_means["clarity"] == pytest.approx(0.8)
            assert summary.dimension_means["depth"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_float_evaluator_no_dimensions(self, simple_experiment):
        """Plain float evaluators still work — dimensions stay None."""
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(content="text", usage={"total_tokens": 50}, cost=0.001, model="test")

            result = await run_experiment(simple_experiment, evaluator=lambda o, e: 0.9)

        for trial in result.trials:
            assert trial.score == 0.9
            assert trial.dimension_scores is None
            assert trial.reasoning is None

        for summary in result.summary.values():
            assert summary.dimension_means is None

    @pytest.mark.asyncio
    async def test_corpus_evaluator_sync(self, simple_experiment):
        """Sync corpus evaluator receives all outputs and produces corpus_score."""
        captured_outputs = {}

        def corpus_eval(outputs):
            # Capture what was passed for verification
            captured_outputs[len(captured_outputs)] = list(outputs)
            return len(outputs) / 10.0  # deterministic score based on count

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="summary text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(simple_experiment, corpus_evaluator=corpus_eval)

        # 2 variants, so corpus_eval called twice
        assert len(captured_outputs) == 2
        # Each variant: 1 input * 2 runs = 2 outputs
        for outputs in captured_outputs.values():
            assert len(outputs) == 2
            assert all(o == "summary text" for o in outputs)

        for summary in result.summary.values():
            assert summary.corpus_score == pytest.approx(0.2)  # 2/10

    @pytest.mark.asyncio
    async def test_corpus_evaluator_async(self, simple_experiment):
        """Async corpus evaluator is awaited correctly."""
        async def corpus_eval(outputs):
            return 0.75

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(simple_experiment, corpus_evaluator=corpus_eval)

        for summary in result.summary.values():
            assert summary.corpus_score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_corpus_evaluator_eval_score(self, simple_experiment):
        """Corpus evaluator returning EvalScore populates dimension scores."""
        def corpus_eval(outputs):
            return EvalScore(
                score=0.82,
                dimension_scores={"fluency": 0.9, "consistency": 0.74},
            )

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(simple_experiment, corpus_evaluator=corpus_eval)

        for summary in result.summary.values():
            assert summary.corpus_score == pytest.approx(0.82)
            assert summary.corpus_dimension_scores == {"fluency": 0.9, "consistency": 0.74}

    @pytest.mark.asyncio
    async def test_corpus_evaluator_error_doesnt_crash(self, simple_experiment):
        """Corpus evaluator exception results in None score, not a crash."""
        def corpus_eval(outputs):
            raise RuntimeError("fingerprint failed")

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(simple_experiment, corpus_evaluator=corpus_eval)

        for summary in result.summary.values():
            assert summary.corpus_score is None
            assert summary.corpus_dimension_scores is None

    @pytest.mark.asyncio
    async def test_corpus_evaluator_with_regular_evaluator(self, simple_experiment):
        """Both per-trial and corpus evaluators can run together."""
        def per_trial(output, expected):
            return 0.85

        def corpus_eval(outputs):
            return 0.6

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(
                simple_experiment, evaluator=per_trial, corpus_evaluator=corpus_eval,
            )

        # Per-trial scores populated
        for trial in result.trials:
            assert trial.score == 0.85
        # Corpus scores populated
        for summary in result.summary.values():
            assert summary.mean_score == pytest.approx(0.85)
            assert summary.corpus_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_no_corpus_evaluator_fields_none(self, simple_experiment):
        """Without corpus_evaluator, corpus fields stay None (backward compat)."""
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="text", usage={"total_tokens": 50}, cost=0.001, model="test"
            )
            result = await run_experiment(simple_experiment)

        for summary in result.summary.values():
            assert summary.corpus_score is None
            assert summary.corpus_dimension_scores is None
