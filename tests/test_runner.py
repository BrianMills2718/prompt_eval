"""Tests for experiment runner."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from prompt_eval.experiment import (
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
)
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
        mock_meta = MagicMock(cost=0.001, usage={"total_tokens": 50})
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("summary text", mock_meta)

            result = await run_experiment(simple_experiment)

        # 2 variants * 1 input * 2 runs = 4 trials
        assert len(result.trials) == 4
        assert mock_llm.call_count == 4
        assert set(result.variants) == {"variant_a", "variant_b"}

    @pytest.mark.asyncio
    async def test_with_evaluator(self, simple_experiment):
        mock_meta = MagicMock(cost=0.001, usage={"total_tokens": 50})
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("summary text", mock_meta)

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

        mock_meta = MagicMock(cost=0.001, usage={"total_tokens": 50})
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
        mock_meta = MagicMock(cost=0.001, usage={"total_tokens": 50})
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("text", mock_meta)

            def bad_evaluator(output, expected):
                raise ValueError("eval broke")

            result = await run_experiment(simple_experiment, evaluator=bad_evaluator)

        # Should complete without crashing, scores are None
        assert all(t.score is None for t in result.trials)
        assert all(t.error is None for t in result.trials)

    @pytest.mark.asyncio
    async def test_async_evaluator(self, simple_experiment):
        """Async evaluators (like llm_judge_evaluator) are awaited correctly."""
        mock_meta = MagicMock(cost=0.001, usage={"total_tokens": 50})
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("summary text", mock_meta)

            async def async_eval(output, expected):
                return 0.75

            result = await run_experiment(simple_experiment, evaluator=async_eval)

        for trial in result.trials:
            assert trial.score == 0.75
