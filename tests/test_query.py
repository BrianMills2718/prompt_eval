"""Tests for prompt_eval.query shared-observability loaders."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult
from llm_client.observability import finish_run, log_item, start_run

from prompt_eval.evaluators import EvalScore
from prompt_eval.experiment import Experiment, ExperimentInput, PromptVariant
from prompt_eval.query import load_result_from_observability
from prompt_eval.runner import run_experiment


class TestLoadResultFromObservability:
    @pytest.fixture
    def simple_experiment(self) -> Experiment:
        return Experiment(
            name="query_exp",
            variants=[
                PromptVariant(
                    name="variant_a",
                    prompt_ref="shared.extraction.entity_extract@2",
                    messages=[{"role": "user", "content": "Summarize: {input}"}],
                    model="gemini/gemini-2.5-flash-lite",
                ),
                PromptVariant(
                    name="variant_b",
                    messages=[{"role": "user", "content": "Explain: {input}"}],
                    model="gemini/gemini-2.5-flash-lite",
                ),
            ],
            inputs=[
                ExperimentInput(id="doc1", content="The quick brown fox."),
                ExperimentInput(id="doc2", content="Jumps over the dog."),
            ],
            n_runs=2,
        )

    @pytest.mark.asyncio
    async def test_reconstructs_eval_result_from_shared_runs(
        self,
        simple_experiment: Experiment,
    ) -> None:
        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="summary text",
                usage={"total_tokens": 50},
                cost=0.001,
                model="test",
            )
            original = await run_experiment(
                simple_experiment,
                evaluator=lambda output, expected: 0.85,
            )
        assert original.execution_id is not None

        loaded = load_result_from_observability(
            original.execution_id,
            project="prompt_eval_tests",
            dataset="query_exp",
        )

        assert loaded.execution_id == original.execution_id
        assert loaded.experiment_name == "query_exp"
        assert loaded.variants == ["variant_a", "variant_b"]
        assert len(loaded.trials) == 8
        assert {trial.replicate for trial in loaded.trials} == {0, 1}
        assert all(trial.output == "summary text" for trial in loaded.trials)
        assert all(trial.score == 0.85 for trial in loaded.trials)
        assert all(trial.tokens_used == 50 for trial in loaded.trials)
        assert loaded.summary["variant_a"].mean_score == pytest.approx(0.85)
        assert loaded.summary["variant_b"].mean_score == pytest.approx(0.85)
        assert loaded.summary["variant_a"].n_trials == 4
        assert loaded.summary["variant_b"].n_trials == 4

    @pytest.mark.asyncio
    async def test_reconstructs_corpus_metrics_from_shared_aggregates(
        self,
        simple_experiment: Experiment,
    ) -> None:
        def corpus_eval(outputs: list[object]) -> EvalScore:
            assert len(outputs) == 4
            return EvalScore(
                score=0.6,
                dimension_scores={"coverage": 0.8, "redundancy": 0.4},
                reasoning="Broad enough but somewhat repetitive.",
            )

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="summary text",
                usage={"total_tokens": 50},
                cost=0.001,
                model="test",
            )
            original = await run_experiment(
                simple_experiment,
                evaluator=lambda output, expected: 0.85,
                corpus_evaluator=corpus_eval,
            )
        assert original.execution_id is not None

        loaded = load_result_from_observability(
            original.execution_id,
            project="prompt_eval_tests",
            dataset="query_exp",
        )

        assert loaded.summary["variant_a"].corpus_score == pytest.approx(0.6)
        assert loaded.summary["variant_b"].corpus_score == pytest.approx(0.6)
        assert loaded.summary["variant_a"].corpus_dimension_scores == {
            "coverage": pytest.approx(0.8),
            "redundancy": pytest.approx(0.4),
        }
        assert loaded.summary["variant_b"].corpus_dimension_scores == {
            "coverage": pytest.approx(0.8),
            "redundancy": pytest.approx(0.4),
        }

    def test_missing_execution_id_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="No shared prompt_eval run family found"):
            load_result_from_observability("does-not-exist", project="prompt_eval_tests")

    def test_incomplete_family_fails_loudly(self) -> None:
        run_id = start_run(
            dataset="manual_exp",
            model="test-model",
            task="prompt_eval.run",
            condition_id="variant_a",
            replicate=0,
            scenario_id="manual_exp",
            phase="evaluation",
            provenance={
                "source_package": "prompt_eval",
                "experiment_name": "manual_exp",
                "experiment_execution_id": "family_incomplete",
                "variant_name": "variant_a",
                "variant_count": 2,
                "input_count": 1,
                "n_runs": 1,
                "prompt_template_sha256": "abc",
                "prompt_source": "inline_messages",
            },
            allow_missing_agent_spec=True,
            missing_agent_spec_reason="test fixture for prompt_eval query loader",
            project="prompt_eval_tests",
        )
        log_item(
            run_id=run_id,
            item_id="doc1",
            metrics={"score": 0.5},
            predicted="answer",
            latency_s=0.1,
            cost=0.01,
            trace_id="prompt_eval/family_incomplete/variant_a/r0/doc1",
            extra={"tokens_used": 10, "predicted_format": "text"},
        )
        finish_run(run_id=run_id)

        with pytest.raises(ValueError, match="incomplete or truncated"):
            load_result_from_observability(
                "family_incomplete",
                project="prompt_eval_tests",
                dataset="manual_exp",
            )
