"""Tests for experiment data models."""

import pytest
from pydantic import ValidationError

from prompt_eval.experiment import (
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    EvalResult,
    VariantSummary,
)


class TestPromptVariant:

    def test_requires_explicit_model(self):
        with pytest.raises(ValidationError, match="model"):
            PromptVariant(
                name="test",
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_defaults(self):
        v = PromptVariant(
            name="test",
            messages=[{"role": "user", "content": "hello"}],
            model="gemini/gemini-2.5-flash-lite",
        )
        assert v.temperature == 1.0
        assert v.kwargs == {}

    def test_custom_values(self):
        v = PromptVariant(
            name="custom",
            messages=[{"role": "user", "content": "hi"}],
            prompt_ref="shared.extraction.entity_extract@2",
            model="claude-sonnet-4-5-20250929",
            temperature=0.5,
            kwargs={"max_tokens": 100},
        )
        assert v.prompt_ref == "shared.extraction.entity_extract@2"
        assert v.model == "claude-sonnet-4-5-20250929"
        assert v.kwargs["max_tokens"] == 100


class TestExperiment:

    def test_minimal(self):
        exp = Experiment(
            name="test",
            variants=[
                PromptVariant(
                    name="a",
                    messages=[{"role": "user", "content": "hi"}],
                    model="gemini/gemini-2.5-flash-lite",
                ),
            ],
        )
        assert exp.n_runs == 3
        assert exp.inputs == []
        assert exp.response_model is None

    def test_with_inputs(self):
        exp = Experiment(
            name="test",
            variants=[
                PromptVariant(
                    name="a",
                    messages=[{"role": "user", "content": "{input}"}],
                    model="gemini/gemini-2.5-flash-lite",
                ),
            ],
            inputs=[
                ExperimentInput(id="i1", content="hello"),
                ExperimentInput(id="i2", content="world", expected="greeting"),
            ],
        )
        assert len(exp.inputs) == 2
        assert exp.inputs[1].expected == "greeting"


class TestTrial:

    def test_successful_trial(self):
        t = Trial(
            variant_name="a", input_id="i1", output="hello",
            score=0.9, cost=0.001, latency_ms=150, tokens_used=50,
        )
        assert t.error is None
        assert t.replicate == 0
        assert t.score == 0.9

    def test_failed_trial(self):
        t = Trial(
            variant_name="a", input_id="i1", output=None,
            error="timeout",
        )
        assert t.error == "timeout"
        assert t.score is None


class TestVariantSummary:

    def test_defaults(self):
        vs = VariantSummary(variant_name="test")
        assert vs.n_trials == 0
        assert vs.mean_score is None


class TestEvalResult:

    def test_empty(self):
        r = EvalResult(experiment_name="test")
        assert r.execution_id is None
        assert r.trials == []
        assert r.summary == {}
