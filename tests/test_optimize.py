"""Tests for prompt_eval.optimize."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm_client import LLMCallResult

from prompt_eval.experiment import ExperimentInput
from prompt_eval.optimize import (
    FewShotPool,
    OptimizeResult,
    SearchSpace,
    _inject_examples,
    few_shot_selection,
    grid_search,
    instruction_search,
    optimize,
)
from prompt_eval.prompt_templates import INSTRUCTION_REWRITE_TEMPLATE_PATH


def _make_inputs() -> list[ExperimentInput]:
    return [ExperimentInput(id="i1", content="test input", expected="expected")]


def _always_one(output, expected=None) -> float:
    return 1.0


def _score_by_model(output, expected=None) -> float:
    """Used to differentiate models in tests — score depends on output content."""
    return 0.9 if output and "good" in str(output) else 0.3


DEFAULT_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_REWRITE_MODEL = "gpt-5-mini"


# mock-ok: we're testing orchestration logic, not LLM calls
@pytest.fixture
def mock_llm():
    with patch("prompt_eval.runner.acall_llm") as mock:
        mock.return_value = LLMCallResult(content="mock output", usage={"total_tokens": 50}, cost=0.001, model="test")
        yield mock


class TestGridSearch:
    async def test_single_combo(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "say {input}"}]],
            models=[DEFAULT_MODEL],
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)

        assert isinstance(result, OptimizeResult)
        assert result.strategy == "grid_search"
        assert result.n_trials == 1
        assert result.best_score == 1.0
        assert len(result.all_results) == 1

    async def test_multi_combo_picks_best(self, mock_llm) -> None:
        """With two templates, the one scoring higher wins."""
        call_count = 0

        async def varying_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First template's calls return "bad", second's return "good"
            if call_count <= 1:
                return LLMCallResult(content="bad output", usage={"total_tokens": 50}, cost=0.001, model="test")
            return LLMCallResult(content="good output", usage={"total_tokens": 50}, cost=0.001, model="test")

        mock_llm.side_effect = varying_llm

        space = SearchSpace(
            prompt_templates=[
                [{"role": "user", "content": "bad: {input}"}],
                [{"role": "user", "content": "good: {input}"}],
            ],
            models=[DEFAULT_MODEL],
        )
        result = await grid_search(space, _make_inputs(), _score_by_model, n_runs=1)

        assert result.n_trials == 2
        assert result.best_score == 0.9
        assert "tmpl1" in result.best_variant

    async def test_multiple_models(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=["model_a", "model_b"],
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)
        assert result.n_trials == 2

    async def test_multiple_temperatures(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            temperatures=[0.0, 0.5, 1.0],
            models=[DEFAULT_MODEL],
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)
        assert result.n_trials == 3

    async def test_all_errors_handled(self, mock_llm) -> None:
        """When all LLM calls error, best_score should be 0.0."""
        mock_llm.side_effect = Exception("LLM down")

        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)
        assert result.best_score == 0.0
        assert result.n_trials == 1

    def test_requires_explicit_models(self) -> None:
        with pytest.raises(ValidationError, match="models"):
            SearchSpace(prompt_templates=[[{"role": "user", "content": "{input}"}]])


class TestInjectExamples:
    def test_basic_injection(self) -> None:
        messages = [{"role": "user", "content": "Examples:\n{examples}\n\nNow do: {input}"}]
        result = _inject_examples(messages, ["ex1", "ex2"], "\n")
        assert result[0]["content"] == "Examples:\nex1\nex2\n\nNow do: {input}"

    def test_custom_separator(self) -> None:
        messages = [{"role": "user", "content": "{examples}"}]
        result = _inject_examples(messages, ["a", "b", "c"], " | ")
        assert result[0]["content"] == "a | b | c"

    def test_no_placeholder(self) -> None:
        messages = [{"role": "user", "content": "no placeholder here"}]
        result = _inject_examples(messages, ["ex1"], "\n")
        assert result[0]["content"] == "no placeholder here"


class TestFewShotSelection:
    async def test_single_combo(self, mock_llm) -> None:
        pool = FewShotPool(
            examples=["ex1", "ex2"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
            model=DEFAULT_MODEL,
        )
        result = await few_shot_selection(pool, _make_inputs(), _always_one, n_runs=1)
        assert result.strategy == "few_shot_selection"
        assert result.n_trials == 1  # C(2,2) = 1
        assert result.best_score == 1.0

    async def test_multiple_combos(self, mock_llm) -> None:
        pool = FewShotPool(
            examples=["ex1", "ex2", "ex3"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
            model=DEFAULT_MODEL,
        )
        result = await few_shot_selection(pool, _make_inputs(), _always_one, n_runs=1)
        assert result.n_trials == 3  # C(3,2) = 3

    async def test_budget_limits_combos(self, mock_llm) -> None:
        pool = FewShotPool(
            examples=["ex1", "ex2", "ex3", "ex4"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
            model=DEFAULT_MODEL,
        )
        result = await few_shot_selection(pool, _make_inputs(), _always_one, n_runs=1, budget=2)
        assert result.n_trials == 2  # budget caps at 2

    def test_requires_explicit_model(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            FewShotPool(
                examples=["ex1", "ex2"],
                k=2,
                base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
            )


class TestInstructionSearch:
    async def test_basic_search(self, mock_llm) -> None:
        """Instruction search evaluates base + rewrites."""
        # mock-ok: testing orchestration, not rewrite quality
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_mock.return_value = LLMCallResult(content="rewrite_a\n---\nrewrite_b", usage={"total_tokens": 10}, cost=0.0, model="test")

            result = await instruction_search(
                base_instruction="do the thing: {input}",
                inputs=_make_inputs(),
                evaluator=_always_one,
                n_iterations=1,
                n_rewrites=2,
                n_runs=1,
                model=DEFAULT_MODEL,
                rewrite_model=DEFAULT_REWRITE_MODEL,
            )

        assert result.strategy == "instruction_search"
        # 1 base + 2 rewrites = 3 trials
        assert result.n_trials == 3
        assert result.best_score == 1.0

    async def test_no_rewrites_generated(self, mock_llm) -> None:
        """If LLM returns empty rewrites, search continues without crashing."""
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_mock.return_value = LLMCallResult(content="", usage={"total_tokens": 10}, cost=0.0, model="test")  # empty

            result = await instruction_search(
                base_instruction="do the thing: {input}",
                inputs=_make_inputs(),
                evaluator=_always_one,
                n_iterations=1,
                n_rewrites=2,
                n_runs=1,
                model=DEFAULT_MODEL,
                rewrite_model=DEFAULT_REWRITE_MODEL,
            )

        assert result.strategy == "instruction_search"
        assert result.n_trials == 1  # only base evaluated

    async def test_uses_yaml_template(self, mock_llm) -> None:
        with (
            patch(
                "prompt_eval.prompt_templates.render_prompt",
                return_value=[{"role": "user", "content": "rendered rewrite prompt"}],
            ) as mock_render,
            patch("prompt_eval.optimize.acall_llm") as rewrite_mock,
        ):
            rewrite_mock.return_value = LLMCallResult(
                content="rewrite_a",
                usage={"total_tokens": 10},
                cost=0.0,
                model="test",
            )
            await instruction_search(
                base_instruction="do the thing: {input}",
                inputs=_make_inputs(),
                evaluator=_always_one,
                n_iterations=1,
                n_rewrites=1,
                n_runs=1,
                model=DEFAULT_MODEL,
                rewrite_model=DEFAULT_REWRITE_MODEL,
            )

        assert mock_render.call_args.kwargs["template_path"] == INSTRUCTION_REWRITE_TEMPLATE_PATH
        assert rewrite_mock.call_args.args[1] == [{"role": "user", "content": "rendered rewrite prompt"}]


class TestOptimize:
    async def test_grid_search_strategy(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        result = await optimize(space, _make_inputs(), _always_one, strategy="grid_search", n_runs=1)
        assert result.strategy == "grid_search"

    async def test_few_shot_strategy(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        pool = FewShotPool(
            examples=["ex1", "ex2"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
            model=DEFAULT_MODEL,
        )
        result = await optimize(
            space, _make_inputs(), _always_one,
            strategy="few_shot_selection", n_runs=1, pool=pool,
        )
        assert result.strategy == "few_shot_selection"

    async def test_few_shot_missing_pool_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with pytest.raises(ValueError, match="requires 'pool'"):
            await optimize(space, _make_inputs(), _always_one, strategy="few_shot_selection")

    async def test_instruction_search_strategy(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_mock.return_value = LLMCallResult(content="rewrite_a", usage={"total_tokens": 10}, cost=0.0, model="test")

            result = await optimize(
                space, _make_inputs(), _always_one,
                strategy="instruction_search", n_runs=1,
                base_instruction="do: {input}", n_iterations=1, n_rewrites=1,
                model=DEFAULT_MODEL,
                rewrite_model=DEFAULT_REWRITE_MODEL,
            )
        assert result.strategy == "instruction_search"

    async def test_instruction_search_missing_instruction_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with pytest.raises(ValueError, match="requires 'base_instruction'"):
            await optimize(space, _make_inputs(), _always_one, strategy="instruction_search")

    async def test_instruction_search_missing_model_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with pytest.raises(ValueError, match="requires explicit 'model'"):
            await optimize(
                space,
                _make_inputs(),
                _always_one,
                strategy="instruction_search",
                base_instruction="do: {input}",
                rewrite_model=DEFAULT_REWRITE_MODEL,
            )

    async def test_instruction_search_missing_rewrite_model_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with pytest.raises(ValueError, match="requires explicit 'rewrite_model'"):
            await optimize(
                space,
                _make_inputs(),
                _always_one,
                strategy="instruction_search",
                base_instruction="do: {input}",
                model=DEFAULT_MODEL,
            )

    async def test_unknown_strategy_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
            models=[DEFAULT_MODEL],
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            await optimize(space, _make_inputs(), _always_one, strategy="magic")
