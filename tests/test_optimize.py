"""Tests for prompt_eval.optimize."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

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


def _make_inputs() -> list[ExperimentInput]:
    return [ExperimentInput(id="i1", content="test input", expected="expected")]


def _always_one(output, expected=None) -> float:
    return 1.0


def _score_by_model(output, expected=None) -> float:
    """Used to differentiate models in tests — score depends on output content."""
    return 0.9 if output and "good" in str(output) else 0.3


# mock-ok: we're testing orchestration logic, not LLM calls
@pytest.fixture
def mock_llm():
    with patch("prompt_eval.runner.acall_llm") as mock:
        meta = AsyncMock()
        meta.cost = 0.001
        meta.usage = {"total_tokens": 50}
        mock.return_value = ("mock output", meta)
        yield mock


class TestGridSearch:
    async def test_single_combo(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "say {input}"}]],
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
            meta = AsyncMock()
            meta.cost = 0.001
            meta.usage = {"total_tokens": 50}
            # First template's calls return "bad", second's return "good"
            if call_count <= 1:
                return ("bad output", meta)
            return ("good output", meta)

        mock_llm.side_effect = varying_llm

        space = SearchSpace(
            prompt_templates=[
                [{"role": "user", "content": "bad: {input}"}],
                [{"role": "user", "content": "good: {input}"}],
            ],
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
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)
        assert result.n_trials == 3

    async def test_all_errors_handled(self, mock_llm) -> None:
        """When all LLM calls error, best_score should be 0.0."""
        mock_llm.side_effect = Exception("LLM down")

        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        result = await grid_search(space, _make_inputs(), _always_one, n_runs=1)
        assert result.best_score == 0.0
        assert result.n_trials == 1


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
        )
        result = await few_shot_selection(pool, _make_inputs(), _always_one, n_runs=1)
        assert result.n_trials == 3  # C(3,2) = 3

    async def test_budget_limits_combos(self, mock_llm) -> None:
        pool = FewShotPool(
            examples=["ex1", "ex2", "ex3", "ex4"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
        )
        result = await few_shot_selection(pool, _make_inputs(), _always_one, n_runs=1, budget=2)
        assert result.n_trials == 2  # budget caps at 2


class TestInstructionSearch:
    async def test_basic_search(self, mock_llm) -> None:
        """Instruction search evaluates base + rewrites."""
        # mock-ok: testing orchestration, not rewrite quality
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_meta = AsyncMock()
            rewrite_meta.cost = 0.0
            rewrite_meta.usage = {"total_tokens": 10}
            rewrite_mock.return_value = ("rewrite_a\n---\nrewrite_b", rewrite_meta)

            result = await instruction_search(
                base_instruction="do the thing: {input}",
                inputs=_make_inputs(),
                evaluator=_always_one,
                n_iterations=1,
                n_rewrites=2,
                n_runs=1,
            )

        assert result.strategy == "instruction_search"
        # 1 base + 2 rewrites = 3 trials
        assert result.n_trials == 3
        assert result.best_score == 1.0

    async def test_no_rewrites_generated(self, mock_llm) -> None:
        """If LLM returns empty rewrites, search continues without crashing."""
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_meta = AsyncMock()
            rewrite_meta.cost = 0.0
            rewrite_meta.usage = {"total_tokens": 10}
            rewrite_mock.return_value = ("", rewrite_meta)  # empty

            result = await instruction_search(
                base_instruction="do the thing: {input}",
                inputs=_make_inputs(),
                evaluator=_always_one,
                n_iterations=1,
                n_rewrites=2,
                n_runs=1,
            )

        assert result.strategy == "instruction_search"
        assert result.n_trials == 1  # only base evaluated


class TestOptimize:
    async def test_grid_search_strategy(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        result = await optimize(space, _make_inputs(), _always_one, strategy="grid_search", n_runs=1)
        assert result.strategy == "grid_search"

    async def test_few_shot_strategy(self, mock_llm) -> None:
        space = SearchSpace(prompt_templates=[[{"role": "user", "content": "{input}"}]])
        pool = FewShotPool(
            examples=["ex1", "ex2"],
            k=2,
            base_messages=[{"role": "user", "content": "{examples}\n\n{input}"}],
        )
        result = await optimize(
            space, _make_inputs(), _always_one,
            strategy="few_shot_selection", n_runs=1, pool=pool,
        )
        assert result.strategy == "few_shot_selection"

    async def test_few_shot_missing_pool_raises(self) -> None:
        space = SearchSpace(prompt_templates=[[{"role": "user", "content": "{input}"}]])
        with pytest.raises(ValueError, match="requires 'pool'"):
            await optimize(space, _make_inputs(), _always_one, strategy="few_shot_selection")

    async def test_instruction_search_strategy(self, mock_llm) -> None:
        space = SearchSpace(prompt_templates=[[{"role": "user", "content": "{input}"}]])
        with patch("prompt_eval.optimize.acall_llm") as rewrite_mock:
            rewrite_meta = AsyncMock()
            rewrite_meta.cost = 0.0
            rewrite_meta.usage = {"total_tokens": 10}
            rewrite_mock.return_value = ("rewrite_a", rewrite_meta)

            result = await optimize(
                space, _make_inputs(), _always_one,
                strategy="instruction_search", n_runs=1,
                base_instruction="do: {input}", n_iterations=1, n_rewrites=1,
            )
        assert result.strategy == "instruction_search"

    async def test_instruction_search_missing_instruction_raises(self) -> None:
        space = SearchSpace(prompt_templates=[[{"role": "user", "content": "{input}"}]])
        with pytest.raises(ValueError, match="requires 'base_instruction'"):
            await optimize(space, _make_inputs(), _always_one, strategy="instruction_search")

    async def test_unknown_strategy_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            await optimize(space, _make_inputs(), _always_one, strategy="magic")
