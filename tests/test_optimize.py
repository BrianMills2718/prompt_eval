"""Tests for prompt_eval.optimize."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prompt_eval.experiment import ExperimentInput
from prompt_eval.optimize import OptimizeResult, SearchSpace, grid_search, optimize


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


class TestOptimize:
    async def test_grid_search_strategy(self, mock_llm) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        result = await optimize(space, _make_inputs(), _always_one, strategy="grid_search", n_runs=1)
        assert result.strategy == "grid_search"

    async def test_few_shot_not_implemented(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        with pytest.raises(NotImplementedError, match="few_shot_selection"):
            await optimize(space, _make_inputs(), _always_one, strategy="few_shot_selection")

    async def test_instruction_search_not_implemented(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        with pytest.raises(NotImplementedError, match="instruction_search"):
            await optimize(space, _make_inputs(), _always_one, strategy="instruction_search")

    async def test_unknown_strategy_raises(self) -> None:
        space = SearchSpace(
            prompt_templates=[[{"role": "user", "content": "{input}"}]],
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            await optimize(space, _make_inputs(), _always_one, strategy="magic")
