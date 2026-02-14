"""Tests for prompt_eval.mcp_server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

fastmcp = pytest.importorskip("fastmcp")

from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    VariantSummary,
)
from prompt_eval.mcp_server import (
    _compare_impl,
    _evaluate_output_impl,
    _get_result_impl,
    _list_experiments_impl,
    _run_experiment_impl,
)
from prompt_eval.store import save_result


def _make_result(name: str = "test_exp") -> EvalResult:
    return EvalResult(
        experiment_name=name,
        variants=["a", "b"],
        trials=[
            Trial(variant_name="a", input_id="i1", output="hello", score=0.9),
            Trial(variant_name="a", input_id="i1", output="hello", score=0.8),
            Trial(variant_name="b", input_id="i1", output="world", score=0.5),
            Trial(variant_name="b", input_id="i1", output="world", score=0.6),
        ],
        summary={
            "a": VariantSummary(variant_name="a", n_trials=2, mean_score=0.85),
            "b": VariantSummary(variant_name="b", n_trials=2, mean_score=0.55),
        },
    )


# mock-ok: testing MCP tool orchestration, not LLM calls
@pytest.fixture
def mock_llm():
    with patch("prompt_eval.runner.acall_llm") as mock:
        meta = AsyncMock()
        meta.cost = 0.001
        meta.usage = {"total_tokens": 50}
        mock.return_value = ("mock output", meta)
        yield mock


class TestRunExperimentTool:
    async def test_runs_and_saves(self, mock_llm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)

        exp = Experiment(
            name="mcp_test",
            variants=[PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}])],
            inputs=[ExperimentInput(id="i1", content="hi")],
            n_runs=1,
        )
        result = await _run_experiment_impl(exp.model_dump_json())
        assert result["experiment_name"] == "mcp_test"
        assert result["n_trials"] == 1
        assert "saved_to" in result

    async def test_unknown_evaluator(self, mock_llm) -> None:
        exp = Experiment(
            name="test",
            variants=[PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}])],
            inputs=[ExperimentInput(id="i1", content="hi")],
            n_runs=1,
        )
        result = await _run_experiment_impl(exp.model_dump_json(), evaluator_name="nonexistent")
        assert "error" in result


    async def test_kappa_evaluator(self, mock_llm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)

        exp = Experiment(
            name="kappa_test",
            variants=[PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}])],
            inputs=[ExperimentInput(id="i1", content="hi", expected=["code_a", "code_b"])],
            n_runs=1,
        )
        # Mock returns a list (kappa evaluator expects list output)
        mock_llm.return_value = (["code_a", "code_b"], mock_llm.return_value[1])
        result = await _run_experiment_impl(exp.model_dump_json(), evaluator_name="kappa")
        assert result["experiment_name"] == "kappa_test"
        assert result["summary"]["v1"]["mean_score"] == 1.0

    async def test_llm_judge_evaluator(self, mock_llm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)

        exp = Experiment(
            name="judge_test",
            variants=[PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}])],
            inputs=[ExperimentInput(id="i1", content="hi")],
            n_runs=1,
        )
        # mock acall_llm for both the experiment run AND the judge call
        with patch("prompt_eval.evaluators.acall_llm", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = ("0.85", AsyncMock())
            result = await _run_experiment_impl(
                exp.model_dump_json(),
                evaluator_name="llm_judge",
                rubric="Is the output helpful?",
            )
        assert result["experiment_name"] == "judge_test"
        assert result["summary"]["v1"]["mean_score"] == 0.85

    async def test_llm_judge_requires_rubric(self, mock_llm) -> None:
        exp = Experiment(
            name="test",
            variants=[PromptVariant(name="v1", messages=[{"role": "user", "content": "{input}"}])],
            inputs=[ExperimentInput(id="i1", content="hi")],
            n_runs=1,
        )
        result = await _run_experiment_impl(exp.model_dump_json(), evaluator_name="llm_judge")
        assert "error" in result
        assert "rubric" in result["error"]


class TestGetResult:
    async def test_by_path(self, tmp_path: Path) -> None:
        path = save_result(_make_result(), path=tmp_path / "r.json")
        result = await _get_result_impl(path=str(path))
        assert result["experiment_name"] == "test_exp"

    async def test_by_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        monkeypatch.setattr("prompt_eval.mcp_server.list_results", lambda name, **kw: [tmp_path / "r.json"])
        save_result(_make_result("my_exp"), path=tmp_path / "r.json")
        result = await _get_result_impl(experiment_name="my_exp")
        assert result["experiment_name"] == "my_exp"

    async def test_no_args(self) -> None:
        result = await _get_result_impl()
        assert "error" in result


class TestListExperiments:
    async def test_lists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        monkeypatch.setattr("prompt_eval.mcp_server.list_results", lambda **kw: [])
        result = await _list_experiments_impl()
        assert "experiments" in result
        assert result["total_results"] == 0


class TestEvaluateOutput:
    async def test_exact_match(self) -> None:
        result = await _evaluate_output_impl("hello", "exact_match", expected="hello")
        assert result["score"] == 1.0
        assert result["evaluator"] == "exact_match"

    async def test_exact_match_no_match(self) -> None:
        result = await _evaluate_output_impl("hello", "exact_match", expected="world")
        assert result["score"] == 0.0

    async def test_contains(self) -> None:
        result = await _evaluate_output_impl("the answer is 42", "contains", expected="42")
        assert result["score"] == 1.0

    async def test_kappa(self) -> None:
        # kappa with string input wraps in list, so single string vs single string
        result = await _evaluate_output_impl("code_a", "kappa", expected="code_a")
        assert result["score"] == 1.0

    async def test_llm_judge(self) -> None:
        # mock-ok: testing tool orchestration, not actual LLM judge quality
        with patch("prompt_eval.evaluators.acall_llm", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = ("0.9", AsyncMock())
            result = await _evaluate_output_impl(
                "some output", "llm_judge", rubric="Is it good?"
            )
        assert result["score"] == 0.9

    async def test_llm_judge_requires_rubric(self) -> None:
        result = await _evaluate_output_impl("output", "llm_judge")
        assert "error" in result

    async def test_unknown_evaluator(self) -> None:
        result = await _evaluate_output_impl("output", "nonexistent")
        assert "error" in result


class TestCompare:
    async def test_compare_variants(self, tmp_path: Path) -> None:
        path = save_result(_make_result(), path=tmp_path / "r.json")
        result = await _compare_impl(str(path), "a", "b")
        assert result["variant_a"] == "a"
        assert result["variant_b"] == "b"
        assert "significant" in result
        assert "difference" in result
