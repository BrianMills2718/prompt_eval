"""Tests for prompt_eval.store — persistence layer."""

from pathlib import Path

import pytest

from prompt_eval.experiment import (
    EvalResult,
    Experiment,
    ExperimentInput,
    PromptVariant,
    Trial,
    VariantSummary,
)
from prompt_eval.store import (
    list_results,
    load_experiment,
    load_result,
    save_experiment,
    save_result,
)


def _make_result(name: str = "test_exp") -> EvalResult:
    return EvalResult(
        experiment_name=name,
        variants=["a", "b"],
        trials=[
            Trial(variant_name="a", input_id="i1", output="hello", score=0.8),
            Trial(variant_name="b", input_id="i1", output="world", score=0.6),
        ],
        summary={
            "a": VariantSummary(variant_name="a", n_trials=1, mean_score=0.8),
            "b": VariantSummary(variant_name="b", n_trials=1, mean_score=0.6),
        },
    )


def _make_experiment(name: str = "test_exp") -> Experiment:
    return Experiment(
        name=name,
        variants=[
            PromptVariant(name="a", messages=[{"role": "user", "content": "say {input}"}]),
        ],
        inputs=[ExperimentInput(id="i1", content="hello")],
        n_runs=2,
    )


class TestSaveLoadResult:
    def test_roundtrip(self, tmp_path: Path) -> None:
        result = _make_result()
        path = save_result(result, path=tmp_path / "r.json")
        loaded = load_result(path)
        assert loaded.experiment_name == result.experiment_name
        assert len(loaded.trials) == 2
        assert loaded.trials[0].score == 0.8
        assert loaded.summary["a"].mean_score == 0.8

    def test_auto_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        result = _make_result("my_exp")
        path = save_result(result)
        assert path.parent.name == "my_exp"
        assert path.name.startswith("my_exp_")
        assert path.suffix == ".json"
        loaded = load_result(path)
        assert loaded.experiment_name == "my_exp"

    def test_creates_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "result.json"
        save_result(_make_result(), path=deep)
        assert deep.exists()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid result file"):
            load_result(bad)


class TestSaveLoadExperiment:
    def test_roundtrip(self, tmp_path: Path) -> None:
        exp = _make_experiment()
        path = save_experiment(exp, path=tmp_path / "e.json")
        loaded = load_experiment(path)
        assert loaded.name == exp.name
        assert len(loaded.variants) == 1
        assert loaded.inputs[0].content == "hello"
        assert loaded.n_runs == 2

    def test_auto_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        exp = _make_experiment("my_exp")
        path = save_experiment(exp)
        assert path.name == "my_exp_def.json"

    def test_response_model_not_roundtripped(self, tmp_path: Path) -> None:
        """response_model is a Python type and can't survive JSON roundtrip."""
        exp = _make_experiment()
        exp.response_model = None  # explicitly None
        path = save_experiment(exp, path=tmp_path / "e.json")
        loaded = load_experiment(path)
        assert loaded.response_model is None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid")
        with pytest.raises(ValueError, match="Invalid experiment file"):
            load_experiment(bad)


class TestListResults:
    def test_empty_base(self, tmp_path: Path) -> None:
        assert list_results(base_dir=tmp_path / "nonexistent") == []

    def test_lists_results_not_defs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        result = _make_result("exp1")
        save_result(result)
        exp = _make_experiment("exp1")
        save_experiment(exp)
        paths = list_results("exp1", base_dir=tmp_path)
        names = [p.name for p in paths]
        assert all("_def" not in n for n in names)
        assert len(paths) == 1

    def test_filter_by_experiment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        save_result(_make_result("exp_a"))
        save_result(_make_result("exp_b"))
        a_results = list_results("exp_a", base_dir=tmp_path)
        assert len(a_results) == 1
        assert "exp_a" in a_results[0].name

    def test_list_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("prompt_eval.store._DEFAULT_BASE", tmp_path)
        save_result(_make_result("exp_a"))
        save_result(_make_result("exp_b"))
        all_results = list_results(base_dir=tmp_path)
        assert len(all_results) == 2
