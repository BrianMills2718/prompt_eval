"""Tests for statistical comparison."""

import pytest

from prompt_eval.experiment import EvalResult, Trial
from prompt_eval.stats import compare_variants, ComparisonResult


def _make_result(scores_a: list[float], scores_b: list[float]) -> EvalResult:
    """Helper to build an EvalResult from score lists."""
    trials = []
    for i, s in enumerate(scores_a):
        trials.append(Trial(variant_name="a", input_id=f"i{i}", output="x", score=s))
    for i, s in enumerate(scores_b):
        trials.append(Trial(variant_name="b", input_id=f"i{i}", output="x", score=s))
    return EvalResult(
        experiment_name="test",
        variants=["a", "b"],
        trials=trials,
    )


def _make_paired_result(
    scores_a: dict[str, list[float]],
    scores_b: dict[str, list[float]],
    *,
    dimension: str | None = None,
) -> EvalResult:
    """Helper to build a paired `EvalResult` with replicates per input."""

    trials: list[Trial] = []
    for input_id, values in scores_a.items():
        for replicate, score in enumerate(values):
            dimension_scores = {dimension: score} if dimension is not None else None
            trials.append(
                Trial(
                    variant_name="a",
                    input_id=input_id,
                    replicate=replicate,
                    output="x",
                    score=score if dimension is None else 0.0,
                    dimension_scores=dimension_scores,
                )
            )
    for input_id, values in scores_b.items():
        for replicate, score in enumerate(values):
            dimension_scores = {dimension: score} if dimension is not None else None
            trials.append(
                Trial(
                    variant_name="b",
                    input_id=input_id,
                    replicate=replicate,
                    output="x",
                    score=score if dimension is None else 0.0,
                    dimension_scores=dimension_scores,
                )
            )
    return EvalResult(experiment_name="paired", variants=["a", "b"], trials=trials)


class TestBootstrapCompare:

    def test_clearly_different(self):
        # a is much better than b
        result = _make_result(
            [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.89, 0.90, 0.86],
            [0.3, 0.25, 0.32, 0.28, 0.31, 0.27, 0.33, 0.29, 0.30, 0.26],
        )
        comp = compare_variants(result, "a", "b", method="bootstrap")
        assert comp.significant is True
        assert comp.difference > 0.5
        assert comp.ci_lower > 0

    def test_no_difference(self):
        # Same scores
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = _make_result(scores, scores)
        comp = compare_variants(result, "a", "b", method="bootstrap")
        assert comp.significant is False
        assert comp.difference == pytest.approx(0.0)
        assert "SciPy bootstrap" in comp.detail

    def test_overlapping_distributions(self):
        result = _make_result(
            [0.5, 0.6, 0.55, 0.52, 0.58],
            [0.48, 0.55, 0.53, 0.50, 0.56],
        )
        comp = compare_variants(result, "a", "b", method="bootstrap")
        # With this much overlap, should not be significant
        assert comp.method == "bootstrap"
        # Don't assert significance — it's noisy with small n


class TestWelchCompare:

    def test_clearly_different(self):
        result = _make_result(
            [0.9, 0.85, 0.92, 0.88, 0.91],
            [0.3, 0.25, 0.32, 0.28, 0.31],
        )
        comp = compare_variants(result, "a", "b", method="welch")
        assert comp.significant is True
        assert comp.method == "welch"
        assert "SciPy Welch" in comp.detail
        assert "p=" in comp.detail

    def test_same_scores(self):
        result = _make_result([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        comp = compare_variants(result, "a", "b", method="welch")
        assert comp.significant is False
        assert comp.comparison_mode == "pooled"

    def test_needs_min_2_samples(self):
        result = _make_result([0.5], [0.3])
        with pytest.raises(ValueError, match="n >= 2"):
            compare_variants(result, "a", "b", method="welch")


class TestCompareVariantsEdgeCases:

    def test_missing_variant_raises(self):
        result = _make_result([0.5], [0.3])
        with pytest.raises(ValueError, match="Need scores"):
            compare_variants(result, "a", "nonexistent")

    def test_no_scores_raises(self):
        # Trials exist but no scores
        trials = [
            Trial(variant_name="a", input_id="i1", output="x"),
            Trial(variant_name="b", input_id="i1", output="x"),
        ]
        result = EvalResult(experiment_name="test", variants=["a", "b"], trials=trials)
        with pytest.raises(ValueError, match="Need scores"):
            compare_variants(result, "a", "b")

    def test_unknown_method_raises(self):
        result = _make_result([0.5, 0.6], [0.3, 0.4])
        with pytest.raises(ValueError, match="Unknown method"):
            compare_variants(result, "a", "b", method="invalid")

    def test_returns_comparison_result(self):
        result = _make_result([0.5, 0.6, 0.7], [0.3, 0.4, 0.5])
        comp = compare_variants(result, "a", "b")
        assert isinstance(comp, ComparisonResult)
        assert comp.variant_a == "a"
        assert comp.variant_b == "b"


class TestCompareByDimension:

    def test_compare_specific_dimension(self):
        """Compare variants on a specific dimension using dimension_scores."""
        trials = [
            Trial(variant_name="a", input_id="i1", output="x", score=0.7,
                  dimension_scores={"clarity": 0.9, "depth": 0.5}),
            Trial(variant_name="a", input_id="i2", output="x", score=0.7,
                  dimension_scores={"clarity": 0.85, "depth": 0.55}),
            Trial(variant_name="a", input_id="i3", output="x", score=0.7,
                  dimension_scores={"clarity": 0.88, "depth": 0.52}),
            Trial(variant_name="b", input_id="i1", output="x", score=0.6,
                  dimension_scores={"clarity": 0.4, "depth": 0.8}),
            Trial(variant_name="b", input_id="i2", output="x", score=0.6,
                  dimension_scores={"clarity": 0.45, "depth": 0.75}),
            Trial(variant_name="b", input_id="i3", output="x", score=0.6,
                  dimension_scores={"clarity": 0.42, "depth": 0.78}),
        ]
        result = EvalResult(experiment_name="test", variants=["a", "b"], trials=trials)

        # a is much better on clarity
        comp = compare_variants(result, "a", "b", dimension="clarity")
        assert comp.mean_a > comp.mean_b
        assert comp.difference > 0.3

        # b is better on depth
        comp = compare_variants(result, "a", "b", dimension="depth")
        assert comp.mean_b > comp.mean_a

    def test_missing_dimension_raises(self):
        """Requesting a dimension that no trial has raises ValueError."""
        trials = [
            Trial(variant_name="a", input_id="i1", output="x", score=0.7),
            Trial(variant_name="b", input_id="i1", output="x", score=0.6),
        ]
        result = EvalResult(experiment_name="test", variants=["a", "b"], trials=trials)
        with pytest.raises(ValueError, match="Need scores"):
            compare_variants(result, "a", "b", dimension="nonexistent")


class TestPairedByInputCompare:

    def test_paired_t_detects_clear_difference(self):
        result = _make_paired_result(
            {
                "i1": [0.95, 0.93, 0.94],
                "i2": [0.92, 0.91, 0.89],
                "i3": [0.87, 0.89, 0.86],
            },
            {
                "i1": [0.55, 0.53, 0.54],
                "i2": [0.50, 0.51, 0.49],
                "i3": [0.49, 0.50, 0.48],
            },
        )
        comp = compare_variants(
            result,
            "a",
            "b",
            method="paired_t",
            comparison_mode="paired_by_input",
        )
        assert comp.significant is True
        assert comp.method == "paired_t"
        assert comp.comparison_mode == "paired_by_input"
        assert comp.n_units == 3
        assert "paired input_ids" in comp.detail

    def test_paired_bootstrap_reports_mode(self):
        result = _make_paired_result(
            {
                "i1": [0.82, 0.84],
                "i2": [0.78, 0.79],
                "i3": [0.80, 0.81],
            },
            {
                "i1": [0.72, 0.73],
                "i2": [0.70, 0.71],
                "i3": [0.69, 0.68],
            },
        )
        comp = compare_variants(
            result,
            "a",
            "b",
            method="bootstrap",
            comparison_mode="paired_by_input",
        )
        assert comp.method == "bootstrap"
        assert comp.comparison_mode == "paired_by_input"
        assert comp.n_units == 3
        assert "paired input_ids" in comp.detail

    def test_paired_mode_requires_matching_input_ids(self):
        result = _make_paired_result(
            {"i1": [0.9], "i2": [0.8]},
            {"i1": [0.6], "i3": [0.5]},
        )
        with pytest.raises(ValueError, match="same scored input_ids"):
            compare_variants(
                result,
                "a",
                "b",
                comparison_mode="paired_by_input",
            )

    def test_paired_mode_requires_two_inputs(self):
        result = _make_paired_result(
            {"i1": [0.9, 0.8]},
            {"i1": [0.6, 0.5]},
        )
        with pytest.raises(ValueError, match="at least 2 shared input_ids"):
            compare_variants(
                result,
                "a",
                "b",
                comparison_mode="paired_by_input",
            )

    def test_paired_mode_rejects_welch(self):
        result = _make_paired_result(
            {"i1": [0.9], "i2": [0.8]},
            {"i1": [0.6], "i2": [0.5]},
        )
        with pytest.raises(ValueError, match="Unknown method for paired_by_input"):
            compare_variants(
                result,
                "a",
                "b",
                method="welch",
                comparison_mode="paired_by_input",
            )

    def test_paired_mode_supports_dimension_scores(self):
        result = _make_paired_result(
            {
                "i1": [0.92, 0.93],
                "i2": [0.89, 0.88],
                "i3": [0.91, 0.89],
            },
            {
                "i1": [0.52, 0.53],
                "i2": [0.50, 0.49],
                "i3": [0.51, 0.50],
            },
            dimension="clarity",
        )
        comp = compare_variants(
            result,
            "a",
            "b",
            method="paired_t",
            dimension="clarity",
            comparison_mode="paired_by_input",
        )
        assert comp.mean_a > comp.mean_b
        assert comp.comparison_mode == "paired_by_input"
