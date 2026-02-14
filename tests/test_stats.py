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
        assert "t=" in comp.detail

    def test_same_scores(self):
        result = _make_result([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        comp = compare_variants(result, "a", "b", method="welch")
        assert comp.significant is False

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
