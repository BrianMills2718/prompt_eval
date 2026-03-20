"""Statistical comparison of prompt variants."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from prompt_eval.experiment import EvalResult


@dataclass
class ComparisonResult:
    """Result of comparing two variants."""

    variant_a: str
    variant_b: str
    mean_a: float
    mean_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    significant: bool
    method: str
    detail: str


def compare_variants(
    result: EvalResult,
    variant_a: str,
    variant_b: str,
    confidence: float = 0.95,
    method: str = "bootstrap",
    dimension: str | None = None,
) -> ComparisonResult:
    """Compare two variants from an experiment result.

    Args:
        result: Completed experiment result.
        variant_a: Name of first variant.
        variant_b: Name of second variant.
        confidence: Confidence level for CI (default 0.95).
        method: "bootstrap" (default) or "welch" (Welch's t-test).
        dimension: Optional dimension name to compare on (uses dimension_scores).

    Returns:
        ComparisonResult with means, difference, CI, and significance.
    """
    if dimension is not None:
        scores_a = [
            t.dimension_scores[dimension]
            for t in result.trials
            if t.variant_name == variant_a
            and t.dimension_scores is not None
            and dimension in t.dimension_scores
        ]
        scores_b = [
            t.dimension_scores[dimension]
            for t in result.trials
            if t.variant_name == variant_b
            and t.dimension_scores is not None
            and dimension in t.dimension_scores
        ]
    else:
        scores_a = [
            t.score for t in result.trials
            if t.variant_name == variant_a and t.score is not None
        ]
        scores_b = [
            t.score for t in result.trials
            if t.variant_name == variant_b and t.score is not None
        ]

    if not scores_a or not scores_b:
        raise ValueError(
            f"Need scores for both variants. Got {len(scores_a)} for '{variant_a}', "
            f"{len(scores_b)} for '{variant_b}'."
        )

    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)
    diff = mean_a - mean_b

    if method == "bootstrap":
        return _bootstrap_compare(
            variant_a, variant_b, scores_a, scores_b, mean_a, mean_b, diff, confidence,
        )
    elif method == "welch":
        return _welch_compare(
            variant_a, variant_b, scores_a, scores_b, mean_a, mean_b, diff, confidence,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'welch'.")


def _bootstrap_compare(
    name_a: str,
    name_b: str,
    scores_a: list[float],
    scores_b: list[float],
    mean_a: float,
    mean_b: float,
    diff: float,
    confidence: float,
    n_bootstrap: int = 10_000,
) -> ComparisonResult:
    """SciPy-backed bootstrap confidence interval for the difference in means."""
    bootstrap_result = scipy_stats.bootstrap(
        data=(
            np.asarray(scores_a, dtype=float),
            np.asarray(scores_b, dtype=float),
        ),
        statistic=_mean_difference_statistic,
        confidence_level=confidence,
        n_resamples=n_bootstrap,
        method="percentile",
        paired=False,
        vectorized=False,
        rng=np.random.default_rng(0),
    )
    ci_lower = float(bootstrap_result.confidence_interval.low)
    ci_upper = float(bootstrap_result.confidence_interval.high)

    significant = not (ci_lower <= 0 <= ci_upper)

    return ComparisonResult(
        variant_a=name_a,
        variant_b=name_b,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
        method="bootstrap",
        detail=f"SciPy bootstrap CI ({confidence:.0%}): [{ci_lower:.4f}, {ci_upper:.4f}]",
    )


def _welch_compare(
    name_a: str,
    name_b: str,
    scores_a: list[float],
    scores_b: list[float],
    mean_a: float,
    mean_b: float,
    diff: float,
    confidence: float,
) -> ComparisonResult:
    """SciPy-backed unequal-variance comparison with Welch-Satterthwaite CI."""

    n_a, n_b = len(scores_a), len(scores_b)

    if n_a < 2 or n_b < 2:
        raise ValueError(f"Welch's t-test needs n >= 2 per group. Got {n_a}, {n_b}.")

    var_a = statistics.variance(scores_a)
    var_b = statistics.variance(scores_b)
    se = float(np.sqrt(var_a / n_a + var_b / n_b))

    if se == 0:
        return ComparisonResult(
            variant_a=name_a, variant_b=name_b,
            mean_a=mean_a, mean_b=mean_b, difference=diff,
            ci_lower=diff, ci_upper=diff, significant=diff != 0,
            method="welch", detail="Zero variance in both groups",
        )

    welch_result = scipy_stats.ttest_ind(
        np.asarray(scores_a, dtype=float),
        np.asarray(scores_b, dtype=float),
        equal_var=False,
    )
    t_stat = float(welch_result.statistic)
    p_value = float(welch_result.pvalue)

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1

    alpha = 1 - confidence
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df))
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    significant = not (ci_lower <= 0 <= ci_upper)

    return ComparisonResult(
        variant_a=name_a, variant_b=name_b,
        mean_a=mean_a, mean_b=mean_b, difference=diff,
        ci_lower=ci_lower, ci_upper=ci_upper, significant=significant,
        method="welch",
        detail=(
            f"SciPy Welch t={t_stat:.3f}, df={df:.1f}, p={p_value:.4g}, "
            f"CI ({confidence:.0%}): [{ci_lower:.4f}, {ci_upper:.4f}]"
        ),
    )


def _mean_difference_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Return the mean difference used by the bootstrap comparison."""

    return float(np.mean(sample_a) - np.mean(sample_b))
