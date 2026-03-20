"""Statistical comparison of prompt variants.

The public helper keeps backward-compatible pooled comparison as the default and
also supports an explicit paired-by-input mode for the common
`variant x input_id x replicate` experiment shape.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

from prompt_eval.experiment import EvalResult, Trial

ComparisonMethod = Literal["bootstrap", "welch", "paired_t"]
ComparisonMode = Literal["pooled", "paired_by_input"]


@dataclass
class ComparisonResult:
    """Result of comparing two variants.

    `comparison_mode` and `n_units` make the unit of analysis explicit. In
    pooled mode, `n_units` is the number of scored trials per variant. In
    paired-by-input mode, `n_units` is the number of matched `input_id`s.
    """

    variant_a: str
    variant_b: str
    mean_a: float
    mean_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    significant: bool
    method: ComparisonMethod
    comparison_mode: ComparisonMode
    n_units: int
    detail: str


def compare_variants(
    result: EvalResult,
    variant_a: str,
    variant_b: str,
    confidence: float = 0.95,
    method: ComparisonMethod = "bootstrap",
    dimension: str | None = None,
    comparison_mode: ComparisonMode = "pooled",
) -> ComparisonResult:
    """Compare two variants from an experiment result.

    Args:
        result: Completed experiment result.
        variant_a: Name of first variant.
        variant_b: Name of second variant.
        confidence: Confidence level for CI (default 0.95).
        method: pooled mode supports `"bootstrap"` (default) or `"welch"`;
            paired-by-input mode supports `"bootstrap"` or `"paired_t"`.
        dimension: Optional dimension name to compare on (uses dimension_scores).
        comparison_mode: `"pooled"` (default) compares scored trials directly.
            `"paired_by_input"` aggregates replicates to per-input means and
            compares matched `input_id`s.

    Returns:
        ComparisonResult with means, difference, CI, and significance.
    """
    prepared = _prepare_scores(
        result=result,
        variant_a=variant_a,
        variant_b=variant_b,
        dimension=dimension,
        comparison_mode=comparison_mode,
    )

    if comparison_mode == "pooled":
        if method == "bootstrap":
            return _bootstrap_compare(
                variant_a,
                variant_b,
                prepared.scores_a,
                prepared.scores_b,
                prepared.mean_a,
                prepared.mean_b,
                prepared.difference,
                confidence,
                comparison_mode=comparison_mode,
                n_units=prepared.n_units,
                detail_suffix=f"across {prepared.n_units} scored trials per variant",
                paired=False,
            )
        if method == "welch":
            return _welch_compare(
                variant_a,
                variant_b,
                prepared.scores_a,
                prepared.scores_b,
                prepared.mean_a,
                prepared.mean_b,
                prepared.difference,
                confidence,
                comparison_mode=comparison_mode,
                n_units=prepared.n_units,
            )
        raise ValueError(
            "Unknown method for pooled comparison: "
            f"{method}. Use 'bootstrap' or 'welch'."
        )

    if method == "bootstrap":
        return _bootstrap_compare(
            variant_a,
            variant_b,
            prepared.scores_a,
            prepared.scores_b,
            prepared.mean_a,
            prepared.mean_b,
            prepared.difference,
            confidence,
            comparison_mode=comparison_mode,
            n_units=prepared.n_units,
            detail_suffix=f"across {prepared.n_units} paired input_ids",
            paired=True,
        )
    if method == "paired_t":
        return _paired_t_compare(
            variant_a,
            variant_b,
            prepared.scores_a,
            prepared.scores_b,
            prepared.mean_a,
            prepared.mean_b,
            prepared.difference,
            confidence,
            comparison_mode=comparison_mode,
            n_units=prepared.n_units,
        )
    raise ValueError(
        "Unknown method for paired_by_input comparison: "
        f"{method}. Use 'bootstrap' or 'paired_t'."
    )


@dataclass(frozen=True)
class _PreparedComparison:
    """Normalized scores and summary stats for one comparison request."""

    scores_a: list[float]
    scores_b: list[float]
    mean_a: float
    mean_b: float
    difference: float
    n_units: int


def _prepare_scores(
    *,
    result: EvalResult,
    variant_a: str,
    variant_b: str,
    dimension: str | None,
    comparison_mode: ComparisonMode,
) -> _PreparedComparison:
    """Collect scores and summary statistics for one comparison mode."""

    if comparison_mode == "pooled":
        scores_a = _collect_pooled_scores(result, variant_a, dimension)
        scores_b = _collect_pooled_scores(result, variant_b, dimension)
    elif comparison_mode == "paired_by_input":
        scores_a, scores_b = _collect_paired_input_scores(
            result=result,
            variant_a=variant_a,
            variant_b=variant_b,
            dimension=dimension,
        )
    else:
        raise ValueError(
            f"Unknown comparison_mode: {comparison_mode}. "
            "Use 'pooled' or 'paired_by_input'."
        )

    if not scores_a or not scores_b:
        raise ValueError(
            f"Need scores for both variants. Got {len(scores_a)} for '{variant_a}', "
            f"{len(scores_b)} for '{variant_b}'."
        )

    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)
    return _PreparedComparison(
        scores_a=scores_a,
        scores_b=scores_b,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=mean_a - mean_b,
        n_units=len(scores_a),
    )


def _collect_pooled_scores(
    result: EvalResult,
    variant_name: str,
    dimension: str | None,
) -> list[float]:
    """Collect scored trials directly for pooled comparison."""

    collected: list[float] = []
    for trial in result.trials:
        if trial.variant_name != variant_name:
            continue
        score = _trial_score(trial, dimension)
        if score is not None:
            collected.append(score)
    return collected


def _collect_paired_input_scores(
    *,
    result: EvalResult,
    variant_a: str,
    variant_b: str,
    dimension: str | None,
) -> tuple[list[float], list[float]]:
    """Collect matched per-input means for paired comparison.

    The comparison fails loudly if the two variants do not have scored trials
    for the same set of `input_id`s. That avoids silently dropping mismatched
    inputs and misrepresenting the unit of analysis.
    """

    scores_by_input_a = _scores_by_input(result, variant_a, dimension)
    scores_by_input_b = _scores_by_input(result, variant_b, dimension)

    input_ids_a = set(scores_by_input_a)
    input_ids_b = set(scores_by_input_b)
    if input_ids_a != input_ids_b:
        missing_for_a = sorted(input_ids_b - input_ids_a)
        missing_for_b = sorted(input_ids_a - input_ids_b)
        raise ValueError(
            "paired_by_input comparison requires the same scored input_ids for "
            f"both variants. Missing for '{variant_a}': {missing_for_a}; "
            f"missing for '{variant_b}': {missing_for_b}."
        )

    ordered_input_ids = sorted(input_ids_a)
    if len(ordered_input_ids) < 2:
        raise ValueError(
            "paired_by_input comparison needs scores for at least 2 shared "
            f"input_ids. Got {len(ordered_input_ids)}."
        )

    scores_a = [statistics.mean(scores_by_input_a[input_id]) for input_id in ordered_input_ids]
    scores_b = [statistics.mean(scores_by_input_b[input_id]) for input_id in ordered_input_ids]
    return scores_a, scores_b


def _scores_by_input(
    result: EvalResult,
    variant_name: str,
    dimension: str | None,
) -> dict[str, list[float]]:
    """Group scored trials for one variant by input identifier."""

    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for trial in result.trials:
        if trial.variant_name != variant_name:
            continue
        score = _trial_score(trial, dimension)
        if score is not None:
            grouped[trial.input_id].append(score)
    return dict(grouped)


def _trial_score(trial: Trial, dimension: str | None) -> float | None:
    """Extract one comparable score from a trial."""

    if dimension is not None:
        dimension_scores = getattr(trial, "dimension_scores", None)
        if dimension_scores is None or dimension not in dimension_scores:
            return None
        return float(dimension_scores[dimension])

    score = getattr(trial, "score", None)
    if score is None:
        return None
    return float(score)


def _bootstrap_compare(
    name_a: str,
    name_b: str,
    scores_a: list[float],
    scores_b: list[float],
    mean_a: float,
    mean_b: float,
    diff: float,
    confidence: float,
    *,
    comparison_mode: ComparisonMode,
    n_units: int,
    detail_suffix: str,
    paired: bool,
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
        paired=paired,
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
        comparison_mode=comparison_mode,
        n_units=n_units,
        detail=(
            f"SciPy bootstrap CI ({confidence:.0%}) {detail_suffix}: "
            f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        ),
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
    *,
    comparison_mode: ComparisonMode,
    n_units: int,
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
            method="welch",
            comparison_mode=comparison_mode,
            n_units=n_units,
            detail="Zero variance in both pooled groups",
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
        comparison_mode=comparison_mode,
        n_units=n_units,
        detail=(
            f"SciPy Welch t={t_stat:.3f}, df={df:.1f}, p={p_value:.4g}, "
            f"CI ({confidence:.0%}) across {n_units} pooled trials per variant: "
            f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        ),
    )


def _paired_t_compare(
    name_a: str,
    name_b: str,
    scores_a: list[float],
    scores_b: list[float],
    mean_a: float,
    mean_b: float,
    diff: float,
    confidence: float,
    *,
    comparison_mode: ComparisonMode,
    n_units: int,
) -> ComparisonResult:
    """SciPy-backed paired comparison over matched per-input means."""

    if len(scores_a) != len(scores_b):
        raise ValueError(
            "paired_t comparison requires matched inputs with equal counts. "
            f"Got {len(scores_a)} and {len(scores_b)}."
        )

    paired_differences = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if len(paired_differences) < 2:
        raise ValueError(
            f"paired_t comparison needs n >= 2 paired inputs. Got {len(paired_differences)}."
        )

    variance = float(np.var(paired_differences, ddof=1))
    if variance == 0:
        return ComparisonResult(
            variant_a=name_a,
            variant_b=name_b,
            mean_a=mean_a,
            mean_b=mean_b,
            difference=diff,
            ci_lower=diff,
            ci_upper=diff,
            significant=diff != 0,
            method="paired_t",
            comparison_mode=comparison_mode,
            n_units=n_units,
            detail="Zero variance in paired input differences",
        )

    paired_result = scipy_stats.ttest_rel(
        np.asarray(scores_a, dtype=float),
        np.asarray(scores_b, dtype=float),
    )
    t_stat = float(paired_result.statistic)
    p_value = float(paired_result.pvalue)
    se = float(np.sqrt(variance / len(paired_differences)))
    alpha = 1 - confidence
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, len(paired_differences) - 1))
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
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
        method="paired_t",
        comparison_mode=comparison_mode,
        n_units=n_units,
        detail=(
            f"SciPy paired t={t_stat:.3f}, df={len(paired_differences) - 1}, "
            f"p={p_value:.4g}, CI ({confidence:.0%}) across {n_units} paired "
            f"input_ids: [{ci_lower:.4f}, {ci_upper:.4f}]"
        ),
    )


def _mean_difference_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Return the mean difference used by the bootstrap comparison."""

    return float(np.mean(sample_a) - np.mean(sample_b))
