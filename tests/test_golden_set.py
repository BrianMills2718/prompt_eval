"""Tests for the growing acceptable set evaluator (ADR-0004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_eval.evaluators import EvalScore
from prompt_eval.golden_set import GoldenSetManager, JudgeDecision


def _exact_evaluator(output: str, expected: str | None = None) -> float:
    """Simple exact-match evaluator for testing."""
    return 1.0 if output == expected else 0.0


def _always_reasonable_judge(output: str, expected: str | None = None) -> dict:
    """Judge that always says reasonable."""
    return {"reasonable": True, "reasoning": "Looks good", "judge_model": "test-judge"}


def _always_unreasonable_judge(output: str, expected: str | None = None) -> dict:
    """Judge that always says unreasonable."""
    return {"reasonable": False, "reasoning": "Not valid", "judge_model": "test-judge"}


def _counting_judge():
    """Return a judge that counts invocations."""
    calls = {"count": 0}

    def judge(output: str, expected: str | None = None) -> dict:
        calls["count"] += 1
        return {"reasonable": True, "reasoning": f"Call {calls['count']}", "judge_model": "test"}

    return judge, calls


def _invalid_judge_payload(output: str, expected: str | None = None) -> dict:
    """Return a malformed judge payload for fail-loud validation tests."""
    return {"reasonable": True, "reasoning": "Missing judge model"}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test_golden.db"


def test_primary_hit_skips_judge(db_path: Path) -> None:
    """When primary evaluator scores above threshold, judge is never called."""
    judge, calls = _counting_judge()
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=judge,
        db_path=db_path,
        acceptance_threshold=0.5,
    )
    result = gsm.evaluate("hello", "hello")
    assert result.score == 1.0
    assert calls["count"] == 0
    assert gsm.stats()["primary_hits"] == 1


def test_cache_miss_invokes_judge_and_persists(db_path: Path) -> None:
    """Unknown alternative triggers judge and stores the result."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
    )
    result = gsm.evaluate("MilitaryOrg", "MilitaryOrganization")
    assert result.score >= 0.5
    assert gsm.stats()["cache_misses"] == 1
    assert gsm.stats()["judge_accepted"] == 1

    alts = gsm.get_alternatives("MilitaryOrganization")
    assert len(alts) == 1
    assert alts[0].alternative_value == "MilitaryOrg"
    assert alts[0].status == "accepted"


def test_cache_hit_skips_judge(db_path: Path) -> None:
    """Known alternative returns cached score without judge call."""
    judge, calls = _counting_judge()
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=judge,
        db_path=db_path,
    )
    gsm.evaluate("MilitaryOrg", "MilitaryOrganization")
    assert calls["count"] == 1

    result = gsm.evaluate("MilitaryOrg", "MilitaryOrganization")
    assert calls["count"] == 1
    assert result.score >= 0.5
    assert gsm.stats()["cache_hits"] == 1


def test_rejected_alternative_not_rejudged(db_path: Path) -> None:
    """Previously rejected pair returns 0 without re-invoking judge."""
    judge, calls = _counting_judge()

    def rejecting_judge(output: str, expected: str | None = None) -> dict:
        calls["count"] += 1
        return {"reasonable": False, "reasoning": "Bad", "judge_model": "test"}

    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=rejecting_judge,
        db_path=db_path,
    )
    result1 = gsm.evaluate("BadType", "MilitaryOrganization")
    assert result1.score == 0.0
    assert calls["count"] == 1

    result2 = gsm.evaluate("BadType", "MilitaryOrganization")
    assert result2.score == 0.0
    assert calls["count"] == 1


def test_scoping_by_dataset_dimension(db_path: Path) -> None:
    """Different datasets maintain separate stores."""
    gsm_a = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
        dataset="dataset_a",
        dimension="types",
    )
    gsm_b = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_unreasonable_judge,
        db_path=db_path,
        dataset="dataset_b",
        dimension="types",
    )
    gsm_a.evaluate("Alt", "Ref")
    gsm_b.evaluate("Alt", "Ref")

    alts_a = gsm_a.get_alternatives("Ref")
    alts_b = gsm_b.get_alternatives("Ref")
    assert alts_a[0].status == "accepted"
    assert alts_b[0].status == "rejected"


def test_manual_override(db_path: Path) -> None:
    """override() changes stored status."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_unreasonable_judge,
        db_path=db_path,
    )
    gsm.evaluate("Alt", "Ref")
    assert gsm.get_alternatives("Ref")[0].status == "rejected"

    gsm.override("Ref", "Alt", "accepted")
    assert gsm.get_alternatives("Ref")[0].status == "accepted"


def test_stats_counts(db_path: Path) -> None:
    """stats() returns correct counts."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
    )
    gsm.evaluate("exact", "exact")
    gsm.evaluate("alt1", "ref")
    gsm.evaluate("alt1", "ref")
    gsm.evaluate("alt2", "ref")

    s = gsm.stats()
    assert s["primary_hits"] == 1
    assert s["cache_hits"] == 1
    assert s["cache_misses"] == 2
    assert s["judge_accepted"] == 2


def test_eval_score_primary(db_path: Path) -> None:
    """Primary evaluator returning EvalScore is handled correctly."""
    def rich_evaluator(output: str, expected: str | None = None) -> EvalScore:
        return EvalScore(score=1.0, reasoning="exact") if output == expected else EvalScore(score=0.2, reasoning="partial")

    gsm = GoldenSetManager(
        primary_evaluator=rich_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
        acceptance_threshold=0.5,
    )
    result = gsm.evaluate("Alt", "Ref")
    assert result.score >= 0.5

    result2 = gsm.evaluate("Ref", "Ref")
    assert result2.score == 1.0


def test_invalid_primary_result_fails_loudly(db_path: Path) -> None:
    """Unexpected primary evaluator return types raise instead of scoring zero."""

    def invalid_primary(output: str, expected: str | None = None) -> object:
        return object()

    gsm = GoldenSetManager(
        primary_evaluator=invalid_primary,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
    )

    with pytest.raises(TypeError, match="Primary evaluator must return"):
        gsm.evaluate("Alt", "Ref")


def test_override_invalid_status_raises(db_path: Path) -> None:
    """override() rejects invalid status values."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
    )
    override = getattr(gsm, "override")
    with pytest.raises(ValueError, match="status must be"):
        override("Ref", "Alt", "maybe")


@pytest.mark.asyncio
async def test_async_judge_is_supported(db_path: Path) -> None:
    """Async judges work through the async acceptable-set entrypoint."""
    calls = {"count": 0}

    async def async_judge(output: str, expected: str | None = None) -> JudgeDecision:
        calls["count"] += 1
        return JudgeDecision(
            reasonable=True,
            reasoning="Reviewed asynchronously",
            judge_model="async-judge",
        )

    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=async_judge,
        db_path=db_path,
    )

    result = await gsm.aevaluate("MilitaryOrg", "MilitaryOrganization")
    assert result.score >= 0.5
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_malformed_judge_decision_fails_loudly(db_path: Path) -> None:
    """Malformed judge payloads raise instead of silently defaulting."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_invalid_judge_payload,
        db_path=db_path,
    )

    with pytest.raises(ValueError, match="invalid JudgeDecision"):
        await gsm.aevaluate("MilitaryOrg", "MilitaryOrganization")


def test_override_missing_entry_fails_loudly(db_path: Path) -> None:
    """Manual override of a missing record should fail loudly."""
    gsm = GoldenSetManager(
        primary_evaluator=_exact_evaluator,
        fallback_judge=_always_reasonable_judge,
        db_path=db_path,
    )
    with pytest.raises(KeyError, match="No acceptable-set record exists"):
        gsm.override("UnknownRef", "UnknownAlt", "accepted")
