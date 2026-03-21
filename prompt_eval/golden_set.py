"""Growing acceptable set evaluator for multi-correct-answer scenarios.

Wraps any evaluator with a persistent alternatives store that caches LLM
judge decisions. On a primary evaluator miss, the store is checked before
invoking the (expensive) fallback judge. Accepted alternatives are reused
across runs, amortizing judge cost to near-zero over time.

Implements ADR-0004: growing acceptable set evaluator.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from prompt_eval.evaluators import EvalScore

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".prompt_eval"
_DEFAULT_DB_NAME = "golden_sets.db"
AlternativeStatus: TypeAlias = Literal["accepted", "rejected"]
PrimaryEvaluatorResult: TypeAlias = float | EvalScore
PrimaryEvaluator: TypeAlias = Callable[
    [str, str | None],
    PrimaryEvaluatorResult | Awaitable[PrimaryEvaluatorResult],
]
JudgeEvaluator: TypeAlias = Callable[
    [str, str | None],
    "JudgeDecision | Mapping[str, object] | Awaitable[JudgeDecision | Mapping[str, object]]",
]


@dataclass(frozen=True)
class AlternativeRecord:
    """One entry in the acceptable-alternatives store."""

    reference_value: str
    alternative_value: str
    status: str
    judge_reasoning: str
    judge_model: str
    dataset: str
    dimension: str
    created_at: str
    experiment_context: str


class JudgeDecision(BaseModel):
    """Typed result of reviewing one alternative against one reference.

    The acceptable-set cache only stores explicit accept/reject decisions. The
    decision must include the judge model and a short reasoning string so the
    stored record remains auditable.
    """

    model_config = ConfigDict(extra="forbid")

    reasonable: bool
    reasoning: str
    judge_model: str


class GoldenSetManager:
    """Wraps an evaluator with a growing acceptable-alternatives store.

    On a primary evaluator miss (score below *acceptance_threshold*), the
    manager checks a persistent SQLite store for a cached judge decision.
    If the alternative has not been seen before, the fallback judge is
    invoked and the result is persisted.
    """

    def __init__(
        self,
        *,
        primary_evaluator: PrimaryEvaluator,
        fallback_judge: JudgeEvaluator,
        db_path: Path | None = None,
        dataset: str = "default",
        dimension: str = "default",
        acceptance_threshold: float = 0.5,
    ) -> None:
        """Initialize the golden set manager with evaluator and store."""
        self._primary = primary_evaluator
        self._judge = fallback_judge
        self._dataset = dataset
        self._dimension = dimension
        self._threshold = acceptance_threshold

        resolved_path = db_path or (_DEFAULT_DB_DIR / _DEFAULT_DB_NAME)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

        self._hits = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._judge_accepted = 0
        self._judge_rejected = 0

    def _ensure_schema(self) -> None:
        """Create the alternatives table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS acceptable_alternatives (
                dataset TEXT NOT NULL,
                dimension TEXT NOT NULL,
                reference_value TEXT NOT NULL,
                alternative_value TEXT NOT NULL,
                status TEXT NOT NULL,
                judge_reasoning TEXT NOT NULL DEFAULT '',
                judge_model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                experiment_context TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (dataset, dimension, reference_value, alternative_value)
            )
        """)
        self._conn.commit()

    def evaluate(
        self,
        output: str,
        expected: str | None,
        *,
        experiment_context: str = "",
    ) -> EvalScore:
        """Score *output* against *expected* with cache-before-judge logic.

        This synchronous entrypoint is intentionally narrow. It supports normal
        synchronous primary evaluators and synchronous judges. If either
        callable is asynchronous, use `aevaluate()` or `build_evaluator()`
        instead so the async contract remains explicit.
        """
        reference = _require_string_expected(expected)
        candidate = _require_string_output(output)
        primary_result = self._primary(output, expected)
        if inspect.isawaitable(primary_result):
            raise TypeError(
                "GoldenSetManager.evaluate() does not support async primary "
                "evaluators. Use aevaluate() or build_evaluator() instead."
            )
        primary_score = _extract_score(primary_result)
        if primary_score >= self._threshold:
            self._hits += 1
            logger.debug(
                "acceptable-set primary hit",
                extra={"dataset": self._dataset, "dimension": self._dimension},
            )
            return _to_eval_score(primary_result, source="primary")

        cached = self._lookup(reference, candidate)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                "acceptable-set cache hit",
                extra={
                    "dataset": self._dataset,
                    "dimension": self._dimension,
                    "reference_value": reference,
                    "alternative_value": candidate,
                    "status": cached.status,
                },
            )
            if cached.status == "accepted":
                return EvalScore(
                    score=self._threshold,
                    reasoning=f"Cached accepted alternative: {cached.judge_reasoning}",
                )
            return EvalScore(score=0.0, reasoning=f"Cached rejected: {cached.judge_reasoning}")

        self._cache_misses += 1
        judge_result = self._judge(output, expected)
        if inspect.isawaitable(judge_result):
            raise TypeError(
                "GoldenSetManager.evaluate() does not support async judges. "
                "Use aevaluate() or build_evaluator() instead."
            )
        decision = _coerce_judge_decision(judge_result)
        return self._record_judge_decision(
            reference=reference,
            alternative=candidate,
            decision=decision,
            experiment_context=experiment_context,
        )

    async def aevaluate(
        self,
        output: str,
        expected: str | None,
        *,
        experiment_context: str = "",
    ) -> EvalScore:
        """Async-compatible acceptable-set evaluation entrypoint.

        This is the primary integration path for `prompt_eval.run_experiment()`
        because it supports both synchronous and asynchronous primary
        evaluators and judge callables.
        """
        reference = _require_string_expected(expected)
        candidate = _require_string_output(output)
        primary_result = self._primary(output, expected)
        if inspect.isawaitable(primary_result):
            primary_result = await primary_result
        primary_score = _extract_score(primary_result)
        if primary_score >= self._threshold:
            self._hits += 1
            logger.debug(
                "acceptable-set primary hit",
                extra={"dataset": self._dataset, "dimension": self._dimension},
            )
            return _to_eval_score(primary_result, source="primary")

        cached = self._lookup(reference, candidate)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                "acceptable-set cache hit",
                extra={
                    "dataset": self._dataset,
                    "dimension": self._dimension,
                    "reference_value": reference,
                    "alternative_value": candidate,
                    "status": cached.status,
                },
            )
            if cached.status == "accepted":
                return EvalScore(
                    score=self._threshold,
                    reasoning=f"Cached accepted alternative: {cached.judge_reasoning}",
                )
            return EvalScore(score=0.0, reasoning=f"Cached rejected: {cached.judge_reasoning}")

        self._cache_misses += 1
        judge_result = self._judge(output, expected)
        if inspect.isawaitable(judge_result):
            judge_result = await judge_result
        decision = _coerce_judge_decision(judge_result)
        return self._record_judge_decision(
            reference=reference,
            alternative=candidate,
            decision=decision,
            experiment_context=experiment_context,
        )

    def build_evaluator(
        self,
        *,
        experiment_context: str = "",
    ) -> Callable[[str, str | None], Awaitable[EvalScore]]:
        """Build an async evaluator suitable for `prompt_eval.run_experiment()`.

        The returned async callable closes over this manager instance and keeps a
        fixed `experiment_context` for persisted acceptable-set decisions.
        """

        async def evaluate(output: str, expected: str | None) -> EvalScore:
            """Evaluate one candidate/reference pair through the acceptable-set cache."""

            return await self.aevaluate(
                output,
                expected,
                experiment_context=experiment_context,
            )

        return evaluate

    def _record_judge_decision(
        self,
        *,
        reference: str,
        alternative: str,
        decision: JudgeDecision,
        experiment_context: str,
    ) -> EvalScore:
        """Persist one validated judge decision and return the evaluation result."""
        status: AlternativeStatus = "accepted" if decision.reasonable else "rejected"
        self._persist(
            reference=reference,
            alternative=alternative,
            status=status,
            reasoning=decision.reasoning,
            judge_model=decision.judge_model,
            experiment_context=experiment_context,
        )

        if decision.reasonable:
            self._judge_accepted += 1
            logger.info(
                "acceptable-set judge accepted alternative",
                extra={
                    "dataset": self._dataset,
                    "dimension": self._dimension,
                    "reference_value": reference,
                    "alternative_value": alternative,
                    "judge_model": decision.judge_model,
                },
            )
            return EvalScore(
                score=self._threshold,
                reasoning=f"Judge accepted: {decision.reasoning}",
            )
        self._judge_rejected += 1
        logger.info(
            "acceptable-set judge rejected alternative",
            extra={
                "dataset": self._dataset,
                "dimension": self._dimension,
                "reference_value": reference,
                "alternative_value": alternative,
                "judge_model": decision.judge_model,
            },
        )
        return EvalScore(score=0.0, reasoning=f"Judge rejected: {decision.reasoning}")

    def get_alternatives(self, reference: str) -> list[AlternativeRecord]:
        """Return all stored alternatives for a reference value."""
        cursor = self._conn.execute(
            """SELECT * FROM acceptable_alternatives
               WHERE dataset = ? AND dimension = ? AND reference_value = ?
               ORDER BY created_at""",
            (self._dataset, self._dimension, reference),
        )
        return [_row_to_record(row) for row in cursor.fetchall()]

    def override(self, reference: str, alternative: str, status: AlternativeStatus) -> None:
        """Manually override a stored judge decision."""
        if status not in ("accepted", "rejected"):
            msg = f"status must be 'accepted' or 'rejected', got '{status}'"
            raise ValueError(msg)
        cursor = self._conn.execute(
            """UPDATE acceptable_alternatives SET status = ?
               WHERE dataset = ? AND dimension = ? AND reference_value = ? AND alternative_value = ?""",
            (status, self._dataset, self._dimension, reference, alternative),
        )
        if cursor.rowcount == 0:
            raise KeyError(
                "No acceptable-set record exists for "
                f"reference={reference!r}, alternative={alternative!r}, "
                f"dataset={self._dataset!r}, dimension={self._dimension!r}."
            )
        self._conn.commit()
        logger.info(
            "acceptable-set override applied",
            extra={
                "dataset": self._dataset,
                "dimension": self._dimension,
                "reference_value": reference,
                "alternative_value": alternative,
                "status": status,
            },
        )

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss/accepted/rejected counts for this session."""
        return {
            "primary_hits": self._hits,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "judge_accepted": self._judge_accepted,
            "judge_rejected": self._judge_rejected,
        }

    def _lookup(self, reference: str, alternative: str) -> AlternativeRecord | None:
        """Check the store for a cached decision."""
        cursor = self._conn.execute(
            """SELECT * FROM acceptable_alternatives
               WHERE dataset = ? AND dimension = ? AND reference_value = ? AND alternative_value = ?""",
            (self._dataset, self._dimension, reference, alternative),
        )
        row = cursor.fetchone()
        return _row_to_record(row) if row else None

    def _persist(
        self,
        *,
        reference: str,
        alternative: str,
        status: AlternativeStatus,
        reasoning: str,
        judge_model: str,
        experiment_context: str,
    ) -> None:
        """Insert or update an alternative in the store."""
        now = datetime.now(tz=timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO acceptable_alternatives
               (dataset, dimension, reference_value, alternative_value, status,
                judge_reasoning, judge_model, created_at, experiment_context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self._dataset, self._dimension, reference, alternative,
             status, reasoning, judge_model, now, experiment_context),
        )
        self._conn.commit()
        logger.info(
            "acceptable-set decision persisted",
            extra={
                "dataset": self._dataset,
                "dimension": self._dimension,
                "reference_value": reference,
                "alternative_value": alternative,
                "status": status,
                "judge_model": judge_model,
            },
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> GoldenSetManager:
        """Enter context manager, returning self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager, closing the database connection."""
        self.close()


def _row_to_record(row: sqlite3.Row) -> AlternativeRecord:
    """Convert a database row to an AlternativeRecord."""
    return AlternativeRecord(
        reference_value=row["reference_value"],
        alternative_value=row["alternative_value"],
        status=row["status"],
        judge_reasoning=row["judge_reasoning"],
        judge_model=row["judge_model"],
        dataset=row["dataset"],
        dimension=row["dimension"],
        created_at=row["created_at"],
        experiment_context=row["experiment_context"],
    )


def _extract_score(result: Any) -> float:
    """Extract a float score from an evaluator result."""
    if isinstance(result, EvalScore):
        return result.score
    if isinstance(result, (int, float)):
        return float(result)
    return 0.0


def _coerce_judge_decision(raw_decision: JudgeDecision | Mapping[str, object]) -> JudgeDecision:
    """Validate and normalize one fallback-judge decision."""
    if isinstance(raw_decision, JudgeDecision):
        return raw_decision
    try:
        return JudgeDecision.model_validate(raw_decision)
    except ValidationError as exc:
        raise ValueError(
            "Fallback judge returned an invalid JudgeDecision payload."
        ) from exc


def _require_string_output(output: str) -> str:
    """Validate the v1 acceptable-set candidate contract."""
    if not isinstance(output, str):
        raise TypeError(
            "Growing acceptable-set evaluation currently supports only string "
            f"outputs, got {type(output).__name__}."
        )
    return output


def _require_string_expected(expected: str | None) -> str:
    """Validate the v1 acceptable-set reference contract."""
    if expected is None:
        raise ValueError(
            "Growing acceptable-set evaluation requires a non-null expected reference."
        )
    if not isinstance(expected, str):
        raise TypeError(
            "Growing acceptable-set evaluation currently supports only string "
            f"expected references, got {type(expected).__name__}."
        )
    return expected


def _to_eval_score(result: Any, *, source: str) -> EvalScore:
    """Convert an evaluator result to an EvalScore."""
    if isinstance(result, EvalScore):
        return result
    score = _extract_score(result)
    return EvalScore(score=score, reasoning=f"Primary evaluator ({source})")
