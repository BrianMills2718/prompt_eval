"""Growing acceptable set evaluator for multi-correct-answer scenarios.

Wraps any evaluator with a persistent alternatives store that caches LLM
judge decisions. On a primary evaluator miss, the store is checked before
invoking the (expensive) fallback judge. Accepted alternatives are reused
across runs, amortizing judge cost to near-zero over time.

Implements ADR-0004: growing acceptable set evaluator.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from prompt_eval.evaluators import EvalScore

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".prompt_eval"
_DEFAULT_DB_NAME = "golden_sets.db"


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
        primary_evaluator: Callable[..., Any],
        fallback_judge: Callable[..., Any],
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
        expected: str,
        *,
        experiment_context: str = "",
    ) -> EvalScore:
        """Score *output* against *expected* with cache-before-judge logic."""
        primary_result = self._primary(output, expected)
        primary_score = _extract_score(primary_result)
        if primary_score >= self._threshold:
            self._hits += 1
            return _to_eval_score(primary_result, source="primary")

        cached = self._lookup(expected, output)
        if cached is not None:
            self._cache_hits += 1
            if cached.status == "accepted":
                return EvalScore(
                    score=self._threshold,
                    reasoning=f"Cached accepted alternative: {cached.judge_reasoning}",
                )
            return EvalScore(score=0.0, reasoning=f"Cached rejected: {cached.judge_reasoning}")

        self._cache_misses += 1
        judge_result = self._judge(output, expected)
        if not isinstance(judge_result, dict):
            raise TypeError(
                f"Fallback judge must return a dict, got {type(judge_result).__name__}"
            )
        reasonable = judge_result.get("reasonable", False)
        reasoning = judge_result.get("reasoning", "")
        judge_model = judge_result.get("judge_model", "unknown")

        status = "accepted" if reasonable else "rejected"
        self._persist(
            reference=expected,
            alternative=output,
            status=status,
            reasoning=reasoning,
            judge_model=judge_model,
            experiment_context=experiment_context,
        )

        if reasonable:
            self._judge_accepted += 1
            return EvalScore(score=self._threshold, reasoning=f"Judge accepted: {reasoning}")
        self._judge_rejected += 1
        return EvalScore(score=0.0, reasoning=f"Judge rejected: {reasoning}")

    def get_alternatives(self, reference: str) -> list[AlternativeRecord]:
        """Return all stored alternatives for a reference value."""
        cursor = self._conn.execute(
            """SELECT * FROM acceptable_alternatives
               WHERE dataset = ? AND dimension = ? AND reference_value = ?
               ORDER BY created_at""",
            (self._dataset, self._dimension, reference),
        )
        return [_row_to_record(row) for row in cursor.fetchall()]

    def override(self, reference: str, alternative: str, status: str) -> None:
        """Manually override a stored judge decision."""
        if status not in ("accepted", "rejected"):
            msg = f"status must be 'accepted' or 'rejected', got '{status}'"
            raise ValueError(msg)
        self._conn.execute(
            """UPDATE acceptable_alternatives SET status = ?
               WHERE dataset = ? AND dimension = ? AND reference_value = ? AND alternative_value = ?""",
            (status, self._dataset, self._dimension, reference, alternative),
        )
        self._conn.commit()

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
        status: str,
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


def _to_eval_score(result: Any, *, source: str) -> EvalScore:
    """Convert an evaluator result to an EvalScore."""
    if isinstance(result, EvalScore):
        return result
    score = _extract_score(result)
    return EvalScore(score=score, reasoning=f"Primary evaluator ({source})")
