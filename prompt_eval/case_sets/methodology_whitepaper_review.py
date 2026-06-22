"""Frozen case set for quality-optimal methodology review outputs.

This module scores precomputed `llm_client review-artifact` JSON outputs. It
does not execute reviewer models; prompt/model comparison remains a caller
responsibility that feeds frozen outputs into this evaluator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prompt_eval.evaluators import EvalScore
from prompt_eval.experiment import ExperimentInput, PrecomputedOutput

CaseCategory = Literal["known_good", "known_defective", "spurious_trap"]

_CASE_DATA_PATH = Path(__file__).with_name("methodology_whitepaper_review_cases.json")
_BASELINE_OUTPUTS_PATH = Path(__file__).with_name(
    "methodology_whitepaper_review_baseline_outputs.json"
)


class MethodologyReviewExpectation(BaseModel):
    """Expected review behavior for one methodology-whitepaper case."""

    model_config = ConfigDict(extra="forbid")

    required_defect_terms: list[str] = Field(
        default_factory=list,
        description="Terms that must appear in actionable defect or optimum-gap text.",
    )
    required_optimum_gap_terms: list[str] = Field(
        default_factory=list,
        description="Terms that must appear in optimum-gap annotation text.",
    )
    spurious_trap_terms: list[str] = Field(
        default_factory=list,
        description="Terms that must not appear in actionable findings.",
    )
    expected_spurious_terms: list[str] = Field(
        default_factory=list,
        description="Terms that should appear in spurious/rejected-addition text.",
    )
    max_actionable_findings: int | None = Field(
        default=None,
        ge=0,
        description="Maximum acceptable actionable findings for known-good cases.",
    )


class MethodologyReviewCase(BaseModel):
    """One frozen methodology-review benchmark input."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable case identifier.")
    title: str = Field(description="Human-readable case title.")
    category: CaseCategory = Field(description="Case class used for aggregate diagnostics.")
    artifact_section: str = Field(description="Methodology section under review.")
    expected: MethodologyReviewExpectation = Field(description="Deterministic scoring expectations.")


def load_methodology_review_cases(path: Path | None = None) -> list[MethodologyReviewCase]:
    """Load the frozen methodology-review case set."""
    payload = json.loads((path or _CASE_DATA_PATH).read_text(encoding="utf-8"))
    return [MethodologyReviewCase.model_validate(item) for item in payload["cases"]]


def load_methodology_review_baseline_outputs(
    path: Path | None = None,
) -> list[PrecomputedOutput]:
    """Load deterministic gold-reference outputs for smoke testing the evaluator."""
    payload = json.loads((path or _BASELINE_OUTPUTS_PATH).read_text(encoding="utf-8"))
    return [PrecomputedOutput.model_validate(item) for item in payload["outputs"]]


def methodology_review_inputs(
    cases: list[MethodologyReviewCase] | None = None,
) -> list[ExperimentInput]:
    """Convert frozen cases into prompt_eval precomputed-evaluation inputs."""
    resolved = cases or load_methodology_review_cases()
    return [
        ExperimentInput(
            id=case.id,
            content=case.artifact_section,
            expected=case.model_dump(),
        )
        for case in resolved
    ]


def methodology_review_evaluator(output: Any, expected: Any | None) -> EvalScore:
    """Score one review JSON payload against a methodology-review case."""
    if expected is None:
        return EvalScore(score=0.0, reasoning="Missing expected case metadata.")
    case = MethodologyReviewCase.model_validate(expected)
    payload = _coerce_review_payload(output)
    sections = _extract_review_sections(payload)

    defect_score = _term_coverage(
        case.expected.required_defect_terms,
        sections["actionable_text"],
    )
    gap_score = _term_coverage(
        case.expected.required_optimum_gap_terms,
        sections["optimum_gap_text"],
    )
    false_positive_score = _false_positive_score(
        case=case,
        actionable_count=len(sections["actionable_claims"]),
    )
    spurious_score = _spurious_trap_score(case=case, sections=sections)
    stability_score = _actionable_stability_score(sections["actionable_claims"])

    if case.category == "known_defective":
        detection_score = _mean([defect_score, gap_score])
    else:
        detection_score = 1.0

    dimensions = {
        "detects_required_defects": detection_score,
        "avoids_false_positives": false_positive_score,
        "rejects_spurious_additions": spurious_score,
        "actionable_finding_stability": stability_score,
    }
    overall = _mean(list(dimensions.values()))
    reasoning = (
        f"{case.id}: category={case.category}; "
        f"actionable={len(sections['actionable_claims'])}; "
        f"defect_terms={defect_score:.2f}; optimum_gap_terms={gap_score:.2f}; "
        f"false_positive={false_positive_score:.2f}; "
        f"spurious={spurious_score:.2f}; stability={stability_score:.2f}"
    )
    return EvalScore(score=overall, dimension_scores=dimensions, reasoning=reasoning)


def summarize_methodology_review_trials(trials: list[Any]) -> dict[str, Any]:
    """Summarize false positives, missed defects, and stability failures."""
    false_positives: list[str] = []
    missed_defects: list[str] = []
    spurious_failures: list[str] = []
    unstable_actionables: list[str] = []
    for trial in trials:
        dims = trial.dimension_scores or {}
        if dims.get("avoids_false_positives", 1.0) < 1.0:
            false_positives.append(trial.input_id)
        if dims.get("detects_required_defects", 1.0) < 1.0:
            missed_defects.append(trial.input_id)
        if dims.get("rejects_spurious_additions", 1.0) < 1.0:
            spurious_failures.append(trial.input_id)
        if dims.get("actionable_finding_stability", 1.0) < 1.0:
            unstable_actionables.append(trial.input_id)
    return {
        "false_positives": sorted(set(false_positives)),
        "missed_defects": sorted(set(missed_defects)),
        "spurious_failures": sorted(set(spurious_failures)),
        "unstable_actionable_findings": sorted(set(unstable_actionables)),
    }


def _coerce_review_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, str):
        return json.loads(output)
    if isinstance(output, dict):
        return output
    raise TypeError(f"Unsupported review output type: {type(output).__name__}")


def _extract_review_sections(payload: dict[str, Any]) -> dict[str, Any]:
    correctness = [
        item
        for item in payload.get("correctness_findings", [])
        if str(item.get("severity", "warn")) == "high"
    ]
    contracts = payload.get("contract_violations", [])
    annotations = payload.get("profile_annotations", []) or []
    optimum_gaps = [item for item in annotations if item.get("kind") == "optimum_gap"]
    spurious = [item for item in annotations if item.get("kind") == "spurious"]

    actionable_claims = [
        str(item.get("claim", ""))
        for item in correctness
    ] + [
        f"{item.get('constraint', '')} {item.get('violation', '')}"
        for item in contracts
    ] + [
        f"{item.get('claim', '')} {item.get('validity_loss_without_change', '')}"
        for item in optimum_gaps
    ]
    return {
        "actionable_claims": actionable_claims,
        "actionable_text": " ".join(actionable_claims),
        "optimum_gap_text": " ".join(
            f"{item.get('claim', '')} {item.get('validity_loss_without_change', '')}"
            for item in optimum_gaps
        ),
        "spurious_text": " ".join(
            f"{item.get('claim', '')} {item.get('why_rejected_or_uncertain', '')}"
            for item in spurious
        ),
    }


def _term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    normalized_text = _normalize(text)
    matched = sum(1 for term in terms if _normalize(term) in normalized_text)
    return matched / len(terms)


def _false_positive_score(*, case: MethodologyReviewCase, actionable_count: int) -> float:
    if case.expected.max_actionable_findings is None:
        return 1.0
    return 1.0 if actionable_count <= case.expected.max_actionable_findings else 0.0


def _spurious_trap_score(
    *,
    case: MethodologyReviewCase,
    sections: dict[str, Any],
) -> float:
    if case.category != "spurious_trap":
        return 1.0
    if _term_coverage(case.expected.spurious_trap_terms, sections["actionable_text"]) > 0.0:
        return 0.0
    return _term_coverage(case.expected.expected_spurious_terms, sections["spurious_text"])


def _actionable_stability_score(claims: list[str]) -> float:
    if not claims:
        return 1.0
    normalized = [_normalize(claim) for claim in claims if _normalize(claim)]
    if not normalized:
        return 1.0
    return len(set(normalized)) / len(normalized)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
