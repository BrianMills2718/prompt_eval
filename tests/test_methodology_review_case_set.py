"""Tests for the methodology-whitepaper review case set."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prompt_eval.case_sets.methodology_whitepaper_review import (
    load_methodology_review_baseline_outputs,
    load_methodology_review_cases,
    methodology_review_evaluator,
    methodology_review_inputs,
    summarize_methodology_review_trials,
)
from prompt_eval.runner import evaluate_precomputed_variants


def test_methodology_review_case_set_shape() -> None:
    cases = load_methodology_review_cases()
    categories = [case.category for case in cases]

    assert len(cases) == 6
    assert categories.count("known_good") == 2
    assert categories.count("known_defective") == 2
    assert categories.count("spurious_trap") == 2
    assert {case.id for case in cases} == {inp.id for inp in methodology_review_inputs(cases)}


def test_gold_baseline_scores_all_cases() -> None:
    cases = load_methodology_review_cases()
    outputs = load_methodology_review_baseline_outputs()

    result = asyncio.run(
        evaluate_precomputed_variants(
            experiment_name="methodology_review_baseline",
            inputs=methodology_review_inputs(cases),
            outputs=outputs,
            evaluator=methodology_review_evaluator,
            observability=False,
        )
    )
    diagnostics = summarize_methodology_review_trials(result.trials)

    assert len(result.trials) == 6
    assert result.summary["gold_reference"].mean_score == pytest.approx(1.0)
    assert diagnostics == {
        "false_positives": [],
        "missed_defects": [],
        "spurious_failures": [],
        "unstable_actionable_findings": [],
    }


def test_known_good_false_positive_is_penalized() -> None:
    case = next(
        case
        for case in load_methodology_review_cases()
        if case.id == "good_preserves_time_and_modality"
    )
    output = {
        "artifact_label": case.id,
        "verdict": "concerns",
        "summary": "bad false positive",
        "correctness_findings": [
            {
                "file_path": "case.md",
                "line": 1,
                "claim": "Add another validation table.",
                "severity": "high",
            }
        ],
        "contract_violations": [],
        "nits": [],
        "unverified_claims": [],
        "scope_drift_findings": [],
        "profile_annotations": [],
    }

    score = methodology_review_evaluator(output, case.model_dump())

    assert score.dimension_scores["avoids_false_positives"] == 0.0
    assert score.score < 1.0


def test_spurious_trap_actionable_addition_is_penalized() -> None:
    case = next(
        case
        for case in load_methodology_review_cases()
        if case.id == "spurious_dashboard_trap"
    )
    output = {
        "artifact_label": case.id,
        "verdict": "concerns",
        "summary": "bad spurious action",
        "correctness_findings": [],
        "contract_violations": [],
        "nits": [],
        "unverified_claims": [],
        "scope_drift_findings": [],
        "profile_annotations": [
            {
                "annotation_id": "og1",
                "kind": "optimum_gap",
                "claim": "Add a dashboard.",
                "linked_finding_index": 0,
                "validity_loss_without_change": "Dashboard absence hurts validity.",
            }
        ],
    }

    score = methodology_review_evaluator(output, case.model_dump())

    assert score.dimension_scores["rejects_spurious_additions"] == 0.0
    assert score.score < 1.0


def test_methodology_review_case_runner_smoke(tmp_path: Path) -> None:
    out = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_methodology_review_cases.py",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))

    assert "gold_reference" in report["summary"]
    assert report["summary"]["gold_reference"]["mean_score"] == 1.0
    assert report["diagnostics"]["missed_defects"] == []
    assert completed.stdout.strip().startswith("{")
