"""Evaluate precomputed methodology-whitepaper review outputs.

The default run uses bundled gold-reference outputs as an offline smoke test.
Callers can pass `--outputs` with review-artifact JSON outputs grouped as
PrecomputedOutput records to compare reviewer profile/model variants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from prompt_eval.case_sets.methodology_whitepaper_review import (
    load_methodology_review_baseline_outputs,
    load_methodology_review_cases,
    methodology_review_evaluator,
    methodology_review_inputs,
    summarize_methodology_review_trials,
)
from prompt_eval.experiment import PrecomputedOutput
from prompt_eval.runner import evaluate_precomputed_variants


def _load_outputs(path: Path | None) -> list[PrecomputedOutput]:
    """Load caller-supplied outputs or bundled gold-reference outputs."""
    if path is None:
        return load_methodology_review_baseline_outputs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["outputs"] if isinstance(payload, dict) and "outputs" in payload else payload
    return [PrecomputedOutput.model_validate(item) for item in records]


async def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_methodology_review_cases(Path(args.cases) if args.cases else None)
    outputs = _load_outputs(Path(args.outputs) if args.outputs else None)
    result = await evaluate_precomputed_variants(
        experiment_name=args.experiment_name,
        inputs=methodology_review_inputs(cases),
        outputs=outputs,
        evaluator=methodology_review_evaluator,
        observability=False,
    )
    diagnostics = summarize_methodology_review_trials(result.trials)
    report = {
        "experiment_name": result.experiment_name,
        "variants": result.variants,
        "n_cases": len(cases),
        "n_trials": len(result.trials),
        "summary": {
            name: summary.model_dump()
            for name, summary in sorted(result.summary.items())
        },
        "diagnostics": diagnostics,
        "trials": [
            {
                "variant_name": trial.variant_name,
                "input_id": trial.input_id,
                "replicate": trial.replicate,
                "score": trial.score,
                "dimension_scores": trial.dimension_scores,
                "reasoning": trial.reasoning,
                "error": trial.error,
            }
            for trial in result.trials
        ],
    }
    return report


def main() -> None:
    """Run the methodology-review case-set evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="",
        help="Optional methodology-review case JSON path. Defaults to bundled cases.",
    )
    parser.add_argument(
        "--outputs",
        default="",
        help="Optional PrecomputedOutput JSON/JSON-list path. Defaults to bundled gold outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        default="methodology_whitepaper_review",
        help="Experiment name for the precomputed evaluation result.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional JSON report path. When omitted, prints to stdout only.",
    )
    args = parser.parse_args()
    report = asyncio.run(_evaluate(args))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
