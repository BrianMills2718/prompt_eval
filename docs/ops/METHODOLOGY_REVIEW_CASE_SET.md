# Methodology Whitepaper Review Case Set

This frozen case set scores precomputed `llm_client review-artifact` outputs
for the `quality_optimal_whitepaper` review profile. It is not a model runner:
generate review JSON elsewhere, then feed those outputs into `prompt_eval`.

## Cases

- 2 known-good methodology sections that should not receive actionable defects.
- 2 known-defective sections with required validity defects and optimum gaps.
- 2 spurious-addition traps that should be rejected rather than proposed as
  defects or optimum gaps.

The deterministic evaluator reports:

- false positives on known-good cases
- missed required defects
- spurious-addition failures
- duplicate actionable-finding instability

## Offline Smoke

```bash
PYTHONPATH=. python scripts/evaluate_methodology_review_cases.py \
  --out /tmp/methodology_review_case_set_report.json
```

Baseline result for the bundled gold-reference outputs:

- variant: `gold_reference`
- cases: 6
- trials: 6
- mean score: `1.0`
- false positives: none
- missed defects: none
- spurious failures: none
- unstable actionable findings: none

## Evaluating Real Review Outputs

Create a JSON file containing `PrecomputedOutput` records:

```json
{
  "outputs": [
    {
      "variant_name": "quality_optimal_whitepaper__claude-code_opus",
      "input_id": "defect_erases_time_and_modality",
      "replicate": 0,
      "subject_model": "claude-code/opus",
      "output": {
        "artifact_label": "defect_erases_time_and_modality",
        "verdict": "blocker",
        "summary": "Review JSON from llm_client goes here.",
        "correctness_findings": [],
        "contract_violations": [],
        "nits": [],
        "unverified_claims": [],
        "scope_drift_findings": [],
        "profile_annotations": []
      }
    }
  ]
}
```

Then run:

```bash
PYTHONPATH=. python scripts/evaluate_methodology_review_cases.py \
  --outputs path/to/precomputed_outputs.json \
  --out /tmp/methodology_review_profile_report.json
```

Do not claim one profile/model pair is better than another without a frozen
output file that covers all six cases for every compared variant.
