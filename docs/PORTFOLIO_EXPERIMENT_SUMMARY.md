# Prompt Eval Portfolio Experiment Summary

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Portfolio Claim

Prompt Eval is supporting AI-engineering evidence. It shows that prompt, schema, rubric, and model changes can be evaluated with frozen cases and statistical comparison instead of manual spot checks.

## Reviewer-Facing Experiment Shape

Use a compact evidence note with these fields:

| Field | Required content |
|-------|------------------|
| Applied project | The project whose behavior changed, such as AC15 or Grounded Research |
| Frozen case set | Dataset name, size, and inclusion criteria |
| Baseline | Existing prompt, schema, model, or pipeline variant |
| Candidate | Proposed change under test |
| Metric | Structural pass rate, rubric score, acceptable-set score, or task-specific benchmark |
| Uncertainty | Confidence interval or statistical comparison result |
| Decision | Keep, reject, narrow, or re-run with a better case set |

## Example Decision Template

`Candidate B improved structural pass rate from X to Y on N frozen cases, with confidence interval Z. Adopt for the bounded workflow, but keep baseline A for cases outside the tested domain.`

## Caveat

Prompt Eval should not be sold as a product by itself. It is best used as evidence that applied projects make prompt and schema changes through measurement rather than intuition.
