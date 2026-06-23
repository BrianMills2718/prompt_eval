# Prompt Eval Project Dossier

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Portfolio Role

`prompt_eval` is supporting shared infrastructure. It is the Brian-built layer
for prompt, schema, rubric, model, and evaluator comparisons on frozen case
sets.

It should not lead the portfolio as a standalone analyst artifact. Its value is
that applied projects can show prompt or schema changes were adopted because
of measured evidence, not manual spot checks or intuition.

## Current Status

Safe current claims:

- experiments can run across prompt variants, inputs, and replicates;
- trial-level and corpus-level evaluators are supported;
- statistical comparison exists for lightweight experiment triage;
- prompt assets and inline prompt variants are both supported;
- prompt-eval records dual-write into `llm_client` shared observability;
- acceptable-set evaluation exists for multi-correct-answer string outputs;
- precomputed-output comparison supports external system outputs;
- ADRs document experiment semantics, statistical boundaries, acceptable-set
  evaluation, prompt asset scope, and paired-by-input comparison;
- a reviewer-facing portfolio experiment summary exists.

Do not claim:

- prompt evaluation proves downstream analytic truth;
- pooled comparison is publication-grade causal inference;
- this repo owns raw LLM execution or observability storage;
- it is a generic workflow, retrieval, or code-optimization framework;
- infrastructure value is obvious without an applied project decision.

## Reviewer Path

1. Read [README.md](README.md) for the package surface and boundary.
2. Read [docs/PORTFOLIO_EXPERIMENT_SUMMARY.md](docs/PORTFOLIO_EXPERIMENT_SUMMARY.md)
   for the reviewer-facing experiment shape.
3. Read [docs/ops/CAPABILITY_DECOMPOSITION.md](docs/ops/CAPABILITY_DECOMPOSITION.md)
   for ownership boundaries.
4. Read [docs/adr/0003-explicit-experiment-semantics.md](docs/adr/0003-explicit-experiment-semantics.md)
   and [docs/adr/0005-statistical-inference-boundary.md](docs/adr/0005-statistical-inference-boundary.md)
   for experiment and inference caveats.
5. Read [docs/VALIDATION.md](docs/VALIDATION.md) and
   [docs/CONCERNS.md](docs/CONCERNS.md) before using results as evidence.

## Why It Matters For An AI Engineer / Analyst Portfolio

Applied LLM systems need a disciplined way to change prompts, models, schemas,
and evaluators. This repo shows the measurement layer: frozen cases, explicit
experiment semantics, repeatable runs, statistical comparison, acceptable-set
handling, and observability integration.

The best public framing is: "I built the shared prompt-evaluation layer so
applied projects can compare prompt and schema changes against frozen cases
before changing system behavior."

## Next Evidence To Create

The strongest portfolio artifact is one applied experiment note:

1. Identify the downstream project and behavior being changed.
2. Freeze the case set and inclusion criteria.
3. Compare baseline versus candidate.
4. Report metric, uncertainty, and failure cases.
5. State the decision: adopt, reject, narrow, or rerun.
