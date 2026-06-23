# Prompt Eval Methodology

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Goal

`prompt_eval` makes prompt and schema changes measurable. The goal is not to
make every LLM output correct. The goal is to let applied projects compare
bounded changes on frozen cases and make explicit decisions from the results.

The target loop is:

```text
case set -> baseline variant -> candidate variant -> evaluator -> comparison -> decision
```

## Design Method

The repo uses an experiment-semantics-first method:

1. Define an experiment with explicit prompt variants, inputs, models, and
   replicates.
2. Run variants through the shared `llm_client` runtime.
3. Score outputs with trial-level and optional corpus-level evaluators.
4. Persist authoritative shared records through `llm_client` observability.
5. Compare variants with an explicit comparison mode.
6. Preserve uncertainty, failure cases, and decision rationale.
7. Keep prompt-eval semantics separate from downstream analytic truth.

This keeps `prompt_eval` focused on prompt-centric experiment design rather
than raw execution, retrieval, orchestration, or project-specific analysis.

## Borrow-Vs-Build

Borrowed:

- `llm_client` for model execution and shared observability;
- SciPy-backed inference where statistical comparison needs numerical support;
- YAML/Jinja prompt assets where explicit prompt identity is available;
- Pydantic contracts for experiment and evaluator models.

Built locally:

- experiment, variant, input, trial, and result models;
- evaluator composition;
- prompt variant construction from assets or inline messages;
- statistical comparison wrapper and paired-by-input mode;
- acceptable-set evaluator and sidecar cache;
- precomputed-output comparison for external systems;
- portfolio experiment summary and boundary ADRs.

## Modality Split

Deductive / plan-first surfaces:

- experiment model contracts;
- required experiment-semantic declarations;
- evaluator input/output shape;
- comparison mode behavior;
- boundary with `llm_client` runtime and observability.

Exploratory / ladder surfaces:

- which metric best captures a downstream behavior;
- how large the frozen case set must be;
- whether a judge model is stable enough for the decision;
- whether observed deltas are actionable or need another run;
- which applied project experiments are worth portfolio presentation.

Exploratory surfaces need frozen cases and saved experiment results, not manual
spot checks.

## ADR Map

- [0003-explicit-experiment-semantics.md](adr/0003-explicit-experiment-semantics.md)
  requires experiment-semantic choices to be caller-declared.
- [0005-statistical-inference-boundary.md](adr/0005-statistical-inference-boundary.md)
  keeps comparison lightweight and explicit about inferential limits.
- [0007-paired-by-input-comparison-mode.md](adr/0007-paired-by-input-comparison-mode.md)
  adds matched-input comparison when inputs are shared across variants.

## Main Failure Modes

| Failure mode | Why it matters | Control |
|---|---|---|
| Manual spot checks treated as evidence | Results will not be reproducible. | Frozen case sets and saved runs. |
| Hidden experiment choices | Reviewers cannot tell what changed. | Explicit subject model and prompt source. |
| Overclaiming statistics | Lightweight comparisons can be useful but not definitive. | ADR 0005 caveats and paired mode. |
| Treating eval as truth | A score is not an analytic claim. | Downstream project owns substantive validation. |
| Rebuilding runtime behavior | Blurs boundary with `llm_client`. | Capability decomposition and ADR controls. |
