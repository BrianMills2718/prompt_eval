# Prompt Eval Artifact Register

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Primary Reviewer Artifacts

| Artifact | Role | Portfolio meaning |
|---|---|---|
| [PROJECT.md](../PROJECT.md) | Dossier entrypoint | Frames the repo as supporting prompt-eval infrastructure. |
| [README.md](../README.md) | Project overview | API surface, boundaries, examples, and docs map. |
| [docs/PORTFOLIO_EXPERIMENT_SUMMARY.md](PORTFOLIO_EXPERIMENT_SUMMARY.md) | Portfolio guide | Best reviewer-facing experiment-note shape. |
| [docs/ops/CAPABILITY_DECOMPOSITION.md](ops/CAPABILITY_DECOMPOSITION.md) | Ownership ledger | Defines prompt-eval vs runtime vs application boundaries. |
| [docs/API_REFERENCE.md](API_REFERENCE.md) | API reference | Public surface summary. |
| [docs/UNCERTAINTIES.md](UNCERTAINTIES.md) | Architecture uncertainty ledger | Resolved and open integration questions. |
| [docs/METHODOLOGY.md](METHODOLOGY.md) | Methodology spine | Explains experiment design and failure modes. |
| [docs/VALIDATION.md](VALIDATION.md) | Validation register | Separates experiment validity from downstream truth. |
| [docs/CONCERNS.md](CONCERNS.md) | Concern register | Tracks open evidence and boundary risks. |

## Code And Execution Surfaces

| Surface | Role |
|---|---|
| `prompt_eval/experiment.py` | Experiment, variant, input, result, and summary models. |
| `prompt_eval/runner.py` | Experiment execution and observability emission. |
| `prompt_eval/stats.py` | Variant comparison and inference helpers. |
| `prompt_eval/evaluators.py` | Trial and corpus evaluator surfaces. |
| `prompt_eval/acceptable_set.py` | Growing acceptable-set evaluator. |
| `prompt_eval/query.py` | Read-side reconstruction from shared observability. |
| `prompt_eval/prompt_assets.py` | Prompt asset compatibility. |
| `tests/` | Unit and integration coverage for experiment semantics. |

## Evidence Artifacts

| Artifact | Evidence | Notes |
|---|---|---|
| [docs/PORTFOLIO_EXPERIMENT_SUMMARY.md](PORTFOLIO_EXPERIMENT_SUMMARY.md) | Portfolio experiment template | Best public-facing shape. |
| [docs/adr/0003-explicit-experiment-semantics.md](adr/0003-explicit-experiment-semantics.md) | Explicit experiment choices | Prevents hidden subject-model defaults. |
| [docs/adr/0005-statistical-inference-boundary.md](adr/0005-statistical-inference-boundary.md) | Inference limits | Documents lightweight comparison caveats. |
| [docs/adr/0007-paired-by-input-comparison-mode.md](adr/0007-paired-by-input-comparison-mode.md) | Stronger comparison mode | Matched-input comparison when inputs overlap. |
| [docs/ops/CAPABILITY_DECOMPOSITION.md](ops/CAPABILITY_DECOMPOSITION.md) | Capability ownership | Keeps prompt-eval scoped. |

## Missing Portfolio Artifacts

- One populated experiment note tied to an applied project decision.
- A small frozen case set excerpt with inclusion criteria.
- A before/after result table with confidence interval or comparison result.
- A failure-case analysis showing where the candidate did not improve.
- A consumer note from AC15, Grounded Research, Qualitative Coding, or another
  project explaining why the experiment changed behavior.
