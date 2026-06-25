# Prompt Eval Validation Register

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Validation Position

`prompt_eval` has implementation evidence for prompt-centric experiments and
comparison. It still needs a concrete applied experiment packet to become
strong portfolio evidence.

The key distinction:

- **experiment-valid:** variants, inputs, replicates, evaluator, and subject
  model are explicit;
- **metric-valid:** the evaluator measures the intended bounded behavior;
- **comparison-valid:** the statistical mode is appropriate for the collected
  trial structure;
- **decision-valid:** the downstream project adopts, rejects, narrows, or reruns
  based on the result;
- **analysis-valid:** the downstream analytic output is substantively correct.

This repo owns the first three categories. Applied projects own decision and
analysis validity.

## Current Evidence

| Evidence area | Current artifact | Claim licensed |
|---|---|---|
| Package boundary | `README.md` | Prompt-eval semantics are separated from `llm_client` runtime. |
| Portfolio experiment shape | `docs/PORTFOLIO_EXPERIMENT_SUMMARY.md` | Reviewer-facing evidence format exists. |
| Capability ownership | `docs/ops/CAPABILITY_DECOMPOSITION.md` | Scope and non-goals are explicit. |
| Experiment semantics | `docs/adr/0003-explicit-experiment-semantics.md` | Important experiment choices must be explicit. |
| Statistical boundary | `docs/adr/0005-statistical-inference-boundary.md` | Lightweight inference caveats are documented. |
| Paired comparison | `docs/adr/0007-paired-by-input-comparison-mode.md` | Matched-input comparison mode is documented. |

## Evidence Not Yet Present

Do not claim the following without new evidence:

- prompt-eval results prove downstream analytic truth;
- a tiny or unrepresentative case set justifies broad behavior changes;
- pooled comparison is publication-grade inference;
- prompt_eval owns generic runtime, retrieval, or workflow optimization;
- a package API overview is enough portfolio evidence without an applied
  decision.

## Commands

Core checks:

```bash
make test
make lint
make typecheck
python scripts/check_markdown_links.py PROJECT.md docs/METHODOLOGY.md docs/ARTIFACTS.md docs/VALIDATION.md docs/CONCERNS.md docs/wiki_manifest.yaml
git diff --check
```

Truth-surface operator view:

```bash
python scripts/render_truth_surface_status.py --config truth_surface_drift.yaml
```

The current truth-surface status has known historical/path issues recorded in
[docs/ops/TRUTH_SURFACE_STATUS.md](ops/TRUTH_SURFACE_STATUS.md). Do not present
that pilot as package ownership without a new boundary decision.

## Portfolio Readiness Gate

The repo is dossier-complete when the documentation spine is present. It
becomes strong external portfolio evidence only after:

1. One applied project experiment packet is published.
2. The case set and inclusion criteria are frozen.
3. Baseline and candidate outputs are saved.
4. The metric, uncertainty, failure cases, and adoption decision are explicit.
