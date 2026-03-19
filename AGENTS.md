# Prompt Eval

Canonical repo-operating instructions for `prompt_eval` live here.

Read these first:

1. [README.md](README.md)
2. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
3. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
4. [docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md)
5. [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
6. [docs/adr/README.md](docs/adr/README.md)

Short version:

- `prompt_eval` is the prompt-evaluation and optimization layer. It is not the
  shared runtime substrate.
- `llm_client` remains authoritative for execution, prompt asset resolution,
  required call metadata, and shared observability.
- Default persistence is dual-write: shared `llm_client` observability plus
  optional local JSON artifacts from `store.py`.
- Prefer explicit prompt assets and `prompt_ref` provenance when available.
  Inline message lists remain a compatibility input, not the preferred
  long-term contract.
- Experiment-semantic choices should be explicit. In particular, `prompt_eval`
  should not silently choose the subject model for an experiment or optimizer
  helper. Hidden defaults are acceptable for operational plumbing, not for the
  thing being evaluated.
- `llm_client` task-selection buckets are an internal convenience and reporting
  vocabulary. They are not the main public experiment-design abstraction in
  `prompt_eval`.
- Keep the package prompt-centric. Do not widen it into generic workflow, code,
  or retrieval optimization without an explicit scope decision and plan update.
- Plans are real in this repo now. Anchor work to the roadmap and continue
  through consecutive unblocked slices. Stop only for a real blocker,
  reprioritization, or an unresolved design decision.

Current roadmap state:

- Program A: shared observability boundary complete
- Program B: prompt asset and scope boundary blocked on ecosystem/product
  decisions
- Program C: documentation surface hardening complete
- Program D: explicit experiment semantics and model-governance alignment
  complete
- Program E: prompts-as-data cleanup complete

Do not invent fresh cleanup work unless it ties to an active plan, issue, or
new evidence.
