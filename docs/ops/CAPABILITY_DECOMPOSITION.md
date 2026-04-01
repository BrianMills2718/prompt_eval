# Capability Decomposition

Last updated: 2026-04-01

## Purpose

This document is the repo-local source of record for what `prompt_eval`
currently owns, what it intentionally consumes from shared infrastructure, and
what it should not quietly expand into.

Use this together with:

- [`../plans/13_linkage-deepening-and-capability-ownership.md`](../plans/13_linkage-deepening-and-capability-ownership.md)
- [`../UNCERTAINTIES.md`](../UNCERTAINTIES.md)
- [`../../README.md`](../../README.md)

## Role

`prompt_eval` is the shared prompt-evaluation and optimization layer for the
ecosystem.

It owns:

- experiment semantics
- prompt variants and prompt-centric input/output contracts
- evaluator composition
- statistical comparison
- optimization loops over prompts and prompt assets

It does not own:

- raw model execution
- shared observability storage
- generic agent runtime orchestration
- retrieval or project-specific application logic

Those stay in shared infrastructure such as `llm_client` or in consuming
projects.

## Capability Ledger

| Capability | Current owner | Intended owner | Class | Posture | Notes |
|---|---|---|---|---|---|
| Prompt-centric experiment semantics, evaluation, statistical comparison, and optimization | `prompt_eval` | `prompt_eval` | shared infrastructure | no move planned | This is the primary shared capability exported by the repo. |
| Precomputed-output evaluation and comparison of external system runs | `prompt_eval` | `prompt_eval` | shared infrastructure | no move planned | This lets consumer repos compare frozen outputs without re-owning the full experiment runner. |
| Raw model execution, prompt rendering substrate, budgets, and shared observability | `llm_client` | `llm_client` | consumed shared infrastructure | consume, do not re-own | `prompt_eval` should stay on top of this surface rather than rebuilding it locally. |
| Repo-local JSON persistence and result exports | `prompt_eval` | `prompt_eval` | compatibility surface | retain as secondary path | Useful for portability and exports, but not the authoritative shared record. |

## Known Consumers

Current known ecosystem consumers include:

- `grounded-research`
- `onto-canon6`
- `Digimon_for_KG_application`

This list is intentionally evidence-based, not aspirational. Add a repo only
when there is a real integration or a maintained dependency.

## Boundary Rules

1. Keep `prompt_eval` prompt-centric.
2. Shared execution, prompt asset resolution, and observability stay in
   `llm_client`.
3. Consumer repos should integrate with `prompt_eval` rather than silently
   re-owning competing prompt-evaluation frameworks.
4. If a new capability would make `prompt_eval` a generic workflow or retrieval
   runtime, stop and document a new boundary decision before implementing it.

## Open Uncertainties

- How quickly consumer repos should be required to prefer `prompt_eval` over
  repo-local evaluation helpers is still unsettled.
- The next enforcement step after this first linkage-and-ownership wave is not
  fixed yet; stricter doc-coupling may come before broader capability-row
  expansion.
- Some evaluator and optimization helpers still look repo-local in adoption
  practice even when the long-term intent is shared use.
