# Implementation Plans

Canonical planning surface for `prompt_eval`.

## Gap Summary

| # | Plan | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [Prompt Eval Master Roadmap](01_master-roadmap.md) | Highest | ✅ Complete | - |
| 2 | [Shared Observability Boundary](02_shared-observability-boundary.md) | Highest | ✅ Complete | - |
| 3 | [Prompt Asset And Scope Boundary](03_prompt-asset-and-scope-boundary.md) | High | ✅ Complete | - |
| 4 | [Documentation Surface Hardening](04_documentation-surface-hardening.md) | Medium | ✅ Complete | - |
| 5 | [Explicit Experiment Semantics And Model Governance Alignment](05_model-governance-alignment.md) | High | ✅ Complete | - |
| 6 | [Prompts As Data Cleanup](06_prompts-as-data-cleanup.md) | Medium | ✅ Complete | - |
| 7 | [Statistical Engine Modernization](07_statistical-engine-modernization.md) | High | ✅ Complete | - |
| 8 | [Paired-By-Input Comparison Mode](08_paired-by-input-comparison-mode.md) | High | ✅ Complete | - |
| 9 | [Growing Acceptable Set Evaluator](09_growing-acceptable-set-evaluator.md) | High | ✅ Complete | - |
| 10 | [CI And Governance Hygiene](10_ci_and_hygiene.md) | Medium | ✅ Complete | - |
| 11 | [Precomputed Variant Comparison](11_precomputed_variant_comparison.md) | High | ✅ Complete | - |
| 12 | [Governed Baseline Repair For Active-Stack Candidacy](12_governed-baseline-repair-for-active-stack-candidacy.md) | High | ✅ Complete | - |
| 13 | [Linkage Deepening And Capability Ownership](13_linkage-deepening-and-capability-ownership.md) | High | ✅ Complete | 12 |
| 14 | [Authoritative coordination wave-1 rollout](14_authoritative-coordination-wave-1-rollout.md) | Critical | ✅ Complete | - |
| 15 | [Semantic Truth-Surface Review Pilot](15_semantic-truth-surface-review-pilot.md) | High | ✅ Complete | 14 |

## Status Key

| Status | Meaning |
|--------|---------|
| `📋 Planned` | Designed and ready to implement |
| `🚧 In Progress` | Active program or implementation work |
| `⏸️ Blocked` | Waiting on an external dependency or design decision |
| `✅ Complete` | Implemented and verified |

## How To Use This Index

1. Read the [master roadmap](01_master-roadmap.md) first.
2. Use child plans for execution details and acceptance criteria.
3. Treat plan files as the source of truth; keep this index in sync with
   `python scripts/meta/sync_plan_status.py --check`.
4. Update plans when the default next slice changes. Do not leave the index as
   placeholder scaffolding.

## Current Default Next Step

No new `prompt_eval`-local follow-on is active by default. Use [Plan 15: Semantic Truth-Surface Review Pilot](15_semantic-truth-surface-review-pilot.md)
as evidence for shared follow-on work in `enforced-planning` if the ecosystem wants stronger deterministic promotion or registry-hygiene enforcement.

## Trivial Changes

Not every edit needs a plan. `[Trivial]` is appropriate for changes like:

- small typo or wording fixes,
- comment/docstring correction with no behavior change,
- narrow test maintenance with no production-code change,
- no new files beyond obvious documentation-only additions.

Avoid calling something trivial if it changes `prompt_eval/`, the public docs
contract, or plan/uncertainty status.
