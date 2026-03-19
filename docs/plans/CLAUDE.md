# Implementation Plans

Canonical planning surface for `prompt_eval`.

## Gap Summary

| # | Plan | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [Prompt Eval Master Roadmap](01_master-roadmap.md) | Highest | 🚧 In Progress | - |
| 2 | [Shared Observability Boundary](02_shared-observability-boundary.md) | Highest | ✅ Complete | - |
| 3 | [Prompt Asset And Scope Boundary](03_prompt-asset-and-scope-boundary.md) | High | ⏸️ Blocked | ecosystem prompt-asset adoption and scope decision |
| 4 | [Documentation Surface Hardening](04_documentation-surface-hardening.md) | Medium | ✅ Complete | - |
| 5 | [Explicit Experiment Semantics And Model Governance Alignment](05_model-governance-alignment.md) | High | 🚧 In Progress | - |

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

The active unblocked program is Plan 05: explicit experiment semantics and
model-governance alignment. Plan 03 remains blocked on prompt-asset and
package-scope decisions, but Plan 05 can proceed independently because it is
about fail-loud experiment contracts, not prompt-asset adoption policy.

## Trivial Changes

Not every edit needs a plan. `[Trivial]` is appropriate for changes like:

- small typo or wording fixes,
- comment/docstring correction with no behavior change,
- narrow test maintenance with no production-code change,
- no new files beyond obvious documentation-only additions.

Avoid calling something trivial if it changes `prompt_eval/`, the public docs
contract, or plan/uncertainty status.
