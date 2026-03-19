# Implementation Plans

Canonical planning surface for `prompt_eval`.

## Gap Summary

| # | Plan | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [Prompt Eval Master Roadmap](01_master-roadmap.md) | Highest | 🚧 In Progress | 03 |
| 2 | [Shared Observability Boundary](02_shared-observability-boundary.md) | Highest | ✅ Complete | - |
| 3 | [Prompt Asset And Scope Boundary](03_prompt-asset-and-scope-boundary.md) | High | ⏸️ Blocked | ecosystem prompt-asset adoption and scope decision |
| 4 | [Documentation Surface Hardening](04_documentation-surface-hardening.md) | Medium | ✅ Complete | - |

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

There is no unblocked implementation slice inside `prompt_eval` today. The open
program is Plan 03, and it is blocked on two explicit decisions:

1. whether inline message lists are permanent compatibility input or a
   deprecating path once prompt assets are broadly adopted,
2. whether `prompt_eval` stays prompt-centric or expands into broader
   non-prompt optimization.

Do not invent new cleanup work while those remain unresolved.

## Trivial Changes

Not every edit needs a plan. `[Trivial]` is appropriate for changes like:

- small typo or wording fixes,
- comment/docstring correction with no behavior change,
- narrow test maintenance with no production-code change,
- no new files beyond obvious documentation-only additions.

Avoid calling something trivial if it changes `prompt_eval/`, the public docs
contract, or plan/uncertainty status.
