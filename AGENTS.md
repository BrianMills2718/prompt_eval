# Prompt Eval

<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->
<!-- generated_by: scripts/meta/render_agents_md.py -->
<!-- canonical_claude: CLAUDE.md -->
<!-- canonical_relationships: scripts/relationships.yaml -->
<!-- canonical_relationships_sha256: b7705d70d437 -->
<!-- sync_check: python scripts/meta/check_agents_sync.py --check -->

This file is a generated Codex-oriented projection of repo governance.
Edit the canonical sources instead of editing this file directly.

Canonical governance sources:
- `CLAUDE.md` — human-readable project rules, workflow, and references
- `scripts/relationships.yaml` — machine-readable ADR, coupling, and required-reading graph

## Purpose

`prompt_eval` is the prompt-evaluation and optimization layer for the Brian
projects ecosystem. It owns experiment semantics, prompt variants, evaluators,
statistics, and optimization loops on top of `llm_client`'s shared execution
and observability substrate.

This file is the canonical repo-governance source. `AGENTS.md` must mirror it
exactly and must not become a second hand-maintained authority.

## Commands

```bash
source .venv/bin/activate
pytest tests/ -v
python scripts/meta/sync_plan_status.py --check
python scripts/meta/check_plan_blockers.py --strict
python scripts/meta/check_plan_tests.py --plan 1
mypy prompt_eval tests
ruff check prompt_eval tests
python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md
```

## Operating Rules

This projection keeps the highest-signal rules in always-on Codex context.
For full project structure, detailed terminology, and any rule omitted here,
read `CLAUDE.md` directly.

### Principles

- `prompt_eval` is prompt-centric. Do not widen it into generic execution,
  retrieval, or workflow tooling without an explicit scope decision.
- `llm_client` remains authoritative for model execution, prompt asset
  resolution, required call metadata, and shared observability.
- Truth-surface deterministic and semantic review tooling belongs to
  `enforced-planning`. `prompt_eval` may be used as a governed consumer pilot,
  but that does not make truth-surface review part of `prompt_eval`'s default
  product scope unless a later explicit plan broadens the repo boundary.
- Experiment-semantic choices such as the subject model should be explicit, not
  silently chosen by package defaults.
- Prompt assets and `prompt_ref` provenance are the preferred long-term
  contract when available. Inline message lists remain an intentionally
  supported input for ad hoc and project-local experiments.
- Local JSON persistence is secondary. Shared `llm_client` observability is the
  authoritative cross-project record.

### Workflow

1. Read `CLAUDE.md`, `README.md`, and the active roadmap or uncertainty docs
   before changing repo behavior.
2. Anchor non-trivial work to `docs/plans/` and keep the plan index honest.
3. Keep package scope explicit when adding features. If a change blurs the
   `prompt_eval` versus `llm_client` boundary, document the decision.
4. Edit `CLAUDE.md` first, then resync `AGENTS.md`. Do not hand-maintain two
   divergent instruction files.
5. Run tests, plan-status checks, and link checks before closing the slice.
6. Repo-local governance helpers such as `scripts/meta/merge_pr.py` must stay
   aligned with the sanctioned worktree contract. The current helper uses
   rename-safe, path-based cleanup when a merged branch name no longer matches
   the worktree directory created earlier in the flow.

## Machine-Readable Governance

`scripts/relationships.yaml` is the source of truth for machine-readable governance in this repo: ADR coupling, required-reading edges, and doc-code linkage. This generated file does not inline that graph; it records the canonical path and sync marker, then points operators and validators back to the source graph. Prefer deterministic validators over prompt-only memory when those scripts are available.

## References

- `README.md` - package boundary, quick start, and docs map
- `docs/plans/01_master-roadmap.md` - canonical roadmap and current program state
- `docs/plans/CLAUDE.md` - plan index and status contract
- `docs/UNCERTAINTIES.md` - unresolved architecture and scope questions
- `docs/API_REFERENCE.md` - public API overview
- `docs/adr/README.md` - ADR index for repo decisions
- `scripts/CLAUDE.md` - repo-local script inventory and workflow helpers
