# Plan 04: Documentation Surface Hardening

**Status:** ✅ Complete
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** trustworthy repo-operating and planning docs

---

## Gap

**Current:** before this cleanup, `prompt_eval` had no `README.md`, duplicated
stale root docs, a placeholder plan index pointing to a nonexistent example
plan, and a large copied `meta-patterns` subtree that looked active but did not
match the repo's real structure.

**Target:** one compact canonical docs surface:

- root `README.md` for project overview and usage,
- `AGENTS.md` for repo-operating instructions,
- `CLAUDE.md` as a thin pointer,
- a real plan index plus child plans,
- archived legacy notes moved out of the active docs surface.

**Why:** documentation should answer what the repo is, what is current, and
what still remains without forcing readers to infer canon from copied
scaffolding.

---

## References Reviewed

- `AGENTS.md`
- `CLAUDE.md`
- `docs/plans/CLAUDE.md`
- `docs/UNCERTAINTIES.md`
- `docs/adr/README.md`
- `scripts/CLAUDE.md`
- `tests/CLAUDE.md`

---

## Files Affected

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/API_REFERENCE.md`
- `docs/archive/README.md`
- `docs/archive/meta-patterns/`
- `docs/plans/CLAUDE.md`
- `docs/plans/01_master-roadmap.md`
- `docs/plans/02_shared-observability-boundary.md`
- `docs/plans/03_prompt-asset-and-scope-boundary.md`
- `scripts/CLAUDE.md`
- `tests/CLAUDE.md`

---

## Plan

### Steps

1. Create a real repo `README.md`.
2. Collapse duplicated root operator docs into `AGENTS.md` plus a thin
   `CLAUDE.md`.
3. Replace the placeholder plan index with real plans and a master roadmap.
4. Move legacy copied process notes into `docs/archive/`.
5. Rewrite stale `scripts/` and `tests/` guidance so it matches the real repo.

---

## Required Tests

### New Tests (TDD)

Documentation-only plan. No new runtime tests required.

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python scripts/meta/sync_plan_status.py --check` | Plan graph is internally consistent |
| markdown link scan over active docs | Canonical docs stop pointing at missing files |
| `git diff --check` | Documentation edits stay mechanically clean |

---

## Acceptance Criteria

- [x] `README.md` exists and accurately states the current architecture
- [x] `AGENTS.md` and `CLAUDE.md` no longer duplicate large stale content
- [x] `docs/plans/CLAUDE.md` points to real plans, not placeholder examples
- [x] Legacy `meta-patterns` notes are archived out of the canonical docs path
- [x] `scripts/CLAUDE.md` and `tests/CLAUDE.md` describe the actual repo

---

## Notes

This plan is complete. Future doc work should update the canonical surface
above, not reintroduce copied generic scaffolding.
