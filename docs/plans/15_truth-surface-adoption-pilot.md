# Plan #15: Truth-surface adoption pilot

**Status:** ✅ Complete

**Verified:** 2026-04-02T17:28:36Z
**Verification Evidence:**
```yaml
completed_by: scripts/meta/complete_plan.py
timestamp: 2026-04-02T17:28:36Z
tests:
  unit: 186 passed in 5.12s
  e2e_smoke: skipped (--skip-e2e)
  e2e_real: skipped (--skip-real-e2e)
  doc_coupling: passed
commit: 1fdc80f
```
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** stricter truth-surface workflow wiring in `prompt_eval`

---

## Gap

**Current:** `prompt_eval` is mechanically governed and already participates in
shared coordination, but it does not yet consume the new truth-surface
validator/renderer through a repo-local config. During pilot setup, the
sanctioned repo-local plan creation path also exposed real friction: the local
`docs/plans/TEMPLATE.md` is stale enough that `create_plan.py` generated a
placeholder plan instead of a usable execution contract.

**Target:** `prompt_eval` has one clear local pilot plan, a repo-local
truth-surface config, a measured audit artifact, and a generated truth-surface
status output. The pilot should also fix the stale local plan template so future
repo-local plan creation remains trustworthy.

**Why:** `prompt_eval` is the cleanest governed repo with real claims/worktree
surfaces enabled. Completing the pilot here proves that the framework can be
consumed by a real repo and exposes the exact friction that still needs to be
resolved before broader default wiring.

---

## References Reviewed

- `CLAUDE.md`
- `README.md`
- `docs/UNCERTAINTIES.md`
- `docs/ops/CAPABILITY_DECOMPOSITION.md`
- `docs/plans/CLAUDE.md`
- `docs/plans/14_authoritative-coordination-wave-1-rollout.md`
- `docs/plans/TEMPLATE.md`
- `meta-process.yaml`
- `scripts/meta/create_plan.py`
- `scripts/meta/check_coordination_claims.py --list`
- `~/projects/enforced-planning/PLANNING_OPERATING_MODEL.md`
- `~/projects/enforced-planning/STATIC_GRAPH_AND_RUNTIME_TRUTH.md`
- `~/projects/enforced-planning/templates/truth_surface_drift.yaml.example`
- `~/projects/enforced-planning/scripts/check_truth_surface_drift.py`
- `~/projects/enforced-planning/scripts/render_truth_surface_status.py`
- `~/projects/project-meta/scripts/meta/audit_governed_repo.py --repo-root . --json`

---

## Files Affected

- `docs/plans/15_truth-surface-adoption-pilot.md` (create/maintain)
- `docs/plans/CLAUDE.md` (modify)
- `docs/plans/TEMPLATE.md` (modify)
- `docs/ops/TRUTH_SURFACE_PILOT_STATUS.md` (create)
- `docs/ops/truth_surface_governed_audit.json` (create/update)
- `truth_surface_drift.yaml` (create)

---

## Plan

### Steps

1. Replace the stale local plan template with the current canonical plan shape so
   repo-local plan creation is trustworthy again.
2. Add a truthful local pilot status surface and a repo-local
   `truth_surface_drift.yaml` that references:
   - the local status surface
   - the local plan index
   - the shared active-work registry
   - a measured governed-audit artifact
3. Run the shared truth-surface validator and renderer from `enforced-planning`
   against the repo-local config.
4. Confirm at least one real drift case is detected. The expected first case is
   the shared registry still showing active work for completed Plan #12.
5. Record whether the pilot is strong enough to justify tighter default wiring,
   and document what still requires semantic LLM/agent review.

---

## Required Tests

### Pilot Checks (Must Pass)

| Command | Why |
|---------|-----|
| `python ~/projects/project-meta/scripts/meta/audit_governed_repo.py --repo-root . --json > docs/ops/truth_surface_governed_audit.json` | refreshes the measured audit surface used by the pilot |
| `python ~/projects/enforced-planning/scripts/check_truth_surface_drift.py --config truth_surface_drift.yaml --json` | proves repo-local truth-surface validation works |
| `python ~/projects/enforced-planning/scripts/render_truth_surface_status.py --config truth_surface_drift.yaml` | proves a generated current-state surface exists |
| `python scripts/meta/sync_plan_status.py --check` | keeps the local plan index truthful |
| `python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md docs/plans/15_truth-surface-adoption-pilot.md docs/ops/TRUTH_SURFACE_PILOT_STATUS.md` | catches local doc/link drift |
| `git diff --check` | proves the pilot slice is syntactically clean |

### New Tests

None. This pilot is a repo-local integration/evidence slice rather than a new
package behavior change.

---

## Acceptance Criteria

- [x] `prompt_eval` has a truthful repo-local pilot plan and status surface.
- [x] `truth_surface_drift.yaml` references real repo-local and shared runtime surfaces.
- [x] The validator runs successfully against the repo-local config.
- [x] At least one real drift case is detected and recorded.
- [x] The renderer emits a compact current-state summary that can replace freehand status prose.
- [x] The stale local plan template is fixed so future `create_plan.py` output is usable.
- [x] The pilot leaves a categorized backlog of what still needs semantic LLM/agent review.

---

## Open Questions

- [ ] Should `prompt_eval` eventually vendor or install the truth-surface scripts locally, or is a framework-path invocation acceptable for the pilot stage only?
- [ ] Should an `!! ACTIVE (no claim)` warning in canonical root become a hard pilot blocker once the truth-surface workflow is standardized?
- [ ] Which semantic drift classes in `prompt_eval` should remain advisory even if deterministic validation becomes default?
