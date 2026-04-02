# Plan #14: Authoritative coordination wave-1 rollout

**Status:** 🚧 In Progress
**Type:** implementation
**Priority:** Critical
**Blocked By:** None
**Blocks:** truthful DIGIMON prerequisite remediation follow-on

---

## Gap

**Current:** `prompt_eval` is already mechanically governed and worktree-ready,
but it still lacked the sanctioned local plan-coordination entrypoints required
by the authoritative coordination rollout:

- `scripts/meta/check_coordination_claims.py`
- `scripts/meta/create_plan.py`
- `scripts/meta/plan_reservations.py`

The governed audit already passed otherwise, so the rollout should stay tightly
bounded to that missing surface.

**Target:** `prompt_eval` exposes the sanctioned local plan-coordination
entrypoints, records the rollout under a local plan, and ends with governed
audit still green and no plan-coordination warnings.

**Why:** `prompt_eval` is the second measured wave-1 repo. Finishing it proves
the coordination rollout is repeatable across more than one governed consumer
instead of being a one-off `llm_client` success.

---

## References Reviewed

- `CLAUDE.md` - local governance and command contract
- `README.md` - package boundary and quick-start contract
- `docs/UNCERTAINTIES.md` - current repo boundary and open-question surface
- `docs/ops/CAPABILITY_DECOMPOSITION.md` - local capability ownership source of record
- `docs/plans/12_governed-baseline-repair-for-active-stack-candidacy.md` - governed baseline precedent
- `docs/plans/13_linkage-deepening-and-capability-ownership.md` - latest governed/capability rollout context
- `docs/plans/CLAUDE.md` - local plan index and status rules
- `scripts/CLAUDE.md` - script inventory and local meta helper surface
- `meta-process.yaml` - local coordination expectation and capability ownership
- `/home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/docs/plans/60_wave-1-authoritative-coordination-adoption-and-digimon-prerequisite-remediation.md`
  - cross-repo rollout contract driving this local slice
- `/home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/install_governed_repo.py`
  - sanctioned narrow bootstrap path
- `python /home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/audit_governed_repo.py --repo-root /home/brian/projects/prompt_eval_worktrees/plan-60-prompt-eval-coordination --json`
  - pre-bootstrap state: governed, but all three plan-coordination scripts missing
- same audit after bootstrap
  - post-bootstrap state: governed, no plan-coordination warnings

---

## Files Affected

- `docs/plans/14_authoritative-coordination-wave-1-rollout.md` (create/modify)
- `docs/plans/CLAUDE.md` (modify)
- `scripts/meta/check_coordination_claims.py` (create)
- `scripts/meta/complete_plan.py` (modify if closeout tooling is not venv-safe)
- `scripts/meta/create_plan.py` (create)
- `scripts/meta/plan_reservations.py` (create)

---

## Plan

### Steps

1. Bootstrap the sanctioned local plan-coordination entrypoints via the narrow
   installer mode.
2. Create this repo-local plan immediately after bootstrap and confirm the diff
   stays bounded to the three scripts plus plan/index files.
3. Re-run governed audit and local coordination smoke checks.
4. Commit the rollout as one rollback point with explicit verification evidence.
5. If formal closeout is blocked by non-venv-safe plan tooling, fix that
   tooling in-repo, rerun closeout, and keep the change bounded to the local
   coordination surface.

---

## Required Tests

### Existing Checks (Must Pass)

| Command | Why |
|---------|-----|
| `python /home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/audit_governed_repo.py --repo-root /home/brian/projects/prompt_eval_worktrees/plan-60-prompt-eval-coordination --json` | proves the repo remains governed and the plan-coordination warnings are gone |
| `python scripts/meta/check_coordination_claims.py --check --project prompt_eval --json` | proves the local coordination CLI runs in-repo |
| `python scripts/meta/create_plan.py --dry-run --title "coordination smoke" --no-fetch` | proves the local plan allocation path executes |
| `git diff --check` | proves the rollout slice is syntactically clean |

### New Tests

None. The rollout copies the already-tested sanctioned scripts from
`project-meta`; repo-local verification is smoke/audit based.

---

## Acceptance Criteria

- [ ] the only production files added in this slice are the three sanctioned
      plan-coordination scripts plus any bounded closeout-tooling fix required
      to use the repo-local `.venv`
- [ ] governed audit still returns `status=PASS` and `classification=governed`
- [ ] governed audit no longer reports missing plan-coordination scripts
- [ ] `python scripts/meta/check_coordination_claims.py --check --project prompt_eval --json` succeeds
- [ ] `python scripts/meta/create_plan.py --dry-run --title "coordination smoke" --no-fetch` succeeds
- [ ] the worktree ends this slice at a clean rollback commit

---

## Notes

- This rollout is intentionally bounded. Do not use this plan as license to
  widen `prompt_eval` governance or package scope.
- If closeout uses a repo-local `.venv`, create it locally rather than mutating
  a shared cross-project interpreter.
