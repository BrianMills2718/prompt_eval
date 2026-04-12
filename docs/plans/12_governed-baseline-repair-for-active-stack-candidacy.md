# Plan 12: Governed Baseline Repair For Active-Stack Candidacy

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** truthful sanctioned worktree opt-in for `prompt_eval`

**Verified:** 2026-04-01T17:19:00Z
**Verification Evidence:**
```yaml
completed_by: codex
timestamp: 2026-04-01T17:19:00Z
checks:
  governed_audit: PASS (classification=governed)
  plan_status_sync: PASS
  plan_validation: PASS
  markdown_links: PASS
  tests: 186 passed in 5.86s
notes:
  - installer-driven repair landed the governed baseline in one pass
  - sanctioned worktree opt-in was then enabled explicitly via meta-process.yaml
  - disposable worktree create/remove proof passed after the shared Makefile block was resynced to release claims on remove
```

---

## Gap

**Current:** `prompt_eval` has a strong local planning and package boundary
surface, but it still audits `partial` against the shared governed-repo
contract. The current mechanical gaps are explicit:

- missing `scripts/relationships.yaml`
- `AGENTS.md` is not refreshable because the machine-readable governance input
  is missing
- missing local validators:
  - `scripts/meta/file_context.py`
  - `scripts/meta/validate_plan.py`
  - `scripts/meta/check_doc_coupling.py`
  - `scripts/check_markdown_links.py`

`prompt_eval` also keeps worktree coordination disabled today, which is
truthful while the governed baseline is incomplete.

**Target:** repair the governed-baseline contract in a clean worktree using the
shared installer path first, then re-audit and decide whether this repo is now
ready to become a sanctioned worktree candidate.

**Why:** `prompt_eval` should join the active stack by satisfying the same
mechanical governed contract as the other repos, not through special-case
exceptions.

---

## References Reviewed

- `CLAUDE.md`
- `README.md`
- `docs/plans/CLAUDE.md`
- `docs/plans/TEMPLATE.md`
- `meta-process.yaml`
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_CONTRACT.md`
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_ROLLOUT_TIERS.md`
- `~/projects/project-meta/docs/plans/44_prompt-eval-governed-baseline-repair-and-candidacy.md`
- `~/projects/project-meta/scripts/meta/audit_governed_repo.py`
- `~/projects/project-meta/scripts/meta/install_governed_repo.py`

---

## Pre-Made Decisions

1. Use the shared governed-repo installer as the primary repair mechanism; do
   not hand-copy each validator unless the installer proves insufficient.
2. Execute all repo-local changes in a clean linked worktree, not the dirty
   main checkout.
3. Keep package/runtime behavior out of scope. This wave is for governed-repo
   mechanics only.
4. Governed-baseline repair and sanctioned worktree opt-in are separate gates:
   becoming `governed` comes first.
5. If the repair still leaves the repo `partial`, record the residual blockers
   explicitly instead of widening scope.

---

## Files Affected

- `docs/plans/12_governed-baseline-repair-for-active-stack-candidacy.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `AGENTS.md` (modify via renderer)
- `scripts/relationships.yaml` (create)
- `scripts/check_markdown_links.py` (create)
- `scripts/meta/file_context.py` (create)
- `scripts/meta/validate_plan.py` (create)
- `scripts/meta/check_doc_coupling.py` (create)
- other bounded `install_governed_repo.py` sync outputs as needed
- `meta-process.yaml` (modify only if candidacy phase is reached)
- `Makefile` (modify only if candidacy phase is reached or installer syncs the sanctioned block)
- `KNOWLEDGE.md` (modify)

---

## Plan

### Step 1: Make the local plan surface truthful

- add this plan and index it as the current repo-local implementation lane
- keep the repo’s “current default next step” aligned with the governed repair
  work rather than the old “no active implementation plan remains” state

### Step 2: Run installer-driven governed-baseline repair

- use `~/projects/project-meta/scripts/meta/install_governed_repo.py --write`
  against this clean worktree
- let the shared installer create/sync the missing governed-baseline surfaces
- regenerate `AGENTS.md` through the shared renderer path, not by hand

### Step 3: Re-audit and do one bounded followthrough

- rerun the governed-repo audit
- if small residual mechanical gaps remain, fix them in one bounded pass
- if the repo still audits `partial`, stop and record the blocker truthfully

### Step 4: Decide sanctioned worktree candidacy

- only if the audit flips to `governed`, decide whether to enable:
  - `claims.enabled: true`
  - `worktrees.enabled: true`
- if opt-in is enabled, prove the disposable create/remove flow

---

## Required Tests

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python ~/projects/project-meta/scripts/meta/audit_governed_repo.py --repo-root . --json` | governed-baseline truth is explicit |
| `python scripts/meta/sync_plan_status.py --check` | local plan index stays truthful |
| `pytest tests/ -q` | governance sync does not quietly break repo-local tests |

### Repo Contract Checks

| Command | Why |
|---------|-----|
| `python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md` | shared link validator is actually wired and runnable locally |
| `python scripts/meta/validate_plan.py --plan-file docs/plans/12_governed-baseline-repair-for-active-stack-candidacy.md` | local plan validator works after the repair |

---

## Acceptance Criteria

- [x] repo-local plan surface is truthful about the governed-baseline work
- [x] the shared installer lands the missing governed-baseline files or proves a
      bounded gap in shared tooling
- [x] `AGENTS.md` becomes refreshable from canonical governance inputs
- [x] the repo ends with an explicit audited outcome: `governed`
- [x] if candidacy is reached, the worktree opt-in decision is explicit and
      proven rather than assumed

---

## Notes

- the main checkout’s untracked local `docs/plans/10_ci_and_hygiene.md` is not
  part of this clean worktree and should not be silently absorbed here

## Execution Notes

- the shared governed-repo installer repaired the missing `prompt_eval`
  governed-baseline surfaces in one bounded write pass
- the repaired worktree re-audited as mechanically `governed`
- `claims.enabled: true` plus `worktrees.enabled: true` were then enabled
  explicitly in `meta-process.yaml` to make sanctioned worktree adoption
  truthful rather than implicit
- disposable create/remove proof succeeded after the shared sanctioned
  `worktree-remove` path was updated to release claims after removal
- because the active worktree already holds Plan 12, the disposable proof branch
  uses an unscoped temporary claim; this is truthful but ergonomically noisy
  and should be treated as a future coordination UX follow-on, not a blocker
- later shared-helper maintenance may still replay into this repo without
  reopening Plan 12. The current bounded example is rename-safe `merge_pr.py`
  cleanup so branch-renamed worktrees release and remove correctly.
- that same governed-helper replay model now also covers push-safety,
  review-claim, and concern-routing surfaces plus worktree-aware dead-code
  interpreter reuse; these are workflow-consumer upgrades, not reopened package
  scope.
- the governed helper replay model now also covers repo-local `hooks/pre-push`
  plus `make publish-check`, with stricter repo-specific publish checks living
  behind an optional `publish-check-extra` target instead of assuming
  `make check` is publish-clean in every governed consumer.
