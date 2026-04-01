# Plan 12: Governed Baseline Repair For Active-Stack Candidacy

**Status:** 🚧 In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** truthful sanctioned worktree opt-in for `prompt_eval`

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

- [ ] repo-local plan surface is truthful about the governed-baseline work
- [ ] the shared installer lands the missing governed-baseline files or proves a
      bounded gap in shared tooling
- [ ] `AGENTS.md` becomes refreshable from canonical governance inputs
- [ ] the repo ends with an explicit audited outcome: `governed` or `partial`
- [ ] if candidacy is reached, the worktree opt-in decision is explicit and
      proven rather than assumed

---

## Notes

- current known uncertainty is whether the shared installer will be sufficient
  without a bounded manual followthrough
- the main checkout’s untracked local `docs/plans/10_ci_and_hygiene.md` is not
  part of this clean worktree and should not be silently absorbed here
