# Truth-Surface Pilot Status

**Status:** In Progress
**Plan:** `docs/plans/15_truth-surface-adoption-pilot.md`
**Governed audit status:** PASS

## Purpose

This is the local status surface for the first `prompt_eval` truth-surface
adoption pilot. It exists so the shared validator has a repo-local tracker-like
surface to compare against the plan index, shared registry, and measured audit
artifact.

## Current State

- `prompt_eval` remains mechanically governed.
- The local governed audit snapshot in `docs/ops/truth_surface_governed_audit.json`
  currently reports `status=PASS` and `classification=governed`.
- The shared validator and renderer now run successfully against
  `truth_surface_drift.yaml`.
- The initial unscoped replay found three real issues, but two were unrelated
  ecosystem noise from other repos.
- After the scoped replay, the repo-local output is now one actionable local
  finding:
  - consumed reservation points to a missing old `prompt_eval` worktree plan file
- The pilot also exposed two framework-level usability issues that were fixed in
  `enforced-planning` during execution:
  - config-declared surface paths were resolved relative to caller cwd instead of config location
  - the status renderer assumed execution from the framework repo root
- The scoped replay additionally proved that canonical repo identity derivation
  is enough to keep unrelated `project-meta` and `agentic_scaffolding` registry
  drift out of the repo-local result.

## Current Recommendation

Repo-local scoped validation should be the default operator view, but workflow
wiring should still remain advisory for now.

Reason:
- scoped validation materially improves the local signal
- one real local historical reservation issue still remains to clean up
- semantic review is still needed for stale or misleading prose that exact rules
  do not capture

## Next Action

Close the local pilot slice truthfully, then feed the measured findings back
into the shared framework as:
- deterministic coverage now
- shared-registry hygiene backlog
- semantic-review backlog later
