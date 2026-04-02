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
- The first measured findings are all external/shared coordination drift, not
  `prompt_eval` package drift:
  - consumed reservation points to a missing old `prompt_eval` worktree plan file
  - consumed reservation points to a missing old `project-meta` worktree plan file
  - active work still references completed `agentic_scaffolding` Plan #8
- The pilot also exposed two framework-level usability issues that were fixed in
  `enforced-planning` during execution:
  - config-declared surface paths were resolved relative to caller cwd instead of config location
  - the status renderer assumed execution from the framework repo root

## Current Recommendation

Default workflow wiring should remain advisory for now.

Reason:
- the validator is useful and catches real drift immediately
- but the first measured failures are dominated by shared-registry hygiene, not
  repo-local contradictions
- semantic review is still needed for stale or misleading prose that exact rules
  do not capture

## Next Action

Close the local pilot slice truthfully, then feed the measured findings back
into the shared framework as:
- deterministic coverage now
- shared-registry hygiene backlog
- semantic-review backlog later
