# Operational Knowledge — prompt_eval

Shared findings from all agent sessions. Any agent brain can read and append.
Human-reviewed periodically.

## Findings

<!-- Append new findings below this line. Do not overwrite existing entries. -->
<!-- Format: ### YYYY-MM-DD — {agent} — {category}                          -->
<!-- Categories: bug-pattern, performance, schema-gotcha, integration-issue, -->
<!--             workaround, best-practice                                   -->
<!-- Agent names: claude-code, codex, openclaw                               -->

---

### 2026-04-01 — codex — best-practice

**`prompt_eval` can reach the mechanical governed baseline through the shared
governed-repo installer in one bounded pass.**

Plan 12 proved that the missing governed-baseline surfaces were not
`prompt_eval`-specific snowflakes. The shared installer added the missing
machine-readable governance file, validators, AGENTS renderer inputs, and local
worktree-coordination surface without requiring repo-local hand copying.

Practical rule:

- use the shared installer first when the audit gaps are mechanical
- only hand-repair after the installer proves an explicit bounded gap

### 2026-04-01 — codex — integration-issue

**Disposable proof worktrees cannot reuse the active plan claim; use an
unscoped temporary claim unless the claim model later grows a proof/feature
lane.**

Once Plan 12 claimed the active `prompt_eval` worktree, a second
`make worktree ... PLAN=12` proof branch was correctly blocked by the claim
registry. The sanctioned proof still succeeded with an unscoped temporary claim
after the shared `worktree-remove` path was fixed to release claims on remove.

Practical rule:

- keep the active implementation worktree claimed under its plan
- use an unscoped temporary proof branch when the current claim model allows
  only one active claim per plan
- treat the warning as ergonomic debt, not as proof that sanctioned worktree
  coordination failed

### 2026-04-01 — codex — best-practice

**Once an active-stack repo reaches the mechanical governed baseline, replace
bootstrap-default linkage with a real ownership and coupling surface before
treating it as durable shared infrastructure.**

Plan 12 proved that `prompt_eval` could be repaired mechanically, but the repo
still looked shallow to agents while `scripts/relationships.yaml` stayed
bootstrap-minimal and no repo-local capability source of record existed. The
next truthful step was not more installer work; it was a small actionable
linkage graph plus a local capability decomposition doc.

Practical rule:

- use the shared installer to repair mechanical gaps
- then deepen linkage and ownership locally before calling the repo a stable
  shared capability source

### 2026-04-02 — codex — best-practice

**A governed consumer pilot does not automatically become `prompt_eval`
product scope.**

The semantic truth-surface review pilot used `prompt_eval` as a real consumer
repo to prove shared governance tooling, but the clean architecture decision
was to keep ownership in `enforced-planning`, not to turn `prompt_eval` into
the canonical owner of truth-surface review. The coordination registry should
therefore record that pilot as historical consumer evidence unless a later
explicit plan reopens the package boundary.

Practical rule:

- use `prompt_eval` to prove evaluation/governance tooling on a real consumer
  repo when that is the fastest way to get evidence
- do not infer permanent repo ownership from that pilot alone
- if registry lineage says a consumer pilot landed canonically but the repo
  boundary docs do not, reconcile the lineage toward historical-unlanded rather
  than widening package scope by accident

### 2026-04-02 — codex — integration-issue

**Scoped truth-surface validation in `prompt_eval` exposes historical consumed
reservation paths after proof/worktree cleanup, and semantic review helps show
how that stale runtime state becomes misleading prose.**

Plan 15 adopted the repo-local truth-surface validator and optional semantic
review. The first scoped deterministic run was not clean: it found two consumed
reservations that still pointed at deleted prompt-eval worktree plan files. The
semantic layer then added useful advisory context by flagging that the plan
index's current-summary prose looked misleading once those missing artifacts
were taken into account.

Practical rule:

- treat missing historical consumed-reservation plan files as shared registry
  hygiene debt, not as a repo-local code bug
- keep the repo-local rendered truth-surface status because it makes that drift
  visible to operators without reading raw registry YAML
- use semantic findings to propose future deterministic promotions, but keep
  them advisory until the shared framework owns the invariant

### 2026-04-02 — codex — bug-pattern

**`prompt_eval`'s pre-commit doc-coupling hook currently checks branch delta
against `origin/main`, not the staged set, so it can false-fail on older branch
changes even when the current staged docs are correct.**

During the rename-safe `merge_pr.py` replay, `python scripts/meta/check_doc_coupling.py --staged`
passed once the coupled docs were staged, but the repo's `hooks/pre-commit`
still failed because it runs `python scripts/check_doc_coupling.py --strict`
without `--staged`. On a branch already ahead of `origin/main` from an earlier
Makefile maintenance commit, that hook reported the old Makefile delta instead
of the staged helper replay.

Practical rule:

- treat `python scripts/meta/check_doc_coupling.py --staged` as the truthful
  pre-commit check for the current index
- if the hook fails on branch history instead of staged changes, verify the
  staged set manually, record the hook false-negative, and fix the hook in a
  later bounded governance slice rather than pretending the docs are missing
