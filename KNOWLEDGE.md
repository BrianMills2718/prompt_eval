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
