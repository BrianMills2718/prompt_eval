# Plan 13: Linkage Deepening And Capability Ownership

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** 12
**Blocks:** stricter governed-repo enforcement and truthful shared capability coverage

---

## Gap

**Current:** Plan 12 made `prompt_eval` mechanically governed and opted it into
sanctioned worktree coordination, but two important governance surfaces remain
shallow:

1. `scripts/relationships.yaml` is still the bootstrap-default scaffold with no
   actionable governance or coupling rules
2. `prompt_eval` does not yet declare repo-local `capability_ownership` even
   though it is a shared-infrastructure repo with known ecosystem consumers

Top-level docs also still contain stale “current active program” language that
predates the governed-baseline wave.

**Target:** give `prompt_eval` one small actionable linkage graph, one dedicated
repo-local capability ownership source of record, and one truthful shared
ownership/discovery footprint in `project-meta`.

**Why:** mechanical governance alone is not enough if agents still see only a
bootstrap scaffold and no ownership source of truth.

---

## References Reviewed

- `CLAUDE.md`
- `README.md`
- `docs/UNCERTAINTIES.md`
- `docs/plans/CLAUDE.md`
- `docs/plans/TEMPLATE.md`
- `scripts/relationships.yaml`
- `meta-process.yaml`
- `KNOWLEDGE.md`
- `~/projects/project-meta/docs/plans/45_prompt-eval-linkage-deepening-and-capability-ownership.md`
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_CONTRACT.md`
- `~/projects/project-meta/docs/ops/CANONICAL_SOURCES_AND_CONSUMER_REPOS.md`
- `~/projects/project-meta/vision/CAPABILITY_OWNERSHIP_AND_MIGRATION_V0.md`
- `~/projects/project-meta/scripts/capability_ownership_registry.yaml`

---

## Pre-Made Decisions

1. `prompt_eval` will declare capability ownership through a dedicated local
   doc, not just README prose.
2. The first linkage deepening wave will stay small and actionable; no attempt
   at full file-by-file coverage.
3. The first shared capability row for `prompt_eval` will describe the repo’s
   core prompt-evaluation and optimization role, not every subfeature.
4. README truthfulness is part of this wave because the current “active
   program” pointer is stale.

---

## Files Affected

- `docs/plans/13_linkage-deepening-and-capability-ownership.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `README.md` (modify)
- `scripts/CLAUDE.md` (modify)
- `scripts/relationships.yaml` (modify)
- `meta-process.yaml` (modify)
- `docs/ops/CAPABILITY_DECOMPOSITION.md` (create)
- `KNOWLEDGE.md` (modify)

---

## Plan

### Step 1: Deepen the local linkage graph

- replace bootstrap-only `scripts/relationships.yaml` with a small actionable
  set of governance/coupling/architecture rules
- make the rules point agents at the actual repo truth surfaces

### Step 2: Correct stale top-level docs

- update README/current-state text so it no longer claims Plan 11 is the
  current active program
- make the repo’s current governance state visible in the doc surface

### Step 3: Declare repo-local capability ownership

- create `docs/ops/CAPABILITY_DECOMPOSITION.md`
- enable `meta_process.capability_ownership` and point it at that source

### Step 4: Hand off to shared registry alignment

- leave the repo ready for the matching `project-meta` registry and canonical
  sources updates

---

## Required Tests

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python ~/projects/project-meta/scripts/meta/audit_governed_repo.py --repo-root . --json` | local governed/capability truth stays explicit |
| `python scripts/meta/sync_plan_status.py --check` | plan index stays truthful |
| `python scripts/check_markdown_links.py README.md CLAUDE.md docs/plans/CLAUDE.md docs/plans/13_linkage-deepening-and-capability-ownership.md docs/ops/CAPABILITY_DECOMPOSITION.md KNOWLEDGE.md` | updated doc graph stays navigable |

---

## Acceptance Criteria

- [x] `scripts/relationships.yaml` is no longer bootstrap-minimal
- [x] README no longer points to a stale “current active program”
- [x] `prompt_eval` declares `capability_ownership` with a real local source of
      record
- [x] repo-local docs make the shared-infrastructure role discoverable without
      needing to open `project-meta` first

---

## Notes

- this plan intentionally stops at the repo boundary; the matching shared
  registry and canonical-source updates happen under `project-meta` Plan 45

## Verification Notes

- `python /home/brian/projects/project-meta_worktrees/plan-41-makefile-meta-sync/scripts/meta/audit_governed_repo.py --repo-root . --json`
  - `status = PASS`
  - `relationships_yaml.linkage.status = actionable`
  - `capability_ownership.declared = true`
- `python scripts/meta/sync_plan_status.py --check`
- `python scripts/check_markdown_links.py README.md CLAUDE.md docs/plans/CLAUDE.md docs/plans/13_linkage-deepening-and-capability-ownership.md docs/ops/CAPABILITY_DECOMPOSITION.md KNOWLEDGE.md scripts/CLAUDE.md scripts/relationships.yaml`
- `python scripts/meta/check_agents_sync.py --check`

## Follow-On Maintenance Note

The linkage surface established here also covers later bounded workflow-helper
replays. In particular, the local workflow contract now expects rename-safe
`scripts/meta/merge_pr.py` cleanup so branch-renamed worktrees are removed by
discovered path rather than a reconstructed branch-derived path.
The same follow-on maintenance rule now covers push-safety, review-claim,
concern-routing, and worktree-aware dead-code helper updates when the shared
sanctioned-worktree contract evolves upstream.
It also now covers repo-local publish-hook alignment: `make publish-check`
tracks the governed push/dead-code baseline, while any stricter repo-local gate
must live behind `publish-check-extra` rather than being assumed by the shared
framework contract.
`prompt_eval` now exercises that stricter path directly: its
`publish-check-extra` target runs `make check`, and the Makefile resolves the
canonical repo `.venv` plus sibling `llm_client` source so the hook and the
interactive worktree path enforce the same mypy contract.
