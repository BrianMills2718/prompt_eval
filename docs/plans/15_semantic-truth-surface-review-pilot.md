# Plan #15: Semantic Truth-Surface Review Pilot

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** 14
**Blocks:** default semantic-review adoption guidance for other governed consumers

---

## Gap

**Current:** `prompt_eval` already proved the deterministic truth-surface pilot,
but it does not yet consume the new semantic review tooling from
`enforced-planning`. The repo currently has no local truth-surface config, no
rendered status surface, and no measured evidence about whether semantic review
adds useful signal beyond the scoped deterministic checks.

**Target:** `prompt_eval` adopts the repo-local truth-surface and semantic
review tooling in one bounded slice, runs both deterministic and semantic
checks against its actual local surfaces, and records which findings remain
advisory versus candidates for future deterministic promotion.

**Why:** `prompt_eval` was already the first deterministic governed-repo pilot.
Using it for the first semantic pilot keeps the signal comparable and proves
that the hybrid model works in a real consumer repo rather than only in the
framework worktree.

---

## References Reviewed

- `CLAUDE.md` - local governance and verification contract
- `README.md` - current repo boundary and governed-surface docs map
- `KNOWLEDGE.md` - prior governed-baseline and linkage findings
- `docs/UNCERTAINTIES.md` - current architecture/open-question surface
- `docs/plans/13_linkage-deepening-and-capability-ownership.md` - latest local governance/capability precedent
- `docs/plans/14_authoritative-coordination-wave-1-rollout.md` - current local coordination baseline
- `docs/plans/CLAUDE.md` - local plan index and next-step contract
- `scripts/CLAUDE.md` - current script inventory
- `scripts/relationships.yaml` - local static planning graph
- `meta-process.yaml` - current governance configuration
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/PLANNING_OPERATING_MODEL.md` - canonical methodology
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/STATIC_GRAPH_AND_RUNTIME_TRUTH.md` - static/runtime split and semantic-review role
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/docs/plans/06_governed-repo-truth-surface-adoption-pilot.md` - deterministic pilot precedent
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/docs/plans/07_llm-semantic-truth-surface-review.md` - semantic-review contract
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/docs/plans/11_semantic-truth-surface-review-execution-sprint.md` - first implementation-slice defaults
- `../enforced-planning_worktrees/plan-10-framework-reconciliation/install.sh` - sanctioned installation path for the portable tooling

---

## Files Affected

- `docs/plans/15_semantic-truth-surface-review-pilot.md` (update during execution)
- `docs/plans/CLAUDE.md` (modify)
- `README.md` (modify)
- `scripts/CLAUDE.md` (modify)
- `KNOWLEDGE.md` (modify)
- `scripts/check_truth_surface_drift.py` (create via installer)
- `scripts/render_truth_surface_status.py` (create via installer)
- `scripts/review_truth_surface_semantic.py` (create via installer)
- `scripts/truth_surface_semantic_models.py` (create via installer)
- `prompts/truth_surface_semantic_review.yaml` (create via installer)
- `truth_surface_drift.yaml.example` (create via installer)
- `truth_surface_drift.yaml` (create)
- `docs/ops/TRUTH_SURFACE_STATUS.md` (create)

---

## Pre-Made Decisions

1. This pilot will use the sanctioned `enforced-planning` installer rather than repo-local hand copying.
2. Repo-local validation will default to scoped mode for `prompt_eval`; unscoped ecosystem review remains optional and advisory.
3. Semantic review will remain opt-in and advisory-only in this slice.
4. The review prompt will consume deterministic rendered status plus bounded evidence snippets; it will not do whole-repo raw-file review.
5. The pilot will commit a rendered Markdown status surface, not raw machine-output artifacts, as the durable repo-local operator view.
6. If the repo-local `.venv` is missing, this slice may create it locally; it will not rely on a shared cross-project interpreter.

---

## Plan

### Phase A: Adopt The Portable Tooling

Success criteria:
- the sanctioned truth-surface and semantic-review scripts exist locally
- the install path is documented truthfully in repo docs

Steps:
1. Run the `enforced-planning` installer in minimal mode against this worktree.
2. Confirm only the expected truth-surface files land locally.
3. Update repo-local docs if the installed surface changes the script inventory or recommended workflow.

### Phase B: Define The Repo-Local Truth Surface

Success criteria:
- a real `truth_surface_drift.yaml` exists and resolves locally
- scoped validation targets `prompt_eval` only
- the config points at real tracker, plan index, registry, and measured audit surfaces

Steps:
1. Create `truth_surface_drift.yaml` from the framework template.
2. Configure the repo name scope, tracker/index paths, and local audit command.
3. Add at least one useful repo-local rule beyond generic file/path checks if the current surfaces justify it.

### Phase C: Run Deterministic And Semantic Review

Success criteria:
- deterministic validation succeeds or fails in a way that exposes a real, actionable drift case
- semantic review runs end-to-end with structured output
- a rendered status surface exists that shows deterministic and semantic findings separately

Steps:
1. Run deterministic validation and capture the result.
2. Run semantic review with explicit `task`, `trace_id`, and `max_budget` through the installed tooling.
3. Render the merged status into `docs/ops/TRUTH_SURFACE_STATUS.md`.

### Phase D: Record Findings And Close The Pilot

Success criteria:
- plan/index/README reflect the new local capability truthfully
- `KNOWLEDGE.md` records the durable repo-local finding
- the repo ends at a clean rollback commit

Steps:
1. Update the plan with measured findings, including any remaining advisory-only drift classes.
2. Update the plan index default next step based on the pilot result.
3. Record the durable pilot lesson in `KNOWLEDGE.md`.

---

## Required Tests

### New Tests (TDD)

None. This pilot adopts already-tested portable tooling and verifies it through repo-local integration commands and measured outputs.

### Existing Tests (Must Pass)

| Command | Why |
|---------|-----|
| `python scripts/meta/sync_plan_status.py --check` | local plan index must stay truthful |
| `python scripts/check_markdown_links.py README.md CLAUDE.md docs/plans/CLAUDE.md docs/plans/15_semantic-truth-surface-review-pilot.md docs/ops/TRUTH_SURFACE_STATUS.md KNOWLEDGE.md scripts/CLAUDE.md` | updated doc graph must remain navigable |
| `python scripts/check_truth_surface_drift.py --config truth_surface_drift.yaml --json` | proves the local deterministic validator runs against real repo surfaces |
| `python scripts/review_truth_surface_semantic.py --config truth_surface_drift.yaml --model gemini/gemini-2.5-flash --max-budget 0.10 --trace-id prompt_eval/semantic_truth_surface_pilot --output-json /tmp/prompt_eval_semantic_truth_surface_review.json` | proves the local semantic-review path works end to end |
| `python scripts/render_truth_surface_status.py --config truth_surface_drift.yaml --semantic-json /tmp/prompt_eval_semantic_truth_surface_review.json` | proves deterministic and semantic findings render together without collapsing certainty |
| `git diff --check` | ensures the slice is syntactically clean |

---

## Acceptance Criteria

- [x] `prompt_eval` installs the sanctioned truth-surface and semantic-review tooling locally without widening repo scope
- [x] `truth_surface_drift.yaml` exists and validates real local surfaces in scoped mode
- [x] the pilot produces one rendered repo-local truth-surface status output that keeps semantic findings advisory-only
- [x] README and script inventory docs mention the new local operator surface
- [x] `KNOWLEDGE.md` records the durable result of the pilot
- [x] the worktree ends with a clean verified rollback commit

---

## Measured Findings

- The sanctioned `enforced-planning` installer initially rejected git worktrees because it only accepted `.git/` directories; the framework was fixed to accept `.git` files and then the prompt-eval pilot proceeded normally.
- The bounded prompt-eval adoption surface is: `scripts/check_truth_surface_drift.py`, `scripts/render_truth_surface_status.py`, `scripts/review_truth_surface_semantic.py`, `scripts/truth_surface_semantic_models.py`, `prompts/truth_surface_semantic_review.yaml`, `truth_surface_drift.yaml.example`, and the repo-local `truth_surface_drift.yaml` config.
- The first scoped deterministic run found two real historical consumed-reservation drifts, both pointing at deleted prompt-eval worktree plan files.
- The final semantic review run succeeded with `gemini/gemini-2.5-flash`, cost approximately `$0.0021`, and added three advisory findings on top of the deterministic failures:
  - `misleading_summary` (`warn`, promotion candidate) for Plan 15 completion prose
  - `stale_prose` (`warn`, promotion candidate) for the current-default-next-step text
  - `misleading_summary` (`warn`, promotion candidate) for Plan 14 completion prose
- The durable operator surface is now `docs/ops/TRUTH_SURFACE_STATUS.md`, which renders deterministic failures and semantic findings together without collapsing certainty classes.
- The next likely follow-on belongs to the shared framework and registry hygiene layer, not to prompt-eval package code.

## Notes

- No material uncertainty blocked this slice. The framework had already decided the semantic-review contract; this repo-local work was an adoption and evidence pass.
- The measured failures are real but historical. They expose registry-hygiene debt rather than a prompt-eval runtime regression.
