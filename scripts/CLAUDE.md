# Scripts

These scripts are small repo-local helpers for plan maintenance and workflow
automation.

## Canonical Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_markdown_links.py` | Verify local documentation links |
| `scripts/check_doc_coupling.py` | Verify coupled docs were updated with linked source changes |
| `scripts/check_truth_surface_drift.py` | Run deterministic scoped truth-surface drift validation |
| `scripts/render_truth_surface_status.py` | Render the compact truth-surface operator status |
| `scripts/review_truth_surface_semantic.py` | Run optional advisory semantic review on truth surfaces |
| `scripts/meta/sync_plan_status.py` | Verify or sync plan index status |
| `scripts/meta/check_plan_blockers.py` | Fail loud when a blocked plan is being bypassed |
| `scripts/meta/check_plan_tests.py` | Inspect or run the tests declared by a plan |
| `scripts/meta/parse_plan.py` | Parse plan structure for other tooling |
| `scripts/meta/validate_plan.py` | Check plan structure and required sections |

## Secondary Helpers

These exist, but they are not part of the canonical `prompt_eval` execution
path:

- `scripts/meta/generate_quiz.py`
- `scripts/meta/merge_pr.py`
- `scripts/meta/pr_auto.py`
- `scripts/meta/complete_plan.py`

Some of those helpers were inherited from broader meta-process tooling and may
still assume conventions that `prompt_eval` does not fully enforce.

## Current Limits

- There is no dedicated standalone relationships-quality validator; linkage
  quality is currently enforced through the shared governed-repo audit and the
  doc-coupling helpers.
- Truth-surface semantic review is advisory-only. Do not treat it as the same
  certainty class as deterministic validator failures.
- There is no formal `tests/e2e/` hierarchy.
- `complete_plan.py` comes from a broader tooling pattern and still assumes
  repo conventions that `prompt_eval` does not fully enforce. Do not treat it
  as the canonical execution path for this repo.

Run `python <script> --help` for details.
