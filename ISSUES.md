# Issues

Observed problems, concerns, and technical debt. Items start as **unconfirmed**
observations and get triaged through investigation into confirmed issues, plans,
or dismissed.

**Last reviewed:** 2026-03-19

---

## Status Key

| Status | Meaning | Next Step |
|--------|---------|-----------|
| `unconfirmed` | Observed, needs investigation | Investigate to confirm/dismiss |
| `monitoring` | Confirmed concern, watching for signals | Watch for trigger conditions |
| `confirmed` | Real problem, needs a fix | Create a plan |
| `planned` | Has a plan (link to plan) | Implement |
| `resolved` | Fixed | Record resolution |
| `dismissed` | Investigated, not a real problem | Record reasoning |

---

## Unconfirmed

(Add observations here with enough context to investigate later)

### ISSUE-001: (Title)

**Observed:** (date)
**Status:** `unconfirmed`

(What was observed. Why it might be a problem.)

**To investigate:** (What would confirm or dismiss this.)

---

## Monitoring

(Items confirmed as real but not yet urgent. Include trigger conditions.)

### ISSUE-001: Inherited Plan Tooling Assumes Repo Scaffolding That Does Not Exist

**Observed:** 2026-03-18
**Status:** `monitoring`

`prompt_eval` inherited plan-process scripts from a broader meta-process toolkit.
Some of that tooling still assumes a doc-coupling validator and a formal
`tests/e2e/` hierarchy that this repo does not currently maintain. The active
docs surface now treats those scripts as secondary rather than canonical, but
the mismatch still exists in the tooling itself.

**Trigger to act:** if the repo starts relying on `complete_plan.py` or broader
meta-process automation as a required workflow, localize or simplify that
tooling first so it matches `prompt_eval`'s real structure.

## Confirmed

(Items that need a fix but don't have a plan yet.)

## Resolved

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| ISSUE-002 | Model Governance Drift In Package Defaults | Public experiment and optimizer surfaces now require explicit subject-model declaration; judge helpers keep only documented internal convenience defaults. | 2026-03-19 |
| ISSUE-003 | Internal LLM Prompts Are Still Embedded In Python | Judge and instruction-search helper prompts now live as local YAML/Jinja templates rendered via `llm_client`, rather than inline Python strings. | 2026-03-19 |
| ISSUE-004 | Statistical Engine Remains Hand-Rolled | `prompt_eval.stats` now uses SciPy-backed Welch and bootstrap inference for the current lightweight comparison API; the remaining paired/clustered design question is tracked separately in `docs/UNCERTAINTIES.md`. | 2026-03-19 |
| ISSUE-005 | Prompt Asset Policy And Package Scope Were Still Ambiguous | The long-term boundary is now explicit: prompt assets are preferred when available, inline messages remain a permanent supported input, and `prompt_eval` stays prompt-centric. | 2026-03-19 |
| ISSUE-006 | Stronger Comparison Needed An Explicit Repeated-Measures Mode | `compare_variants()` now keeps pooled comparison as the default and also supports `comparison_mode=\"paired_by_input\"` for matched-input inference. | 2026-03-19 |

---

## Dismissed

| ID | Description | Why Dismissed | Date |
|----|-------------|---------------|------|
| - | - | - | - |

---

## How to Use This File

1. **Observe something off?** Add under Unconfirmed with context and investigation steps
2. **Investigating?** Update the entry with findings, move to appropriate status
3. **Confirmed and needs a fix?** Create a plan, link it, move to Confirmed/Planned
4. **Not actually a problem?** Move to Dismissed with reasoning
5. **Watching a concern?** Move to Monitoring with trigger conditions
