# Plan #10: CI and Governance Hygiene

**Status:** Complete
**Type:** maintenance
**Priority:** Medium
**Blocked By:** None

---

## Gap

No CI — tests only run locally. No REQUIREMENTS.md. Recent relocation work
(scoring + experiment_eval from llm_client) may have loose ends.

## Steps

1. Add `.github/workflows/test.yml` (pytest + mypy + ruff, py3.10 + py3.12)
2. Verify scoring/experiment_eval relocation is complete (no circular imports, tests pass)
3. Push and verify CI green

## Acceptance Criteria

- [x] GitHub Actions workflow exists and runs on push/PR
- [x] All 185+ tests pass in CI (186 pass locally; CI fix via llm_client data_contracts shim)
- [x] No circular imports between prompt_eval and llm_client
- [x] CI green on both Python versions
