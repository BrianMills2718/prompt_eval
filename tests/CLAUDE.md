# Tests

`prompt_eval` keeps its pytest suite flat under [`tests/`](.).

## Current Layout

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures, including isolated `llm_client` observability state |
| `test_runner.py` | Experiment execution and observability emission |
| `test_query.py` | Shared-backend reconstruction |
| `test_prompt_assets.py` | Prompt asset integration |
| `test_optimize.py` | Search strategies |
| `test_evaluators.py` | Evaluator factories and scoring helpers |
| `test_stats.py` | Statistical comparison |
| `test_store.py` | Local JSON persistence |
| `test_mcp_server.py` | Optional MCP server behavior |
| `test_experiment.py` | Core model contracts |

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_runner.py tests/test_query.py -q
pytest -q --collect-only
```

## Notes

- There is no unit/integration/e2e directory split in this repo today.
- The suite should be read by behavior area, not by artificial test tiers.
- Shared-observability tests rely on the isolation fixture in `conftest.py`;
  do not bypass it with ad hoc local state.
