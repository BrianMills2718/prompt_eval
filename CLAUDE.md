# prompt_eval (v0.1.0)

## What This Project Does

A/B prompt testing and evaluation library. Define prompt variants, run them against test inputs with an LLM, and compare results with statistical rigor (bootstrap CI or Welch's t-test).

## Architecture

```
prompt_eval/
  experiment.py   # Pydantic data models: Experiment, PromptVariant, ExperimentInput, Trial, EvalResult, VariantSummary
  runner.py       # Async experiment runner: calls LLM for each variant x input x run, collects trials, builds summaries
  stats.py        # Statistical comparison: bootstrap confidence intervals or Welch's t-test between two variants
```

## How It Works

1. **Define** an `Experiment` with `PromptVariant`s (prompt messages + model config) and `ExperimentInput`s (test cases with optional ground truth).
2. **Run** via `run_experiment(experiment, evaluator=...)` -- calls the LLM for each variant/input/run combination, optionally scoring outputs with a custom evaluator function. Returns `EvalResult` with all trials and per-variant summaries (mean score, cost, latency, tokens).
3. **Compare** via `compare_variants(result, "variant_a", "variant_b")` -- computes confidence intervals on the score difference. Methods: `"bootstrap"` (default, 10k resamples) or `"welch"` (t-test with z-approximation).

Prompt messages use `{input}` as a placeholder, which gets substituted with each `ExperimentInput.content` at runtime. If `response_model` is set on the experiment, the runner uses `acall_llm_structured` for typed Pydantic output; otherwise it uses `acall_llm` for raw text.

## Usage Example

```python
import asyncio
from prompt_eval import Experiment, PromptVariant, ExperimentInput, run_experiment, compare_variants

experiment = Experiment(
    name="tone_test",
    variants=[
        PromptVariant(
            name="formal",
            messages=[{"role": "user", "content": "Summarize formally: {input}"}],
        ),
        PromptVariant(
            name="casual",
            messages=[{"role": "user", "content": "Summarize casually: {input}"}],
        ),
    ],
    inputs=[
        ExperimentInput(id="article_1", content="The quarterly earnings exceeded..."),
        ExperimentInput(id="article_2", content="New research suggests that..."),
    ],
    n_runs=5,
)

def score_length(output, expected):
    """Simple evaluator: shorter is better (normalized)."""
    return max(0, 1 - len(str(output)) / 1000)

result = asyncio.run(run_experiment(experiment, evaluator=score_length))

comparison = compare_variants(result, "formal", "casual")
print(f"Difference: {comparison.difference:.3f}")
print(f"Significant: {comparison.significant}")
print(comparison.detail)
```

## Dependencies

- `llm_client >= 0.3.0` -- async LLM calls via `acall_llm` / `acall_llm_structured`
- `pydantic >= 2.0` -- data models
- Default model: `gpt-5-mini` (configurable per variant)

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are in `tests/test_experiment.py`, `tests/test_runner.py`, `tests/test_stats.py`.

## Next Steps

- **MCP wrapper for agentic use**: Expose experiment definition and execution as MCP tools so agents can run prompt evaluations programmatically.
- **Optimization layer**: Grid search over prompt/model/temperature combinations, few-shot example selection, instruction search (automated prompt rewriting to maximize evaluator scores).
- **Persistence**: Save/load experiments and results to disk (JSON) so results can be reviewed, compared across sessions, and shared.
- **Integration with qualitative_coding IRR metrics as evaluators**: Use Cohen's kappa, Fleiss' kappa, and stability scores from the qualitative_coding project as evaluator functions for prompt experiments on coding tasks.
