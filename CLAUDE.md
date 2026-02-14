# prompt_eval (v0.2.0)

## What This Project Does

A/B prompt testing and evaluation library. Define prompt variants, run them against test inputs with an LLM, and compare results with statistical rigor (bootstrap CI or Welch's t-test). Includes persistence, evaluator factories, three optimization strategies (grid search, few-shot selection, instruction search), and an MCP server for agentic use.

## Architecture

```
prompt_eval/
  experiment.py   # Pydantic data models: Experiment, PromptVariant, ExperimentInput, Trial, EvalResult, VariantSummary
  runner.py       # Async experiment runner: calls LLM for each variant x input x run, collects trials, builds summaries
  stats.py        # Statistical comparison: bootstrap confidence intervals or Welch's t-test between two variants
  store.py        # Persistence: save/load experiments and results as JSON files (~/.prompt_eval/results/)
  evaluators.py   # Evaluator factories: kappa_evaluator, exact_match_evaluator, contains_evaluator
  optimize.py     # Three strategies: grid_search, few_shot_selection, instruction_search
  mcp_server.py   # FastMCP server with 4 tools (not imported in __init__.py — requires fastmcp)
```

## How It Works

1. **Define** an `Experiment` with `PromptVariant`s (prompt messages + model config) and `ExperimentInput`s (test cases with optional ground truth).
2. **Run** via `run_experiment(experiment, evaluator=...)` — calls the LLM for each variant/input/run combination, optionally scoring outputs with a custom evaluator function. Returns `EvalResult` with all trials and per-variant summaries (mean score, cost, latency, tokens).
3. **Compare** via `compare_variants(result, "variant_a", "variant_b")` — computes confidence intervals on the score difference. Methods: `"bootstrap"` (default, 10k resamples) or `"welch"` (t-test with z-approximation).
4. **Save/Load** via `save_result(result)` / `load_result(path)` — persists to `~/.prompt_eval/results/{name}/`.
5. **Evaluate** with factory functions: `kappa_evaluator(extractor)` for inter-rater reliability, `exact_match_evaluator()`, `contains_evaluator()`.
6. **Optimize** via `optimize(search_space, inputs, evaluator, strategy=...)` — three strategies available.
7. **MCP** via `prompt-eval-mcp` entry point — exposes run/get/list/compare as agent-callable tools.

Prompt messages use `{input}` as a placeholder, which gets substituted with each `ExperimentInput.content` at runtime. If `response_model` is set on the experiment, the runner uses `acall_llm_structured` for typed Pydantic output; otherwise it uses `acall_llm` for raw text.

## Usage Example

```python
import asyncio
from prompt_eval import (
    Experiment, PromptVariant, ExperimentInput,
    run_experiment, compare_variants, save_result, kappa_evaluator,
)

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
save_result(result)  # persists to ~/.prompt_eval/results/tone_test/

comparison = compare_variants(result, "formal", "casual")
print(f"Difference: {comparison.difference:.3f}")
print(f"Significant: {comparison.significant}")
print(comparison.detail)
```

## Evaluators

```python
from prompt_eval import kappa_evaluator, exact_match_evaluator, contains_evaluator

# Cohen's kappa — compares code lists from output vs expected
ev = kappa_evaluator(lambda r: r.codes)  # extractor pulls code names from Pydantic model
score = ev(output_model, expected_model)  # returns kappa float

# Exact match — 1.0 if str(output) == str(expected)
ev = exact_match_evaluator()

# Contains — 1.0 if expected found in output
ev = contains_evaluator()
```

## Optimization

Three strategies, all accessed via `optimize()` or called directly:

### Grid Search (default)

Exhaustive search over prompt/model/temperature combinations.

```python
from prompt_eval import optimize, SearchSpace, ExperimentInput

space = SearchSpace(
    prompt_templates=[
        [{"role": "user", "content": "Summarize: {input}"}],
        [{"role": "user", "content": "Give a brief summary of: {input}"}],
    ],
    models=["gpt-5-mini", "gpt-5"],
    temperatures=[0.0, 0.5, 1.0],
)

result = await optimize(space, inputs, evaluator)
print(f"Best: {result.best_variant} (score: {result.best_score:.3f})")
```

### Few-Shot Selection

Search over C(n, k) combinations of examples from a pool. Use `{examples}` placeholder in the base prompt.

```python
from prompt_eval import optimize, FewShotPool, SearchSpace

pool = FewShotPool(
    examples=["Example 1: ...", "Example 2: ...", "Example 3: ...", "Example 4: ..."],
    k=2,  # pick 2 examples per combo
    base_messages=[{"role": "user", "content": "Here are examples:\n{examples}\n\nNow analyze: {input}"}],
)
space = SearchSpace(prompt_templates=[])  # not used by few_shot_selection

result = await optimize(space, inputs, evaluator, strategy="few_shot_selection", pool=pool)
```

Use `budget=N` to randomly sample N combinations instead of trying all C(n, k).

### Instruction Search

Hill-climbing: LLM rewrites the instruction each iteration, evaluates rewrites, keeps the best.

```python
result = await optimize(
    space, inputs, evaluator,
    strategy="instruction_search",
    base_instruction="Analyze this text for key themes: {input}",
    n_iterations=5,   # hill-climbing iterations
    n_rewrites=3,     # rewrites per iteration
    model="gpt-5-mini",          # model to evaluate with
    rewrite_model="gpt-5-mini",  # model to generate rewrites
)
```

## MCP Server

Requires `pip install prompt-eval[mcp]`. Not imported by default.

```bash
prompt-eval-mcp  # starts FastMCP server
```

Four tools: `run_experiment_tool`, `get_result`, `list_experiments`, `compare`.
Only built-in evaluators (`exact_match`, `contains`) available via MCP — `kappa_evaluator` requires a callable.

## Dependencies

- `llm_client >= 0.3.0` — async LLM calls via `acall_llm` / `acall_llm_structured`
- `pydantic >= 2.0` — data models
- `fastmcp >= 0.1.0` — optional, for MCP server (`pip install prompt-eval[mcp]`)
- Default model: `gpt-5-mini` (configurable per variant)

## Running Tests

```bash
python -m pytest tests/ -v
```

7 test files, 93 tests.

## Completed (v0.2.0)

- **Persistence** — `save_result`, `load_result`, `save_experiment`, `load_experiment`, `list_results`
- **Evaluators** — `kappa_evaluator` (Cohen's kappa), `exact_match_evaluator`, `contains_evaluator`
- **Grid search** — exhaustive search over prompt/model/temperature/kwargs combinations
- **Few-shot selection** — search over C(n,k) example combinations with optional budget cap
- **Instruction search** — LLM-powered hill-climbing prompt rewriting
- **MCP server** — 4 tools via FastMCP, optional dependency

## Next Steps

- **Integration with qualitative_coding IRR metrics** — use QC's kappa as evaluator for coding task prompts
- **Multi-model consensus** — run analysis across GPT/Claude/Gemini and merge results

## Related Projects

| Project | Path | Relationship |
|---------|------|-------------|
| **llm_client** | `~/projects/llm_client/` | Shared LLM calling library. prompt_eval uses `acall_llm` / `acall_llm_structured` |
| **qualitative_coding** | `~/projects/qualitative_coding/` | Uses prompt_eval for A/B prompt testing with IRR kappa as evaluator |
