# Finding: LLM Judge Model Choice Dominates Prompt Evaluation Signal

**Date**: 2026-02-14
**Context**: A/B testing 3 thematic coding prompts using prompt_eval

## Experiment Setup

- **Task**: Qualitative thematic coding of interview transcripts
- **Coding model**: gpt-5-mini (structured output via CodeHierarchy schema)
- **3 prompt variants**: current (full field specs), concise (analytical instructions only), methodological (Braun & Clarke framing)
- **Evaluator**: `llm_judge_dimensional_evaluator` with 4 dimensions (code_clarity, grounding, coverage, analytical_depth)

## Results Across 3 Runs

### Run 1: Single-score evaluator, gpt-5-mini judge
- 1 input, 3 runs/variant, 0.0-1.0 float scale
- **Result**: methodological (0.873) > concise (0.863) > current (0.763)
- concise and methodological significantly beat current

### Run 2: Dimensional evaluator, gemini-2.0-flash judge
- 3 inputs, 3 runs/variant, 0.0-1.0 float scale
- **Result**: methodological (0.946) > concise (0.918) > current (0.846)
- Same ranking as Run 1. Five trials scored perfect 1.0.
- **Problem**: Score compression at ceiling (0.75-1.0 range), 0.05 quantization

### Run 3: Dimensional evaluator, gpt-4o judge
- 3 inputs, 3 runs/variant, 0-100 integer scale with defect-finding prompt
- **Result**: current (0.827) > methodological (0.797) > concise (0.681)
- **Ranking completely reversed.** 4 timeouts + 1 all-zero verdict corrupted data.
- Score spread improved (0.68-0.86), no perfect scores, finer granularity

## Key Findings

### 1. Judge model choice dominates signal
The same prompt variants received opposite rankings from different judge models:
- gemini-flash: methodological > concise > current
- gpt-4o: current > methodological > concise

**Implication**: Single-judge results are not trustworthy for prompt comparison. Multi-judge ensembling (already supported via `judge_models` list) should be the default recommendation.

### 2. 0-100 integer scale improves discrimination
Switching from 0.0-1.0 float to 0-100 integer:
- Broke the 0.05 quantization (scores like 0.78, 0.82, 0.84 instead of 0.75, 0.80, 0.85)
- Eliminated perfect 1.0 scores
- Wider score spread (0.68-0.86 vs 0.85-1.0)

### 3. Defect-finding prompt reduces ceiling compression
Adding "identify specific weaknesses before scoring" to the judge prompt:
- Forced the judge to articulate problems before scoring
- Naturally spread scores away from the ceiling
- Judge reasoning became more detailed and actionable

### 4. Failed trials corrupt means
API timeouts + judge failures (all-zero scores) significantly skewed variant means. The `concise` variant's mean dropped from ~0.80 (excluding failures) to 0.68 (including a 0.000 trial).

## Bugs Found and Fixed

| Bug | Fix |
|-----|-----|
| `llm_client`: GPT-5 responses API doesn't strip temperature | Added temperature to stripped params in `_prepare_responses_kwargs` |
| `prompt_eval`: `acall_llm` returns `LLMCallResult` not tuple | Updated evaluators.py, runner.py, optimize.py to use `.content` |
| `prompt_eval`: 0.05 quantization on 0.0-1.0 scale | Switched to 0-100 integer scale, divide by 100 internally |
| `prompt_eval`: Judge all-zero verdict counted as real score | Raise exception when no judge produces valid scores |
| `prompt_eval`: Judge calls used llm_client's 60s default timeout | Added `timeout=120` parameter to both evaluator factories (llm_client already retries 2x on timeout) |

## Recommendations for prompt_eval Users

1. **Always use 2+ judge models** from different providers (e.g., `judge_models=["gpt-4o", "claude-sonnet-4-5-20250929"]`). Single-judge results may reflect model-specific biases, not actual quality differences.
2. **Use the dimensional evaluator** (`llm_judge_dimensional_evaluator`) over the single-score evaluator. Per-dimension scores reveal what's actually different.
3. **Examine judge reasoning** before trusting scores. The reasoning field shows whether the judge engaged with the actual content or pattern-matched to high scores.
4. **Run enough trials** (5+ per variant per input) to absorb noise from API failures and LLM non-determinism.
5. **Check for errors** in results before drawing conclusions. Variants with high error rates have unreliable means.
