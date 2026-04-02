# DeepEval Evaluation — Decision Document

**Date**: 2026-04-01
**Plan ref**: project-meta Plan #52
**Decision**: **REJECT** — DeepEval v0.21.36 has critical dependency conflicts

---

## What Was Tested

Attempted to install DeepEval (v0.21.36, latest) in the prompt_eval venv
and run the built-in smoke test.

## Findings

### Critical: Dependency Conflict Chain

DeepEval cannot import in a Python 3.12 environment with langchain >= 1.2:

1. `deepeval.models.gpt_model` imports `from langchain.schema import AIMessage, HumanMessage`
2. `langchain.schema` was removed in langchain 1.0+ (moved to `langchain_core.messages`)
3. Downgrading langchain to 0.3.x causes `langchain_openai` to fail:
   `cannot import name 'ContextOverflowError' from 'langchain_core.exceptions'`
4. The dependency chain is circular: DeepEval needs old langchain, but its
   other deps (langchain_openai, ragas) need new langchain-core

### Impact

- DeepEval **cannot be imported** without breaking prompt_eval's existing deps
- Installing DeepEval poisons the import chain (pytest collection fails ecosystem-wide)
- Uninstalling DeepEval restores prompt_eval to working state (168 passed)

### Root Cause

DeepEval v0.21.36 has a hard dependency on `langchain.schema` (a removed module)
and `langchain_openai` (which requires newer langchain-core). These are
incompatible. This is a known community issue — the library has not been updated
for the langchain 1.0+ migration.

## Decision: REJECT

**Evidence**: Cannot complete Plan #52 Step 1 (installation). Steps 2-5 are
blocked. The framework is not usable in our Python 3.12 + langchain 1.2
environment.

**Alternatives evaluated**:
- Pinning langchain to 0.2.x: Would break prompt_eval's langgraph workflow module
- Using a separate venv: Defeats the purpose (we'd need to maintain two venvs)
- Waiting for a fix: DeepEval's GitHub shows this is an active issue but no fix in stable

**What to do instead**:
- Keep prompt_eval's hand-rolled evaluators (they work, 168 tests pass)
- If DeepEval fixes their langchain dependency, re-evaluate
- Consider Braintrust (https://www.braintrust.dev/) as alternative — lighter
  dependency footprint, no langchain requirement

## Cost

- Time spent: ~20 minutes (install, debug, uninstall, restore)
- LLM cost: $0 (never got to the point of running an evaluation)
- Damage: None (prompt_eval fully restored after uninstall)
