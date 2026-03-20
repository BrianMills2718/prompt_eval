# prompt_eval package

This subtree contains the production package for prompt evaluation and
optimization.

## Use This Directory For

- experiment execution and reconstruction
- evaluator and statistics surfaces
- prompt asset resolution at the `prompt_eval` layer
- optimization helpers and optional MCP server behavior

## Route Narrower Work

- packaged prompt templates -> `prompts/`

## Working Rules

- Keep this package prompt-centric.
- Preserve the explicit-model contract for experiment semantics.
- `llm_client` remains the execution substrate; do not reimplement its runtime
  responsibilities here.
