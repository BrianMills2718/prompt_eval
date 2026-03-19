# Prompt Eval

Canonical repo-operating instructions live in [AGENTS.md](AGENTS.md).

This file intentionally stays thin to avoid drift. When working in this repo,
read these first:

1. [AGENTS.md](AGENTS.md)
2. [README.md](README.md)
3. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
4. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
5. [docs/UNCERTAINTIES.md](docs/UNCERTAINTIES.md)

Short version:

- `prompt_eval` is the prompt-evaluation and optimization layer on top of
  `llm_client`.
- Shared observability integration is complete; prompt asset and scope policy
  remain the active blocked boundary.
- Inline messages still work, but explicit `prompt_ref` lineage is the
  preferred direction.
