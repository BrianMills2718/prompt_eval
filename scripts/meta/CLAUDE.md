# scripts/meta

This directory contains the repo-local implementation scripts used by
`prompt_eval` workflow helpers.

## Use This Directory For

- plan-status sync
- plan blocker checks
- plan test inspection
- other inherited meta-process helpers that still apply in this repo

## Working Rules

- Treat these scripts as workflow support, not the main package surface.
- If a helper assumes governance patterns that `prompt_eval` does not enforce,
  say so in the local docs rather than pretending parity.
