"""Local prompt-template helpers for ``prompt_eval`` internal LLM calls.

These prompts are part of ``prompt_eval``'s evaluator and optimizer machinery.
They intentionally live as YAML/Jinja templates under the package
``prompts/`` directory and are rendered through ``llm_client`` so prompt
content stays inspectable without inventing fake shared ``prompt_ref``
identities.
"""

from __future__ import annotations

from pathlib import Path

from llm_client import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SCALAR_JUDGE_TEMPLATE_PATH = _PROMPTS_DIR / "llm_judge_scalar.yaml"
DIMENSIONAL_JUDGE_TEMPLATE_PATH = _PROMPTS_DIR / "llm_judge_dimensional.yaml"
INSTRUCTION_REWRITE_TEMPLATE_PATH = _PROMPTS_DIR / "instruction_search_rewrite.yaml"


def render_scalar_judge_messages(
    *,
    rubric: str,
    output: str,
    expected_section: str,
) -> list[dict[str, str]]:
    """Render the single-score LLM judge prompt."""

    return render_prompt(
        template_path=SCALAR_JUDGE_TEMPLATE_PATH,
        rubric=rubric,
        output=output,
        expected_section=expected_section,
    )


def render_dimensional_judge_messages(
    *,
    dimensions_text: str,
    output: str,
    expected_section: str,
) -> list[dict[str, str]]:
    """Render the dimensional structured judge prompt."""

    return render_prompt(
        template_path=DIMENSIONAL_JUDGE_TEMPLATE_PATH,
        dimensions_text=dimensions_text,
        output=output,
        expected_section=expected_section,
    )


def render_instruction_rewrite_messages(
    *,
    n_rewrites: int,
    current_best_instruction: str,
) -> list[dict[str, str]]:
    """Render the instruction-search rewrite prompt."""

    return render_prompt(
        template_path=INSTRUCTION_REWRITE_TEMPLATE_PATH,
        n_rewrites=n_rewrites,
        current_best_instruction=current_best_instruction,
    )
