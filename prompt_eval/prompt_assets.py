"""Helpers for building prompt_eval variants from explicit prompt assets.

The goal of this module is narrow: turn an explicit shared ``prompt_ref`` plus
render context into a ``PromptVariant`` that can run through the normal
``prompt_eval`` execution path. The prompt asset remains the source of truth;
``prompt_eval`` simply materializes it into messages and records the identity.
"""

from __future__ import annotations

from typing import Any, Mapping

from llm_client import render_prompt

from prompt_eval.experiment import PromptVariant


def build_prompt_variant_from_ref(
    name: str,
    prompt_ref: str,
    *,
    model: str,
    render_context: Mapping[str, Any] | None = None,
    temperature: float = 1.0,
    kwargs: Mapping[str, Any] | None = None,
) -> PromptVariant:
    """Render a shared prompt asset into a ``PromptVariant``.

    Args:
        name: Human-readable variant name for the experiment.
        prompt_ref: Explicit prompt asset reference to resolve through
            ``llm_client``.
        render_context: Jinja2 context used when rendering the prompt asset.
            Prompt assets intended for ``prompt_eval`` should preserve the
            literal ``{input}`` placeholder so the runner can substitute each
            experiment input later.
        model: Explicit subject model identifier for the variant.
        temperature: Sampling temperature passed through to ``llm_client``.
        kwargs: Extra per-call kwargs forwarded to ``llm_client``.

    Returns:
        Prompt variant with rendered messages and explicit prompt asset identity.

    Raises:
        FileNotFoundError: If the prompt asset cannot be resolved.
        ValueError: If the prompt asset or render context is invalid.
        jinja2.UndefinedError: If a required render variable is missing.
    """

    rendered_messages = render_prompt(
        prompt_ref=prompt_ref,
        **dict(render_context or {}),
    )
    return PromptVariant(
        name=name,
        prompt_ref=prompt_ref,
        messages=rendered_messages,
        model=model,
        temperature=temperature,
        kwargs=dict(kwargs or {}),
    )
