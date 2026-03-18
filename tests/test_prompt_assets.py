"""Tests for prompt asset helpers and prompt_eval integration."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult
from llm_client.observability import get_runs

from prompt_eval.experiment import Experiment, ExperimentInput
from prompt_eval.observability import PromptEvalObservabilityConfig
from prompt_eval.prompt_assets import build_prompt_variant_from_ref
from prompt_eval.runner import run_experiment


class TestBuildPromptVariantFromRef:
    """Render shared prompt assets into prompt_eval variants."""

    def test_builds_variant_from_shared_prompt_asset(self) -> None:
        variant = build_prompt_variant_from_ref(
            name="shared_bullet_variant",
            prompt_ref="shared.summarize.bullet@1",
            render_context={"bullet_count": 3},
            kwargs={"max_tokens": 120},
        )

        assert variant.prompt_ref == "shared.summarize.bullet@1"
        assert variant.kwargs == {"max_tokens": 120}
        assert variant.messages[0]["role"] == "system"
        assert variant.messages[1]["content"] == (
            "Summarize the following text as 3 bullet points.\n\n{input}"
        )

    @pytest.mark.asyncio
    async def test_run_experiment_uses_built_prompt_asset_variant(self) -> None:
        variant = build_prompt_variant_from_ref(
            name="shared_bullet_variant",
            prompt_ref="shared.summarize.bullet@1",
            render_context={"bullet_count": 2},
        )
        experiment = Experiment(
            name="shared_prompt_asset_exp",
            variants=[variant],
            inputs=[ExperimentInput(id="doc1", content="The quick brown fox.")],
            n_runs=1,
        )

        with patch("prompt_eval.runner.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="summary text",
                usage={"total_tokens": 50},
                cost=0.001,
                model="test",
            )

            await run_experiment(
                experiment,
                observability=PromptEvalObservabilityConfig(
                    project="prompt_eval_asset_tests",
                    dataset="shared_prompt_asset_dataset",
                ),
            )

        await_args = mock_llm.await_args
        assert await_args is not None
        sent_messages = await_args.args[1]
        assert sent_messages[1]["content"] == (
            "Summarize the following text as 2 bullet points.\n\nThe quick brown fox."
        )

        runs = get_runs(
            project="prompt_eval_asset_tests",
            dataset="shared_prompt_asset_dataset",
            limit=5,
        )
        assert len(runs) == 1
        assert runs[0]["provenance"]["prompt_ref"] == "shared.summarize.bullet@1"
        assert runs[0]["provenance"]["prompt_source"] == "prompt_ref"
