"""Shared pytest fixtures for prompt_eval tests.

prompt_eval now emits llm_client observability records by default, so tests must
isolate llm_client's backing state and environment from the real user machine.

In CI (no model registry / API keys), ``get_model`` raises because no models
qualify. An autouse fixture patches it to return a stub so evaluator-construction
tests that mock the actual LLM call still pass.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_client import get_model, io_log

_STUB_JUDGE_MODEL = "stub-judge-for-ci"


def _model_registry_available() -> bool:
    """Return True if ``get_model('judging')`` resolves without error."""
    try:
        get_model("judging")
        return True
    except Exception:
        return False


_REGISTRY_OK = _model_registry_available()


@pytest.fixture(autouse=True)
def _stub_get_model_in_ci() -> Iterator[None]:
    """When the model registry has no configured models (CI), patch get_model."""
    if _REGISTRY_OK:
        yield
        return
    with patch(
        "prompt_eval.evaluators.get_model",
        return_value=_STUB_JUDGE_MODEL,
    ):
        yield


@pytest.fixture(autouse=True)
def _fix_optimize_module_shadow() -> Iterator[None]:
    """Restore ``prompt_eval.optimize`` attribute to the module, not the function.

    ``prompt_eval.__init__`` re-exports ``optimize`` (the function) which shadows
    the ``prompt_eval.optimize`` *module* when accessed via ``getattr``. This
    breaks ``unittest.mock.patch("prompt_eval.optimize.acall_llm")`` on some
    Python versions.  We temporarily restore the module attribute so patch()
    traverses correctly.
    """
    import sys
    import prompt_eval as pkg

    optimize_mod = sys.modules["prompt_eval.optimize"]
    optimize_fn = pkg.optimize  # the function
    pkg.optimize = optimize_mod  # type: ignore[assignment]
    yield
    pkg.optimize = optimize_fn  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _isolate_llm_client_observability(tmp_path: Path) -> Iterator[None]:
    """Isolate llm_client observability state for every prompt_eval test."""
    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn
    old_run_timers = dict(io_log._run_timers)
    active_token = io_log._active_experiment_run_id.set(None)
    feature_profile_token = io_log._active_feature_profile.set(None)
    old_enforcement_mode = os.environ.get("LLM_CLIENT_EXPERIMENT_ENFORCEMENT")
    old_task_patterns = os.environ.get("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS")
    old_feature_profile = os.environ.get("LLM_CLIENT_FEATURE_PROFILE")
    old_feature_enforcement_mode = os.environ.get("LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT")
    old_feature_task_patterns = os.environ.get("LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS")
    old_agent_spec_enforcement_mode = os.environ.get("LLM_CLIENT_AGENT_SPEC_ENFORCEMENT")
    old_agent_spec_task_patterns = os.environ.get("LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS")

    io_log._enabled = True
    io_log._data_root = tmp_path
    io_log._project = "prompt_eval_tests"
    io_log._db_path = tmp_path / "prompt_eval_tests.db"
    io_log._db_conn = None
    io_log._run_timers.clear()

    yield

    io_log._enabled = old_enabled
    io_log._data_root = old_root
    io_log._project = old_project
    io_log._db_path = old_db_path
    if io_log._db_conn is not None:
        io_log._db_conn.close()
    io_log._db_conn = old_db_conn
    io_log._run_timers.clear()
    io_log._run_timers.update(old_run_timers)
    io_log._active_experiment_run_id.reset(active_token)
    io_log._active_feature_profile.reset(feature_profile_token)

    if old_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_EXPERIMENT_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_EXPERIMENT_ENFORCEMENT"] = old_enforcement_mode
    if old_task_patterns is None:
        os.environ.pop("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_EXPERIMENT_TASK_PATTERNS"] = old_task_patterns
    if old_feature_profile is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE"] = old_feature_profile
    if old_feature_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT"] = old_feature_enforcement_mode
    if old_feature_task_patterns is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS"] = old_feature_task_patterns
    if old_agent_spec_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_AGENT_SPEC_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_AGENT_SPEC_ENFORCEMENT"] = old_agent_spec_enforcement_mode
    if old_agent_spec_task_patterns is None:
        os.environ.pop("LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS"] = old_agent_spec_task_patterns
