"""Persistence for experiments and results — JSON files on disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from prompt_eval.experiment import EvalResult, Experiment

_DEFAULT_BASE = Path.home() / ".prompt_eval" / "results"


def _resolve_dir(experiment_name: str, base_dir: Path | None = None) -> Path:
    """Return the directory for an experiment's results, creating it if needed."""
    base = base_dir or _DEFAULT_BASE
    d = base / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(result: EvalResult, path: Path | None = None) -> Path:
    """Save an EvalResult to JSON. Returns the path written."""
    if path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        d = _resolve_dir(result.experiment_name)
        path = d / f"{result.experiment_name}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_result(path: Path) -> EvalResult:
    """Load an EvalResult from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    try:
        return EvalResult.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid result file {path}: {e}") from e


def save_experiment(experiment: Experiment, path: Path | None = None) -> Path:
    """Save an Experiment definition to JSON. Returns the path written."""
    if path is None:
        d = _resolve_dir(experiment.name)
        path = d / f"{experiment.name}_def.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(experiment.model_dump_json(indent=2))
    return path


def load_experiment(path: Path) -> Experiment:
    """Load an Experiment from JSON.

    Note: response_model (a Python type) cannot roundtrip through JSON.
    It will be None on the loaded experiment.
    """
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")
    try:
        return Experiment.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid experiment file {path}: {e}") from e


def list_results(
    experiment_name: str | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """List saved result files, optionally filtered by experiment name.

    Uses *_[0-9]*.json glob to distinguish results from _def.json files.
    Returns paths sorted by modification time (newest first).
    """
    base = base_dir or _DEFAULT_BASE
    if not base.exists():
        return []

    if experiment_name is not None:
        d = base / experiment_name
        if not d.exists():
            return []
        paths = list(d.glob("*_[0-9]*.json"))
    else:
        paths = list(base.glob("*/*_[0-9]*.json"))

    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
