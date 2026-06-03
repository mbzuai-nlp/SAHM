"""Locate and load the task / model registries.

By default the packaged configs (``sahm_eval/configs/*.yaml``) are used, so the
``sahm-eval`` command works from any directory after ``pip install``. Pass an
explicit path to ``--tasks-file`` / ``--models-file`` to override.
"""

from importlib import resources
from pathlib import Path

import yaml


def _read_yaml(path: str | None, default_name: str) -> dict:
    if path:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    with resources.files("sahm_eval.configs").joinpath(default_name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(path: str | None = None) -> dict:
    return _read_yaml(path, "tasks.yaml")["tasks"]


def load_models(path: str | None = None) -> tuple[dict, dict]:
    y = _read_yaml(path, "models.yaml")
    return y.get("open_models", {}), y.get("api_models", {})


def default_config_path(name: str) -> Path:
    """Filesystem path to a packaged default config (for `sahm-eval config` / docs)."""
    return Path(str(resources.files("sahm_eval.configs").joinpath(name)))
