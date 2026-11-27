"""Configuration loading for the predictive maintenance pipeline.

Reads ``src/config/config.yaml`` and applies overrides on top of it, so every
path, threshold, seed and hyperparameter comes from one place.

Overrides use dotted keys that point into the nested YAML, for example
``{"features.feature_set": "all", "random_state": 7}``. They come from the
command line (built in ``main.py``) and from environment variables named
``EGT309__<SECTION>__<KEY>`` (double underscore = one level of nesting).

Note: this module is ``src/config.py`` and the YAML lives in the ``src/config/``
folder. That folder must NOT get an ``__init__.py``, otherwise it would shadow
this module on import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Prefix of environment variables that override config values.
ENV_PREFIX: str = "EGT309__"

#: Top-level sections that must be present in a valid config.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "paths",
    "database",
    "random_state",
    "split",
    "features",
    "preprocessing",
    "models",
    "tuning",
    "threshold",
    "evaluation",
)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML config, apply env and CLI overrides, validate it and resolve paths.

    Args:
        path: Path to the YAML config file.
        overrides: Dotted-key overrides (e.g. ``{"tuning.enabled": False}``).
            Applied after environment overrides, so the CLI wins.

    Returns:
        The merged configuration as a nested dict, with ``paths`` resolved to
        absolute paths relative to the project root.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is invalid or a required section is missing.
    """
    raise NotImplementedError


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Set each dotted-key override in the nested config dict.

    Args:
        cfg: Nested configuration dict (modified in place and returned).
        overrides: Mapping of dotted keys to new values. ``None`` values are skipped.

    Returns:
        The updated configuration dict.

    Raises:
        KeyError: If a dotted key does not exist in the config (catches typos).
    """
    raise NotImplementedError


def env_overrides(prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Collect config overrides from environment variables that start with ``prefix``.

    ``EGT309__TUNING__N_ITER=5`` becomes ``{"tuning.n_iter": 5}``. Values are
    parsed with YAML so numbers and booleans keep their type.

    Args:
        prefix: Environment variable prefix to look for.

    Returns:
        Dotted-key overrides found in the environment (empty if none).
    """
    raise NotImplementedError


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that the config has every required section and sensible values.

    Checks include: required sections present, ``features.feature_set`` is
    ``sensors_only`` or ``all``, ``split.test_size`` is in (0, 1), and every
    enabled model has a ``defaults`` and ``search_space`` block.

    Args:
        cfg: Nested configuration dict.

    Raises:
        ValueError: With a message naming the first problem found.
    """
    raise NotImplementedError


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Turn every relative entry in ``cfg["paths"]`` into an absolute ``Path`` under ``root``.

    Args:
        cfg: Nested configuration dict (modified in place and returned).
        root: Project root directory (the parent of ``src/``).

    Returns:
        The configuration dict with absolute paths.
    """
    raise NotImplementedError


def project_root() -> Path:
    """Return the project root directory, derived from this file's location.

    Returns:
        Absolute path of the folder that contains ``src/``.
    """
    raise NotImplementedError
