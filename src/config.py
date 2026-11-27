"""Configuration loading for the predictive maintenance pipeline.

Reads ``src/config/config.yaml`` and applies overrides on top of it, so every
path, threshold, seed and hyperparameter comes from one place.

Overrides use dotted keys that point into the nested YAML, for example
``{"features.feature_set": "all", "random_state": 7}``. They come from the
command line (built in ``main.py``) and from environment variables named
``EGT309__<SECTION>__<KEY>`` (double underscore = one level of nesting).
Precedence, lowest to highest: YAML file, environment, command line.

Note: this module is ``src/config.py`` and the YAML lives in the ``src/config/``
folder. That folder must NOT get an ``__init__.py``, otherwise it would shadow
this module on import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Prefix of environment variables that override config values.
ENV_PREFIX: str = "EGT309__"

#: Allowed values of ``features.feature_set`` (single source of truth for the whole pipeline).
FEATURE_SETS: tuple[str, ...] = ("sensors_only", "all")

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

#: Keys that must exist inside ``paths``.
REQUIRED_PATHS: tuple[str, ...] = ("raw_csv", "database", "model_dir", "report_dir")


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
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Config file {config_path} is not valid YAML: {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level.")

    apply_overrides(cfg, env_overrides())
    apply_overrides(cfg, overrides or {})
    validate_config(cfg)
    resolve_paths(cfg, project_root())

    logger.info("Loaded config from %s", config_path)
    return cfg


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
    for dotted_key, value in overrides.items():
        if value is None:
            continue

        *parents, leaf = dotted_key.split(".")
        section = cfg
        for part in parents:
            if not isinstance(section.get(part), dict):
                raise KeyError(f"Unknown config override '{dotted_key}': '{part}' is not a config section.")
            section = section[part]
        if leaf not in section:
            raise KeyError(f"Unknown config override '{dotted_key}': no key '{leaf}' in the config.")

        logger.info("Config override: %s = %r (was %r)", dotted_key, value, section[leaf])
        section[leaf] = value
    return cfg


def env_overrides(prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Collect config overrides from environment variables that start with ``prefix``.

    ``EGT309__TUNING__N_ITER=5`` becomes ``{"tuning.n_iter": 5}``. Values are
    parsed with YAML so numbers and booleans keep their type.

    Args:
        prefix: Environment variable prefix to look for.

    Returns:
        Dotted-key overrides found in the environment (empty if none).
    """
    overrides: dict[str, Any] = {}
    for name, raw_value in os.environ.items():
        if not name.upper().startswith(prefix):
            continue
        dotted_key = name[len(prefix):].lower().replace("__", ".")
        overrides[dotted_key] = yaml.safe_load(raw_value)
    return overrides


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
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"Config is missing required section(s): {', '.join(missing)}.")

    missing_paths = [key for key in REQUIRED_PATHS if key not in cfg["paths"]]
    if missing_paths:
        raise ValueError(f"Config 'paths' is missing: {', '.join(missing_paths)}.")

    if not isinstance(cfg["random_state"], int):
        raise ValueError(f"random_state must be an integer, got {cfg['random_state']!r}.")

    feature_set = cfg["features"].get("feature_set")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"features.feature_set must be one of {FEATURE_SETS}, got {feature_set!r}.")

    for key in ("target", "identifiers", "sensors", "derived"):
        if key not in cfg["features"]:
            raise ValueError(f"Config 'features' is missing '{key}'.")

    test_size = cfg["split"].get("test_size")
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError(f"split.test_size must be a number between 0 and 1, got {test_size!r}.")

    for key in ("temp_threshold", "vib_threshold"):
        if not isinstance(cfg["preprocessing"].get(key), (int, float)):
            raise ValueError(f"preprocessing.{key} must be a number, got {cfg['preprocessing'].get(key)!r}.")

    enabled = cfg["models"].get("enabled") or []
    if not enabled:
        raise ValueError("models.enabled must list at least one model.")
    for name in enabled:
        model_cfg = cfg["models"].get(name)
        if not isinstance(model_cfg, dict) or "defaults" not in model_cfg or "search_space" not in model_cfg:
            raise ValueError(f"Enabled model '{name}' needs a 'models.{name}' block with 'defaults' and 'search_space'.")

    cv_folds = cfg["tuning"].get("cv_folds")
    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError(f"tuning.cv_folds must be an integer >= 2, got {cv_folds!r}.")

    beta = cfg["threshold"].get("beta")
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError(f"threshold.beta must be a positive number, got {beta!r}.")

    min_precision = cfg["threshold"].get("min_precision")
    if not isinstance(min_precision, (int, float)) or not 0 <= min_precision <= 1:
        raise ValueError(f"threshold.min_precision must be between 0 and 1, got {min_precision!r}.")


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Turn every relative entry in ``cfg["paths"]`` into an absolute ``Path`` under ``root``.

    Args:
        cfg: Nested configuration dict (modified in place and returned).
        root: Project root directory (the parent of ``src/``).

    Returns:
        The configuration dict with absolute paths.
    """
    for key, value in cfg["paths"].items():
        path = Path(value)
        cfg["paths"][key] = path if path.is_absolute() else (root / path).resolve()
    return cfg


def project_root() -> Path:
    """Return the project root directory, derived from this file's location.

    Returns:
        Absolute path of the folder that contains ``src/``.
    """
    return Path(__file__).resolve().parent.parent
