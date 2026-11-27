"""Command-line entry point of the predictive maintenance pipeline.

Flow: ingest (CSV -> SQLite) -> load (SQL) -> split -> train -> evaluate -> save.

Usage::

    python src/main.py --config src/config/config.yaml
    python src/main.py --models random_forest --no-tune --seed 7
    python src/main.py --feature-set all      # leakage demonstration only

CLI flags override the matching values in the YAML config. Flags that are not
given leave the config value unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from config import load_config
from data_ingestion import DataIngestor
from evaluation import ModelEvaluator
from models import MODEL_REGISTRY
from preprocessing import FEATURE_SETS, split_data
from trainer import ModelTrainer

logger = logging.getLogger(__name__)

#: Default config location, next to this file (not dependent on the working directory).
DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "config" / "config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Every option defaults to None (or False for switches), so only flags the
    user actually passes become config overrides.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Train and evaluate classifiers that predict maintenance_required "
            "from machine sensor readings."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        default=None,
        metavar="MODEL",
        help=(
            "Models to train, overriding models.enabled. "
            f"Choices: {', '.join(sorted(MODEL_REGISTRY))}."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=None,
        help=(
            "Feature set, overriding features.feature_set. 'all' adds the derived "
            "(target-leaking) columns and exists only to demonstrate leakage."
        ),
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip the hyperparameter search and use each model's config defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, overriding random_state.",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Recreate the SQLite database from the raw CSV before loading.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list without the program name; None means ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    return build_arg_parser().parse_args(argv)


def args_to_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate parsed CLI flags into dotted-key config overrides.

    Only flags the user set are included, e.g. ``--no-tune`` becomes
    ``{"tuning.enabled": False}`` and ``--seed 7`` becomes ``{"random_state": 7}``.

    Args:
        args: Output of ``parse_args``.

    Returns:
        Overrides for ``load_config``.
    """
    raise NotImplementedError


def configure_logging(cfg: dict[str, Any]) -> None:
    """Set up the root logger from the ``logging`` section of the config.

    Args:
        cfg: Full pipeline configuration.
    """
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> str:
    """Run the full pipeline: ingest, load, split, train, evaluate and save.

    Args:
        cfg: Full pipeline configuration (overrides applied).

    Returns:
        Name of the selected best model.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the config and run the pipeline.

    Args:
        argv: Argument list without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    cfg = load_config(args.config, args_to_overrides(args))
    configure_logging(cfg)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
