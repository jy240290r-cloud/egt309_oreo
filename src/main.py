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
    overrides: dict[str, Any] = {
        "models.enabled": args.models,
        "features.feature_set": args.feature_set,
        "random_state": args.seed,
    }
    # Switches are False when not given; only a switch that is ON becomes an override.
    if args.no_tune:
        overrides["tuning.enabled"] = False
    if args.rebuild_db:
        overrides["database.rebuild"] = True
    # load_config skips None values, but dropping them here keeps the log clean.
    return {key: value for key, value in overrides.items() if value is not None}


def configure_logging(cfg: dict[str, Any]) -> None:
    """Set up the root logger from the ``logging`` section of the config.

    Args:
        cfg: Full pipeline configuration.
    """
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        force=True,  # replace any handler set up before the config was loaded
    )


def run(cfg: dict[str, Any]) -> str:
    """Run the full pipeline: ingest, load, split, train, evaluate and save.

    Args:
        cfg: Full pipeline configuration (overrides applied).

    Returns:
        Name of the selected best model.
    """
    paths = cfg["paths"]
    logger.info(
        "Pipeline start | models=%s | feature_set=%s | tuning=%s | seed=%s",
        cfg["models"]["enabled"], cfg["features"]["feature_set"],
        cfg["tuning"]["enabled"], cfg["random_state"],
    )

    # 1. Ingest: CSV -> SQLite (only if needed), then load the readings with SQL.
    ingestor = DataIngestor(cfg)
    ingestor.ensure_database()
    df = ingestor.load_data()

    # 2. Split: drop identifiers (and leaky columns unless feature_set = all), stratified train/test.
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    # 3. Train: hyperparameter search + threshold tuning on the training set only, then save.
    trainer = ModelTrainer(cfg)
    trained = trainer.fit_all(X_train, y_train)
    trainer.save(trained, paths["model_dir"])

    # 4. Evaluate: the test set is used here, once.
    evaluator = ModelEvaluator(cfg)
    results = evaluator.evaluate(trained, X_test, y_test)
    evaluator.feature_importance(trained, X_test, y_test)
    evaluator.plot_confusion_matrices(trained, X_test, y_test, paths["report_dir"])
    evaluator.plot_pr_curves(trained, X_test, y_test, paths["report_dir"])
    best = evaluator.select_best(results)
    evaluator.save_report(paths["report_dir"])

    logger.info("Pipeline finished. Best model: %s | reports in %s", best, paths["report_dir"])
    return best


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the config and run the pipeline.

    Args:
        argv: Argument list without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    # Temporary INFO logging so messages from loading the config (e.g. overrides) are shown;
    # configure_logging then applies the level and format from the config itself.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    cfg = load_config(args.config, args_to_overrides(args))
    configure_logging(cfg)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())