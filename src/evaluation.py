"""Model evaluation on the held-out test set.

The test set is used here **once**. Metrics are reported for class 1
(maintenance required): recall (main), precision, F1, F2, PR-AUC, ROC-AUC and
the confusion matrix. Accuracy is included for reference only, because always
predicting "no maintenance" already scores about 80%.

Every model is evaluated at the default threshold (0.5) and at its tuned
threshold. Feature importance combines permutation importance on the test set
(all models) with native importance where the model has one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trainer import TrainedModel

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the class-1 metrics for one set of probabilities at one threshold.

    Args:
        y_true: True 0/1 labels.
        y_prob: Predicted probabilities of class 1.
        threshold: Probability at or above which a row is predicted as 1.

    Returns:
        Dict with ``recall, precision, f1, f2, pr_auc, roc_auc, accuracy, tn, fp, fn, tp``.
    """
    raise NotImplementedError


class ModelEvaluator:
    """Evaluate trained models on the test set, compare them and write a report.

    Attributes:
        cfg: Full pipeline configuration.
        results: Metrics table from the last ``evaluate`` call (None before).
        importances: Feature-importance table from the last ``feature_importance`` call.
        figures: Paths of the plots written so far.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store the config and initialise empty result holders.

        Args:
            cfg: Full pipeline configuration.
        """
        self.cfg = cfg
        self.results: pd.DataFrame | None = None
        self.importances: pd.DataFrame | None = None
        self.figures: list[Path] = []

    def evaluate(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Score every model on the test set at ``threshold.default`` and at its tuned threshold.

        Logs the metrics of each model.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.

        Returns:
            One row per (model, threshold type) with the ``compute_metrics`` columns
            plus ``model``, ``threshold_type``, ``threshold``, ``cv_score`` and ``train_seconds``.
        """
        raise NotImplementedError

    def feature_importance(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Compute permutation importance on the test set and native importance per model.

        Uses ``evaluation.permutation_repeats``, ``evaluation.permutation_scoring``
        and the config seed. Native importance comes from each model's
        ``native_importance`` (None for HGB).

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.

        Returns:
            Long table with columns ``model, feature, kind, importance, std``
            (``kind`` is ``"permutation"`` or ``"native"``).
        """
        raise NotImplementedError

    def plot_confusion_matrices(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series, directory: str | Path
    ) -> Path:
        """Plot one confusion matrix per model at its tuned threshold and save the figure.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.
            directory: Folder to write the PNG to.

        Returns:
            Path of the saved figure.
        """
        raise NotImplementedError

    def plot_pr_curves(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series, directory: str | Path
    ) -> Path:
        """Plot the precision-recall curve of every model on one figure and save it.

        Marks each model's tuned threshold and shows the positive-class baseline.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.
            directory: Folder to write the PNG to.

        Returns:
            Path of the saved figure.
        """
        raise NotImplementedError

    def select_best(self, results: pd.DataFrame) -> str:
        """Pick the best model from the tuned-threshold rows of ``results``.

        Ranks by ``evaluation.selection_metric`` (F2). Models within
        ``evaluation.tie_tolerance`` of the best are treated as tied and ranked by
        ``evaluation.tie_breaker`` (PR-AUC), then by shorter training time. A tie
        is logged honestly instead of claiming a clear winner.

        Args:
            results: Output of ``evaluate``.

        Returns:
            Name of the selected model.

        Raises:
            ValueError: If ``results`` is empty or lacks the selection columns.
        """
        raise NotImplementedError

    def save_report(self, directory: str | Path) -> list[Path]:
        """Write the metrics and importance tables (CSV) and a short text summary.

        Args:
            directory: Target folder (normally ``paths.report_dir``); created if needed.

        Returns:
            Paths of the files written.

        Raises:
            RuntimeError: If ``evaluate`` has not been run yet.
        """
        raise NotImplementedError
