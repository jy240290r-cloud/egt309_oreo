"""Model training: hyperparameter search, decision-threshold tuning and saving.

Everything here uses the **training data only**. Hyperparameters are chosen
with ``RandomizedSearchCV`` over ``StratifiedKFold`` folds, and the decision
threshold is chosen on out-of-fold probabilities, so the test set stays unseen
until evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted model pipeline together with what was learned while training it.

    Attributes:
        name: Registry name of the model (e.g. ``"random_forest"``).
        pipeline: Pipeline refitted on the full training set with ``best_params``.
        best_params: Hyperparameters chosen by the search (defaults if tuning is off).
        cv_score: Mean cross-validated ``tuning.scoring`` score of ``best_params``.
        threshold: Decision threshold tuned on out-of-fold probabilities.
        train_seconds: Wall-clock time for tuning, threshold tuning and refitting.
    """

    name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_score: float
    threshold: float
    train_seconds: float


class ModelTrainer:
    """Tune, fit and save every enabled model.

    Attributes:
        cfg: Full pipeline configuration.
        random_state: Seed from ``cfg["random_state"]``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store the config and the seed.

        Args:
            cfg: Full pipeline configuration.
        """
        self.cfg = cfg
        self.random_state: int = cfg["random_state"]

    def cv_splitter(self) -> StratifiedKFold:
        """Create the shuffled ``StratifiedKFold`` used for both search and threshold tuning.

        Returns:
            Splitter with ``tuning.cv_folds`` folds and the config seed.
        """
        raise NotImplementedError

    def tune(
        self, name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[Pipeline, dict[str, Any], float]:
        """Search hyperparameters for one model with ``RandomizedSearchCV``.

        Uses the model's ``search_space()``, ``tuning.n_iter``, ``tuning.scoring``,
        ``tuning.n_jobs`` and ``cv_splitter()``. When ``tuning.enabled`` is false,
        the default pipeline is cross-validated instead of searched.

        Args:
            name: Registry name of the model.
            X_train: Training features.
            y_train: Training target.

        Returns:
            ``(best_pipeline, best_params, best_cv_score)``; the pipeline is refitted
            on all of ``X_train``.
        """
        raise NotImplementedError

    def tune_threshold(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """Choose the decision threshold that maximises F-beta on out-of-fold probabilities.

        Out-of-fold probabilities come from ``cross_val_predict`` with
        ``cv_splitter()``. Candidates run from ``threshold.search_min`` to
        ``threshold.search_max`` in steps of ``threshold.search_step``; those with
        precision below ``threshold.min_precision`` are rejected. Returns
        ``threshold.default`` when ``threshold.tune`` is false or no candidate qualifies
        (with a WARNING in the latter case).

        Args:
            pipeline: Unfitted (or fitted; it is cloned) pipeline with the chosen hyperparameters.
            X_train: Training features.
            y_train: Training target.

        Returns:
            The chosen threshold.
        """
        raise NotImplementedError

    def train(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
        """Tune, threshold-tune and time one model.

        Args:
            name: Registry name of the model.
            X_train: Training features.
            y_train: Training target.

        Returns:
            The trained model and its training metadata.
        """
        raise NotImplementedError

    def fit_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, TrainedModel]:
        """Train every model listed in ``models.enabled``.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            Mapping of model name to ``TrainedModel``, in config order.

        Raises:
            ValueError: If no models are enabled.
        """
        raise NotImplementedError

    def save(self, trained: dict[str, TrainedModel], directory: str | Path) -> list[Path]:
        """Save each trained model with joblib as ``<directory>/<name>.joblib``.

        The whole ``TrainedModel`` is saved, so the tuned threshold travels with
        the pipeline. The directory is created if needed.

        Args:
            trained: Output of ``fit_all``.
            directory: Target folder (normally ``paths.model_dir``).

        Returns:
            Paths of the files written.
        """
        raise NotImplementedError

    @staticmethod
    def best_fbeta_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: np.ndarray,
        beta: float,
        min_precision: float,
    ) -> float | None:
        """Return the candidate threshold with the highest F-beta that meets ``min_precision``.

        Args:
            y_true: True 0/1 labels.
            y_prob: Predicted probabilities of class 1.
            thresholds: Candidate thresholds to evaluate.
            beta: F-beta weight (beta > 1 favours recall).
            min_precision: Minimum precision a candidate must reach.

        Returns:
            The best threshold, or None if no candidate reaches ``min_precision``.
        """
        raise NotImplementedError
