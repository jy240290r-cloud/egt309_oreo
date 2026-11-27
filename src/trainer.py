"""Model training: hyperparameter search, decision-threshold tuning and saving.

Everything here uses the **training data only**. Hyperparameters are chosen
with ``RandomizedSearchCV`` over ``StratifiedKFold`` folds, and the decision
threshold is chosen on out-of-fold probabilities, so the test set stays unseen
until evaluation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from models import get_model
from models.base import MODEL_STEP

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

        Stratified folds keep the ~80/20 class ratio in every fold, so no fold ends
        up with too few maintenance cases to score fairly.

        Returns:
            Splitter with ``tuning.cv_folds`` folds and the config seed.
        """
        return StratifiedKFold(
            n_splits=self.cfg["tuning"]["cv_folds"], shuffle=True, random_state=self.random_state
        )

    def tune(
        self, name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[Pipeline, dict[str, Any], float]:
        """Search hyperparameters for one model with ``RandomizedSearchCV``.

        Uses the model's ``search_space()``, ``tuning.n_iter``, ``tuning.scoring``,
        ``tuning.n_jobs`` and ``cv_splitter()``. When ``tuning.enabled`` is false,
        the default pipeline is cross-validated instead of searched.

        RandomizedSearchCV is used instead of GridSearchCV because it tries a fixed
        number of random combinations, which is much faster on 80,000 rows while
        still exploring the whole search space.

        Args:
            name: Registry name of the model.
            X_train: Training features.
            y_train: Training target.

        Returns:
            ``(best_pipeline, best_params, best_cv_score)``; the pipeline is refitted
            on all of ``X_train``.
        """
        tuning = self.cfg["tuning"]
        model = get_model(name, self.cfg)
        pipeline = model.build_pipeline()
        space = model.search_space()

        if not tuning["enabled"] or not space:
            logger.info("%s: tuning disabled, cross-validating the config defaults.", name)
            scores = cross_val_score(
                pipeline, X_train, y_train,
                scoring=tuning["scoring"], cv=self.cv_splitter(), n_jobs=tuning["n_jobs"],
            )
            pipeline.fit(X_train, y_train)
            return pipeline, model.defaults, float(scores.mean())

        # Never ask for more candidates than the search space contains
        # (logistic regression only has 6 values of C).
        n_combinations = math.prod(len(values) for values in space.values())
        n_iter = min(tuning["n_iter"], n_combinations)
        logger.info(
            "%s: RandomizedSearchCV with %d of %d combinations x %d folds, scoring=%s",
            name, n_iter, n_combinations, tuning["cv_folds"], tuning["scoring"],
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=space,
            n_iter=n_iter,
            scoring=tuning["scoring"],
            cv=self.cv_splitter(),
            n_jobs=tuning["n_jobs"],
            random_state=self.random_state,
            refit=True,  # refit the best combination on the whole training set
        )
        search.fit(X_train, y_train)

        # Search keys look like "model__max_depth"; strip the step prefix for readability.
        prefix = f"{MODEL_STEP}__"
        best_params = {key.removeprefix(prefix): value for key, value in search.best_params_.items()}
        logger.info("%s: best CV %s = %.4f with %s", name, tuning["scoring"], search.best_score_, best_params)
        return search.best_estimator_, best_params, float(search.best_score_)

    def tune_threshold(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """Choose the decision threshold that maximises F-beta on out-of-fold probabilities.

        Out-of-fold probabilities come from ``cross_val_predict`` with
        ``cv_splitter()``. Candidates run from ``threshold.search_min`` to
        ``threshold.search_max`` in steps of ``threshold.search_step``; those with
        precision below ``threshold.min_precision`` are rejected. Returns
        ``threshold.default`` when ``threshold.tune`` is false or no candidate qualifies
        (with a WARNING in the latter case).

        Out-of-fold means every training row is scored by a model that did not see
        it, so the threshold is not chosen on over-confident training predictions.

        Args:
            pipeline: Unfitted (or fitted; it is cloned) pipeline with the chosen hyperparameters.
            X_train: Training features.
            y_train: Training target.

        Returns:
            The chosen threshold.
        """
        thr_cfg = self.cfg["threshold"]
        if not thr_cfg["tune"]:
            return float(thr_cfg["default"])

        oof_prob = cross_val_predict(
            clone(pipeline), X_train, y_train,
            cv=self.cv_splitter(), method="predict_proba", n_jobs=self.cfg["tuning"]["n_jobs"],
        )[:, 1]

        # Half a step is added so search_max itself is included despite float rounding.
        candidates = np.round(
            np.arange(thr_cfg["search_min"], thr_cfg["search_max"] + thr_cfg["search_step"] / 2,
                      thr_cfg["search_step"]),
            4,
        )
        best = self.best_fbeta_threshold(
            np.asarray(y_train), oof_prob, candidates, thr_cfg["beta"], thr_cfg["min_precision"]
        )
        if best is None:
            logger.warning(
                "No threshold reached min_precision=%.2f; falling back to default %.2f.",
                thr_cfg["min_precision"], thr_cfg["default"],
            )
            return float(thr_cfg["default"])
        return best

    def train(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
        """Tune, threshold-tune and time one model.

        Args:
            name: Registry name of the model.
            X_train: Training features.
            y_train: Training target.

        Returns:
            The trained model and its training metadata.
        """
        logger.info("Training %s ...", name)
        start = time.perf_counter()
        pipeline, best_params, cv_score = self.tune(name, X_train, y_train)
        threshold = self.tune_threshold(pipeline, X_train, y_train)
        seconds = time.perf_counter() - start
        logger.info("%s: threshold = %.2f, trained in %.1f s", name, threshold, seconds)
        return TrainedModel(
            name=name,
            pipeline=pipeline,
            best_params=best_params,
            cv_score=cv_score,
            threshold=threshold,
            train_seconds=seconds,
        )

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
        names = self.cfg["models"].get("enabled") or []
        if not names:
            raise ValueError("No models enabled: set models.enabled in the config or use --models.")

        # Fail fast: build every model's pipeline before training anything, so a broken or
        # unfinished model class stops the run in seconds instead of after the slow ones.
        for name in names:
            get_model(name, self.cfg).build_pipeline()

        return {name: self.train(name, X_train, y_train) for name in names}

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
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, model in trained.items():
            path = directory / f"{name}.joblib"
            joblib.dump(model, path)
            logger.info("Saved %s -> %s", name, path)
            paths.append(path)
        return paths

    @staticmethod
    def best_fbeta_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: np.ndarray,
        beta: float,
        min_precision: float,
    ) -> float | None:
        """Return the candidate threshold with the highest F-beta that meets ``min_precision``.

        With beta = 2, recall counts twice as much as precision, because missing a
        machine that needs maintenance costs more than a false alarm. The precision
        floor stops the search from picking a threshold that flags almost everything.

        Args:
            y_true: True 0/1 labels.
            y_prob: Predicted probabilities of class 1.
            thresholds: Candidate thresholds to evaluate.
            beta: F-beta weight (beta > 1 favours recall).
            min_precision: Minimum precision a candidate must reach.

        Returns:
            The best threshold, or None if no candidate reaches ``min_precision``.
        """
        best_threshold, best_score = None, -1.0
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            if precision_score(y_true, y_pred, zero_division=0) < min_precision:
                continue
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_threshold, best_score = float(threshold), score
        return best_threshold