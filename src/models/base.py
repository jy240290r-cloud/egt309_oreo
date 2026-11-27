"""Abstract base class that every model in the pipeline follows.

A model class only has to say which estimator it uses and how to read its
native feature importance. Building the full Pipeline (preprocessor, optional
scaler, estimator) and the search space is shared here, so the three model
classes contain no duplicated code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

#: Name of the estimator step in every model Pipeline. Search-space keys use it as prefix.
MODEL_STEP: str = "model"


class BaseModel(ABC):
    """Common interface for a configurable, tunable classifier.

    Subclasses set ``name`` (the key of their block under ``models`` in the
    config), optionally set ``needs_scaling``, and implement
    ``build_estimator`` and ``native_importance``.

    Attributes:
        name: Registry key, also the model's section name in ``cfg["models"]``.
        needs_scaling: Whether a ``StandardScaler`` is placed before the estimator.
        cfg: Full pipeline configuration.
        model_cfg: This model's block from ``cfg["models"][name]``.
        random_state: Seed from ``cfg["random_state"]``.
    """

    name: ClassVar[str] = ""
    needs_scaling: ClassVar[bool] = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store the config and look up this model's settings.

        Args:
            cfg: Full pipeline configuration.

        Raises:
            TypeError: If the subclass did not set ``name``.
            KeyError: If the config has no ``models.<name>`` block, or the block
                lacks ``defaults`` or ``search_space``.
        """
        if not self.name:
            raise TypeError(f"{type(self).__name__} must set the class attribute 'name'.")

        models_cfg = cfg.get("models", {})
        if self.name not in models_cfg:
            raise KeyError(f"Config has no 'models.{self.name}' section for {type(self).__name__}.")

        model_cfg = models_cfg[self.name]
        missing = [key for key in ("defaults", "search_space") if key not in model_cfg]
        if missing:
            raise KeyError(f"Config section 'models.{self.name}' is missing: {', '.join(missing)}.")

        self.cfg = cfg
        self.model_cfg: dict[str, Any] = model_cfg
        self.random_state: int = cfg["random_state"]

    @property
    def defaults(self) -> dict[str, Any]:
        """Default hyperparameters from config (a copy, so callers cannot change the config).

        Returns:
            Mapping of estimator parameter name to value.
        """
        return dict(self.model_cfg["defaults"] or {})

    @abstractmethod
    def build_estimator(self) -> BaseEstimator:
        """Create the unfitted sklearn estimator with the default hyperparameters from config.

        Implementations must pass ``random_state=self.random_state`` where the
        estimator supports it.

        Returns:
            Unfitted sklearn classifier that supports ``predict_proba``.
        """

    def build_pipeline(self) -> Pipeline:
        """Build the full unfitted Pipeline: preprocessor, optional scaler, estimator.

        The estimator step is always named ``model`` so search-space keys can be
        prefixed with ``model__``.

        Returns:
            Unfitted sklearn Pipeline.
        """
        steps: list[tuple[str, Any]] = [("preprocess", build_preprocessor(self.cfg))]
        if self.needs_scaling:
            steps.append(("scale", StandardScaler()))
        steps.append((MODEL_STEP, self.build_estimator()))
        return Pipeline(steps)

    def search_space(self) -> dict[str, list[Any]]:
        """Return the hyperparameter search space from config, keyed for the Pipeline.

        Returns:
            Mapping such as ``{"model__C": [0.01, 0.1, 1.0]}`` for ``RandomizedSearchCV``.
        """
        space = self.model_cfg["search_space"] or {}
        return {f"{MODEL_STEP}__{param}": list(values) for param, values in space.items()}

    @abstractmethod
    def native_importance(self, fitted_pipeline: Pipeline) -> pd.Series | None:
        """Return the model's built-in feature importance, if it has one.

        Args:
            fitted_pipeline: Pipeline returned by ``build_pipeline`` after fitting.

        Returns:
            Series indexed by feature name, or None when the model has no native
            importance (permutation importance is used instead).
        """

    def __repr__(self) -> str:
        """Short representation used in log messages."""
        return f"{type(self).__name__}(name={self.name!r})"
