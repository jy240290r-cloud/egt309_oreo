"""Histogram gradient boosting model: a fast boosted-tree ensemble from scikit-learn."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .base import BaseModel


class HistGradientBoostingModel(BaseModel):
    """``HistGradientBoostingClassifier`` with balanced class weights.

    Part of scikit-learn, so no extra dependency is needed. It has no native
    feature importance, so evaluation falls back to permutation importance.
    """

    name = "hist_gradient_boosting"
    needs_scaling = False

    def build_estimator(self) -> HistGradientBoostingClassifier:
        """Create a ``HistGradientBoostingClassifier`` with the config defaults and the config seed.

        Returns:
            Unfitted ``HistGradientBoostingClassifier``.
        """
        raise NotImplementedError

    def native_importance(self, fitted_pipeline: Pipeline) -> None:
        """Return None: this estimator exposes no native feature importance.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline`` (unused).

        Returns:
            None, so permutation importance is used.
        """
        raise NotImplementedError
