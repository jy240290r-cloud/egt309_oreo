"""Random forest model: a non-linear ensemble that captures threshold effects."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random forest with balanced class weights.

    Trees do not need scaled inputs, so no scaler is used. Native importance is
    the impurity-based ``feature_importances_``.
    """

    name = "random_forest"
    needs_scaling = False

    def build_estimator(self) -> RandomForestClassifier:
        """Create a ``RandomForestClassifier`` with the config defaults and the config seed.

        Returns:
            Unfitted ``RandomForestClassifier``.
        """
        raise NotImplementedError

    def native_importance(self, fitted_pipeline: Pipeline) -> pd.Series:
        """Return ``feature_importances_`` indexed by feature name, sorted descending.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline``.

        Returns:
            Series of impurity-based importances.
        """
        raise NotImplementedError
