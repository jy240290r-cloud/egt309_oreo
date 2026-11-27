"""Logistic regression model: the simple, interpretable baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """L2-regularised logistic regression on standardised features.

    Scaling is enabled because the coefficients (and the regularisation) depend
    on feature scale. Native importance is the absolute coefficient per feature.
    """

    name = "logistic_regression"
    needs_scaling = True

    def build_estimator(self) -> LogisticRegression:
        """Create a ``LogisticRegression`` with the config defaults and the config seed.

        Returns:
            Unfitted ``LogisticRegression``.
        """
        raise NotImplementedError

    def native_importance(self, fitted_pipeline: Pipeline) -> pd.Series:
        """Return the absolute coefficients, indexed by feature name, sorted descending.

        Coefficients are comparable because the features are standardised.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline``.

        Returns:
            Series of absolute coefficients.
        """
        raise NotImplementedError
