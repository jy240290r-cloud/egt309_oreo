"""Logistic regression model: the simple, interpretable baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .base import MODEL_STEP, BaseModel


class LogisticRegressionModel(BaseModel):
    """L2-regularised logistic regression on standardised features.

    Scaling is enabled because the coefficients (and the regularisation) depend
    on feature scale. Native importance is the absolute coefficient per feature.
    """

    name = "logistic_regression"
    needs_scaling = True

    def build_estimator(self) -> LogisticRegression:
        """Create a ``LogisticRegression`` with the config defaults and the config seed.

        ``class_weight="balanced"`` is used unless the config sets ``class_weight``
        explicitly, because the target is imbalanced (~80/20).

        Returns:
            Unfitted ``LogisticRegression``.
        """
        params = {"class_weight": "balanced", **self.defaults, "random_state": self.random_state}
        return LogisticRegression(**params)

    def native_importance(self, fitted_pipeline: Pipeline) -> pd.Series:
        """Return the absolute coefficients, indexed by feature name, sorted descending.

        Coefficients are comparable because the features are standardised.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline``.

        Returns:
            Series of absolute coefficients.

        Raises:
            sklearn.exceptions.NotFittedError: If the pipeline has not been fitted.
        """
        estimator = fitted_pipeline.named_steps[MODEL_STEP]
        check_is_fitted(estimator)
        # Every step before the estimator (preprocessor, scaler) passes the feature names through.
        feature_names = fitted_pipeline[:-1].get_feature_names_out()
        return (
            pd.Series(np.abs(estimator.coef_[0]), index=feature_names, name="abs_coefficient")
            .sort_values(ascending=False)
        )
