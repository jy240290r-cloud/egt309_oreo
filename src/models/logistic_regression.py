"""Logistic regression model: the simple, interpretable baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .base import MODEL_STEP, BaseModel


class LogisticRegressionModel(BaseModel):
    """Regularised (L1, L2 or elastic-net) logistic regression on standardised features.

    Scaling is enabled because the coefficients (and the regularisation) depend
    on feature scale. Saga's convergence also needs it. Native importance is the
    absolute coefficient per feature.

    Both ``C`` and the L1/L2 mix ``l1_ratio`` are tuned (0.0 = pure L2,
    1.0 = pure L1, values in between = elastic-net). Why: the EDA found that
    humidity, pressure and energy_consumption carry no signal for
    ``maintenance_required``. An L1-leaning mix can shrink their coefficients to
    exactly zero, which acts as automatic feature selection, while pure L2 only
    makes them small. The search decides on the training folds which mix
    generalises best.

    ``l1_ratio`` requires the ``saga`` solver (set in config), the only solver
    that supports the whole 0..1 range. The penalty is set through ``l1_ratio``
    alone. No ``penalty`` argument is passed, because sklearn >= 1.8 deprecates it.
    The search space itself comes from config through ``BaseModel.search_space``.
    """

    name = "logistic_regression"
    needs_scaling = True

    def build_estimator(self) -> LogisticRegression:
        """Create a ``LogisticRegression`` with the config defaults and the config seed.

        ``class_weight="balanced"`` is used unless the config sets ``class_weight``
        explicitly, because the target is imbalanced (~80/20). ``solver``,
        ``max_iter``, ``C`` and ``l1_ratio`` all come from the config defaults and
        are not hard-coded, so the search can override ``C`` and ``l1_ratio``.
        ``random_state`` always comes from the top-level config seed and overrides
        any value in ``defaults``.

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
