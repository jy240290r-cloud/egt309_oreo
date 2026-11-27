"""Random forest model: a non-linear ensemble that captures threshold effects."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .base import MODEL_STEP, BaseModel


class RandomForestModel(BaseModel):
    """Random forest with balanced class weights.
 
    Why a random forest for this problem (see eda.ipynb, Sections 4 and 6):

    * Maintenance depends on thresholds (temperature > 90 °C, |vibration| > 80 mm/s).
      Decision trees split on exactly this kind of "value > cut-off" rule, so the
      forest can learn the thresholds directly.
    * The high readings are kept on purpose (they are the overheating cases), and
      trees are robust to outliers.
    * Averaging many trees reduces the overfitting a single decision tree tends to have.

    Trees do not need scaled inputs, so no scaler is used. Native importance is
    the impurity-based ``feature_importances_``.
    """

    name = "random_forest"
    needs_scaling = False

    def build_estimator(self) -> RandomForestClassifier:
        """Create a ``RandomForestClassifier`` with the config defaults and the config seed.

        ``class_weight="balanced"`` is used unless the config sets ``class_weight``
        explicitly, because the target is imbalanced (~80/20). Without it the
        forest would favour the majority class and miss maintenance cases.

        Returns:
            Unfitted ``RandomForestClassifier``.
        """
        # Order matters: later keys win, so config defaults override the balanced
        # fallback, and the seed from config always overrides both.
        params = {"class_weight": "balanced", **self.defaults, "random_state": self.random_state}
        return RandomForestClassifier(**params)

    def native_importance(self, fitted_pipeline: Pipeline) -> pd.Series:
        """Return ``feature_importances_`` indexed by feature name, sorted descending.

        Impurity-based importance measures how much each feature reduces impurity
        across all trees. It can favour features with many unique values, so the
        evaluation step also reports permutation importance for comparison.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline``.

        Returns:
            Series of impurity-based importances (they sum to 1).

        Raises:
            sklearn.exceptions.NotFittedError: If the pipeline has not been fitted.
        """
        estimator = fitted_pipeline.named_steps[MODEL_STEP]
        check_is_fitted(estimator)
        # Every step before the estimator (the preprocessor) passes the feature names through.
        feature_names = fitted_pipeline[:-1].get_feature_names_out()
        return (
            pd.Series(estimator.feature_importances_, index=feature_names, name="impurity_importance")
            .sort_values(ascending=False)
        )