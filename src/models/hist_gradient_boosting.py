"""Histogram gradient boosting model: a fast boosted-tree ensemble from scikit-learn."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .base import BaseModel


class HistGradientBoostingModel(BaseModel):
    """``HistGradientBoostingClassifier`` with balanced class weights.

    Why gradient boosting for this problem (see eda.ipynb, Sections 4 and 6):

    * Like the random forest, it is built from decision trees, so it can learn
      the temperature > 90 °C and |vibration| > 80 mm/s thresholds directly.
    * Boosting builds trees one after another, and each new tree focuses on the
      readings the previous trees got wrong. This often gives better accuracy
      than a random forest, which builds its trees independently.
    * The "histogram" version groups each feature's values into bins before
      splitting, so it trains quickly on our 100,000 rows.
    * It handles missing values natively, although the pipeline imputes
      temperature anyway so all three models see the same inputs.

    Part of scikit-learn, so no extra dependency is needed. Trees do not need
    scaled inputs, so no scaler is used. It has no native feature importance,
    so evaluation falls back to permutation importance.
    """

    name = "hist_gradient_boosting"
    needs_scaling = False

    def build_estimator(self) -> HistGradientBoostingClassifier:
        """Create a ``HistGradientBoostingClassifier`` with the config defaults and the config seed.

        ``class_weight="balanced"`` is used unless the config sets ``class_weight``
        explicitly, because the target is imbalanced (~80/20). Without it the
        model would favour the majority class and miss maintenance cases.

        Returns:
            Unfitted ``HistGradientBoostingClassifier``.
        """
        # Order matters: later keys win, so config defaults override the balanced
        # fallback, and the seed from config always overrides both.
        params = {"class_weight": "balanced", **self.defaults, "random_state": self.random_state}
        return HistGradientBoostingClassifier(**params)

    def native_importance(self, fitted_pipeline: Pipeline) -> None:
        """Return None: this estimator exposes no native feature importance.

        ``HistGradientBoostingClassifier`` has no ``feature_importances_`` or
        ``coef_`` attribute, so the evaluation step uses permutation importance
        on the test set for this model instead.

        Args:
            fitted_pipeline: Fitted pipeline from ``build_pipeline`` (unused).

        Returns:
            None, so permutation importance is used.
        """
        return None