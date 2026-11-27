"""Preprocessing: sklearn-compatible transformers, the preprocessor pipeline and the data split.

Implements the EDA decisions (eda.ipynb Section 9):

* ``vibration`` has sign errors -> ``vibration_abs = abs(vibration)``.
* ``temperature`` is 14% missing (MCAR) -> median imputation fitted on training
  data only, plus a ``temperature_missing`` flag. High outliers are kept.
* Engineered threshold features ``temp_over_90`` and ``vib_over_80``
  (thresholds from config).
* Identifiers are never features. Derived (leaky) columns are dropped unless
  ``features.feature_set`` is ``all``.

Every transformer is fitted inside the model Pipeline, so each CV fold fits its
own imputer and no test information leaks into training. The preprocessor
returns the selected feature columns **unscaled**; scaling is added per model.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

#: Allowed values of ``features.feature_set``.
FEATURE_SETS: tuple[str, ...] = ("sensors_only", "all")


class VibrationAbsTransformer(BaseEstimator, TransformerMixin):
    """Fix the vibration sign error by adding the absolute value as a new column.

    Stateless: ``fit`` learns nothing.
    """

    def __init__(
        self,
        column: str = "vibration",
        output_column: str = "vibration_abs",
        drop_original: bool = True,
    ) -> None:
        """Store the column names (sklearn convention: no logic in ``__init__``).

        Args:
            column: Raw vibration column.
            output_column: Name of the absolute-value column to create.
            drop_original: Whether to drop the raw (signed) column afterwards.
        """
        self.column = column
        self.output_column = output_column
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> VibrationAbsTransformer:
        """Check that the vibration column is present. Nothing is learned.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            self

        Raises:
            KeyError: If ``column`` is not in ``X``.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with ``output_column = abs(column)``.

        Args:
            X: Input features.

        Returns:
            Transformed copy of ``X``.
        """
        raise NotImplementedError


class TemperatureImputer(BaseEstimator, TransformerMixin):
    """Median-impute temperature with a median learned in ``fit``, and flag missing rows.

    The flag is computed before imputation so it records the original missingness.
    """

    def __init__(
        self,
        column: str = "temperature",
        strategy: str = "median",
        add_missing_flag: bool = True,
        flag_column: str = "temperature_missing",
    ) -> None:
        """Store the imputation settings (sklearn convention: no logic in ``__init__``).

        Args:
            column: Column to impute.
            strategy: Statistic used for imputation (only ``"median"`` is supported).
            add_missing_flag: Whether to add the 0/1 missing-value flag.
            flag_column: Name of the flag column.
        """
        self.column = column
        self.strategy = strategy
        self.add_missing_flag = add_missing_flag
        self.flag_column = flag_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TemperatureImputer:
        """Learn the fill value (``fill_value_``) from the training data only.

        Args:
            X: Training features.
            y: Ignored.

        Returns:
            self

        Raises:
            KeyError: If ``column`` is not in ``X``.
            ValueError: If ``strategy`` is not supported or the column is entirely null.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the missing flag (optional) and fill nulls with the learned value.

        Logs how many values were imputed.

        Args:
            X: Input features.

        Returns:
            Transformed copy of ``X``.
        """
        raise NotImplementedError


class ThresholdFeatureAdder(BaseEstimator, TransformerMixin):
    """Add binary features for readings above the EDA thresholds.

    Creates ``temp_over_<temp_threshold>`` and ``vib_over_<vib_threshold>``
    (e.g. ``temp_over_90``, ``vib_over_80``). Stateless.
    """

    def __init__(
        self,
        temp_threshold: float,
        vib_threshold: float,
        temp_column: str = "temperature",
        vib_column: str = "vibration_abs",
    ) -> None:
        """Store the thresholds (read from config, no defaults on purpose).

        Args:
            temp_threshold: Temperature above which ``temp_over_*`` is 1.
            vib_threshold: Absolute vibration above which ``vib_over_*`` is 1.
            temp_column: Temperature column (after imputation).
            vib_column: Absolute vibration column.
        """
        self.temp_threshold = temp_threshold
        self.vib_threshold = vib_threshold
        self.temp_column = temp_column
        self.vib_column = vib_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ThresholdFeatureAdder:
        """Check that the input columns are present. Nothing is learned.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            self

        Raises:
            KeyError: If a required column is missing.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the two 0/1 threshold columns added.

        Args:
            X: Input features.

        Returns:
            Transformed copy of ``X``.
        """
        raise NotImplementedError

    def feature_names(self) -> list[str]:
        """Return the names of the columns this transformer adds.

        Returns:
            ``[f"temp_over_{temp_threshold}", f"vib_over_{vib_threshold}"]`` with
            integer-valued thresholds written without decimals.
        """
        raise NotImplementedError


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Keep only the configured model feature columns, in a fixed order.

    Derived categorical columns (``features.categorical_derived``) are one-hot
    encoded here when ``feature_set`` is ``all``.
    """

    def __init__(self, columns: list[str], categorical: list[str] | None = None) -> None:
        """Store the columns to keep (sklearn convention: no logic in ``__init__``).

        Args:
            columns: Feature columns to output, in order.
            categorical: Columns among ``columns`` to one-hot encode.
        """
        self.columns = columns
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureSelector:
        """Check the columns exist and learn the one-hot categories, if any.

        Args:
            X: Training features.
            y: Ignored.

        Returns:
            self

        Raises:
            KeyError: If a selected column is missing.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the selected (and encoded) columns.

        Args:
            X: Input features.

        Returns:
            DataFrame with the model features.
        """
        raise NotImplementedError

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the output column names (used for feature importance).

        Args:
            input_features: Ignored, present for sklearn compatibility.

        Returns:
            Output feature names.
        """
        raise NotImplementedError


def get_feature_columns(cfg: dict[str, Any]) -> list[str]:
    """List the model feature columns produced for the configured feature set.

    For ``sensors_only``: ``temperature``, ``vibration_abs``, ``humidity``,
    ``pressure``, ``energy_consumption`` plus the enabled engineered features
    (``temperature_missing``, ``temp_over_90``, ``vib_over_80``). For ``all`` the
    derived columns are appended.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Ordered list of feature column names.

    Raises:
        ValueError: If ``features.feature_set`` is not one of ``FEATURE_SETS``.
    """
    raise NotImplementedError


def build_preprocessor(cfg: dict[str, Any]) -> Pipeline:
    """Build the unfitted preprocessing Pipeline from the config.

    Steps: ``VibrationAbsTransformer`` -> ``TemperatureImputer`` ->
    ``ThresholdFeatureAdder`` (if enabled) -> ``FeatureSelector``. The output is
    **unscaled**; models that need scaling add it themselves.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Unfitted sklearn Pipeline.
    """
    raise NotImplementedError


def split_data(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Drop non-feature columns and make a stratified train/test split.

    Identifiers are always dropped. Derived columns are dropped unless
    ``features.feature_set`` is ``all``, in which case a WARNING about target
    leakage is logged. Uses ``split.test_size`` and the top-level ``random_state``.

    Args:
        df: Raw readings as loaded from SQLite.
        cfg: Full pipeline configuration.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        KeyError: If the target or a configured column is missing from ``df``.
        ValueError: If ``features.feature_set`` is not one of ``FEATURE_SETS``.
    """
    raise NotImplementedError
