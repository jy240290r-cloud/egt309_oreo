"""Preprocessing: sklearn-compatible transformers, the preprocessor pipeline and the data split.

Implements the EDA decisions (eda.ipynb Section 9):

* ``vibration`` has sign errors -> ``vibration_abs = abs(vibration)``.
* ``temperature`` is 14% missing (MCAR) -> median imputation fitted on training
  data only, plus a ``temperature_missing`` flag. High outliers are kept.
* Engineered threshold features ``temp_over_90`` and ``vib_over_80``
  (thresholds from config, strict ``>`` as in the EDA).
* Identifiers are never features. Derived (leaky) columns are dropped unless
  ``features.feature_set`` is ``all``; then ``failure_type`` is one-hot encoded.

Every transformer is fitted inside the model Pipeline, so each CV fold fits its
own imputer and no test information leaks into training. The preprocessor
returns the selected feature columns **unscaled**; scaling is added per model.

All transformers take and return pandas DataFrames, so column names survive
to the model and feature importances can be reported by name.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from config import FEATURE_SETS

logger = logging.getLogger(__name__)

#: Raw and engineered column names. These are the dataset schema, not tunable settings.
TEMPERATURE_COL: str = "temperature"
TEMP_MISSING_COL: str = "temperature_missing"
VIBRATION_COL: str = "vibration"
VIBRATION_ABS_COL: str = "vibration_abs"
FAILURE_TYPE_COL: str = "failure_type"

__all__ = [
    "FEATURE_SETS",
    "DataFrameTransformer",
    "VibrationAbsTransformer",
    "TemperatureImputer",
    "ThresholdFeatureAdder",
    "FailureTypeEncoder",
    "FeatureSelector",
    "get_feature_columns",
    "build_preprocessor",
    "split_data",
]


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Shared base for the DataFrame-in, DataFrame-out transformers in this module.

    It holds the plumbing every transformer needs, so subclasses only describe
    what is specific to them:

    * ``_required_columns()`` - columns that must be present in ``X``.
    * ``_learn(X)`` - learn state from training data (default: nothing).
    * ``_apply(X)`` - transform a copy of ``X`` and return it.

    ``fit`` always sets ``is_fitted_ = True`` (also for stateless transformers),
    so ``check_is_fitted`` passes when the fitted pipeline is reused on test data.
    """

    def _required_columns(self) -> list[str]:
        """Return the columns ``X`` must contain. Override in subclasses."""
        return []

    def _learn(self, X: pd.DataFrame) -> None:
        """Learn state from the training data. Stateless transformers keep this no-op."""

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a copy of ``X``. Must be overridden."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DataFrameTransformer:
        """Check the input, learn any state and record the input and output column names.

        Args:
            X: Training features.
            y: Ignored, present for sklearn compatibility.

        Returns:
            self

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            KeyError: If a required column is missing.
        """
        self._check_input(X)
        self._learn(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Transforming zero rows is a cheap way to get the exact output columns.
        self.feature_names_out_ = list(self._apply(X.iloc[:0].copy()).columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform ``X`` with the fitted state, without modifying the caller's DataFrame.

        Args:
            X: Input features.

        Returns:
            Transformed copy of ``X``.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self, "is_fitted_")
        self._check_input(X)
        return self._apply(X.copy())

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the output column names (used for feature importance).

        Args:
            input_features: Ignored; the names were recorded in ``fit``.

        Returns:
            Array of output feature names.
        """
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _check_input(self, X: pd.DataFrame) -> None:
        """Raise a clear error if ``X`` is not a DataFrame or lacks required columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame, got {type(X).__name__}.")
        missing = [column for column in self._required_columns() if column not in X.columns]
        if missing:
            raise KeyError(f"{type(self).__name__} is missing required column(s): {missing}.")


class VibrationAbsTransformer(DataFrameTransformer):
    """Fix the vibration sign error by adding the absolute value as a new column.

    Stateless: ``fit`` learns nothing.
    """

    def __init__(
        self,
        column: str = VIBRATION_COL,
        output_column: str = VIBRATION_ABS_COL,
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

    def _required_columns(self) -> list[str]:
        """The raw vibration column is required."""
        return [self.column]

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add ``output_column = abs(column)`` and optionally drop the raw column."""
        X[self.output_column] = X[self.column].abs()
        if self.drop_original and self.output_column != self.column:
            X = X.drop(columns=self.column)
        return X


class TemperatureImputer(DataFrameTransformer):
    """Median-impute temperature with a median learned in ``fit``, and flag missing rows.

    The flag is computed before imputation so it records the original missingness.
    High outliers are kept; the median is used because they do not move it.

    Attributes (after ``fit``):
        fill_value_: Median of the training data, used to fill nulls.
        n_missing_fit_: Number of nulls seen in the training data.
    """

    _SUPPORTED_STRATEGIES: tuple[str, ...] = ("median",)

    def __init__(
        self,
        column: str = TEMPERATURE_COL,
        strategy: str = "median",
        add_missing_flag: bool = True,
        flag_column: str = TEMP_MISSING_COL,
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

    def _required_columns(self) -> list[str]:
        """The temperature column is required."""
        return [self.column]

    def _learn(self, X: pd.DataFrame) -> None:
        """Learn the fill value from the training data only.

        Raises:
            ValueError: If ``strategy`` is not supported or the column is entirely null.
        """
        if self.strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported impute strategy {self.strategy!r}; use one of {self._SUPPORTED_STRATEGIES}.")
        median = X[self.column].median()
        if pd.isna(median):
            raise ValueError(f"Cannot impute '{self.column}': it is entirely null in the training data.")
        self.fill_value_ = float(median)
        self.n_missing_fit_ = int(X[self.column].isna().sum())
        logger.debug(
            "TemperatureImputer fitted: median %s = %.3f from %d rows (%d missing)",
            self.column, self.fill_value_, len(X), self.n_missing_fit_,
        )

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the missing flag (optional) and fill nulls with the learned median."""
        missing = X[self.column].isna()
        if self.add_missing_flag:
            X[self.flag_column] = missing.astype(int)
        X[self.column] = X[self.column].fillna(self.fill_value_)
        if len(X):
            logger.debug("TemperatureImputer imputed %d of %d values", int(missing.sum()), len(X))
        return X


class ThresholdFeatureAdder(DataFrameTransformer):
    """Add binary features for readings above the EDA thresholds.

    Creates ``temp_over_<temp_threshold>`` and ``vib_over_<vib_threshold>``
    (e.g. ``temp_over_90``, ``vib_over_80``) using a strict ``>`` comparison.
    Stateless. Must run after temperature imputation and ``abs(vibration)``.
    """

    def __init__(
        self,
        temp_threshold: float,
        vib_threshold: float,
        temp_column: str = TEMPERATURE_COL,
        vib_column: str = VIBRATION_ABS_COL,
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

    def _required_columns(self) -> list[str]:
        """Both the temperature and absolute vibration columns are required."""
        return [self.temp_column, self.vib_column]

    def feature_names(self) -> list[str]:
        """Return the names of the columns this transformer adds.

        Returns:
            ``[f"temp_over_{temp_threshold}", f"vib_over_{vib_threshold}"]`` with
            integer-valued thresholds written without decimals.
        """
        return [
            f"temp_over_{_format_threshold(self.temp_threshold)}",
            f"vib_over_{_format_threshold(self.vib_threshold)}",
        ]

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the two 0/1 threshold columns."""
        temp_name, vib_name = self.feature_names()
        X[temp_name] = (X[self.temp_column] > self.temp_threshold).astype(int)
        X[vib_name] = (X[self.vib_column] > self.vib_threshold).astype(int)
        return X


class FailureTypeEncoder(DataFrameTransformer):
    """One-hot encode the ``failure_type`` column (used only when ``feature_set`` is ``all``).

    Categories are learned in ``fit`` from the training data. A category that
    appears only at transform time gets all-zero indicator columns. The raw
    string column is dropped.

    Attributes (after ``fit``):
        categories_: Sorted categories seen in the training data.
    """

    def __init__(self, column: str = FAILURE_TYPE_COL) -> None:
        """Store the column to encode (sklearn convention: no logic in ``__init__``).

        Args:
            column: Categorical column to one-hot encode.
        """
        self.column = column

    def _required_columns(self) -> list[str]:
        """The categorical column is required."""
        return [self.column]

    def _learn(self, X: pd.DataFrame) -> None:
        """Learn the categories present in the training data."""
        values = X[self.column].astype(object).dropna()
        self.categories_ = sorted(values.unique(), key=str)
        logger.debug("FailureTypeEncoder learned %d categories: %s", len(self.categories_), self.categories_)

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace the categorical column with one 0/1 column per learned category."""
        values = X[self.column].astype(object)
        for category in self.categories_:
            X[f"{self.column}_{_safe_name(category)}"] = (values == category).astype(int)
        return X.drop(columns=self.column)


class FeatureSelector(DataFrameTransformer):
    """Keep only the configured model feature columns, in a fixed order. Stateless."""

    def __init__(self, columns: list[str]) -> None:
        """Store the columns to keep (sklearn convention: no logic in ``__init__``).

        Args:
            columns: Feature columns to output, in order.
        """
        self.columns = columns

    def _required_columns(self) -> list[str]:
        """Every selected column is required."""
        return list(self.columns)

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the selected columns."""
        return X[list(self.columns)]


def _format_threshold(value: float) -> str:
    """Format a threshold for a column name: ``90 -> "90"``, ``92.5 -> "92_5"``."""
    return f"{value:g}".replace(".", "_").replace("-", "minus")


def _safe_name(value: Any) -> str:
    """Turn a category value into a safe column-name suffix (non-word characters -> ``_``)."""
    return re.sub(r"\W+", "_", str(value)).strip("_")


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
