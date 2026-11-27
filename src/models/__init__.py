"""Model registry: maps config names to model classes.

Add a new model by writing a ``BaseModel`` subclass and listing it in
``_MODEL_CLASSES``. Its ``name`` becomes the key used in ``models.enabled`` and
on the command line.
"""

from __future__ import annotations

from typing import Any

from .base import BaseModel
from .hist_gradient_boosting import HistGradientBoostingModel
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel

_MODEL_CLASSES: tuple[type[BaseModel], ...] = (
    LogisticRegressionModel,
    RandomForestModel,
    HistGradientBoostingModel,
)

#: Model name (as used in config and CLI) -> model class.
MODEL_REGISTRY: dict[str, type[BaseModel]] = {cls.name: cls for cls in _MODEL_CLASSES}


def get_model(name: str, cfg: dict[str, Any]) -> BaseModel:
    """Create the model registered under ``name``.

    Args:
        name: Registry key, e.g. ``"random_forest"``.
        cfg: Full pipeline configuration.

    Returns:
        A configured, unfitted model instance.

    Raises:
        ValueError: If ``name`` is not in ``MODEL_REGISTRY``.
    """
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}.") from None
    return model_cls(cfg)


__all__ = [
    "BaseModel",
    "HistGradientBoostingModel",
    "LogisticRegressionModel",
    "MODEL_REGISTRY",
    "RandomForestModel",
    "get_model",
]
