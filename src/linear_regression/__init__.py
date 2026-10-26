"""Pacote de regressão linear construído do zero com NumPy."""

from linear_regression.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)
from linear_regression.model import LinearRegression

__all__ = [
    "LinearRegression",
    "train_test_split",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]
