"""Evaluation metrics used across reconstruction experiments."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in decibels.
    PSNR = 10 * log10(MAX^2 / MSE)

    data_range:
        1.0 if images normalized to [0,1]
        255 if using uint8 images
    """
    err = mse(y_true, y_pred)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / err))
