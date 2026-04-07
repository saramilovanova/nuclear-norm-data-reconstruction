"""Shrinkage operators for singular values."""

from __future__ import annotations

import numpy as np


def soft_threshold_singular_values(
    singular_values: np.ndarray, tau: float
) -> np.ndarray:
    """Soft-threshold singular values: max(s - tau, 0)."""
    return np.maximum(singular_values - tau, 0.0)


def hard_threshold_singular_values(
    singular_values: np.ndarray, tau: float
) -> np.ndarray:
    """Hard-threshold singular values: s if s >= tau else 0."""
    return singular_values * (singular_values >= tau)
