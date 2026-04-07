"""Noise generation helpers."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(X, sigma):
    """Add Gaussian noise to a matrix."""
    noise = sigma * np.random.randn(*X.shape)
    return np.clip(X + noise, 0, 1)
