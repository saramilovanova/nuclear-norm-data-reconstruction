"""Utilities for generating missing-entry masks."""

from __future__ import annotations

import numpy as np


def create_mask(shape, observed_fraction, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mask = np.random.rand(*shape) < observed_fraction
    return mask


def apply_mask(X, mask):
    Omega = np.where(mask)
    b = X[Omega]
    return Omega, b
