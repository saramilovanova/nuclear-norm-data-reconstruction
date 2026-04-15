"""Utilities for generating missing-entry masks."""

from __future__ import annotations

import numpy as np


def create_mask(shape, observed_fraction, seed=None):
    rng = np.random.default_rng(seed)

    # mask = np.random.rand(*shape) < observed_fraction
    mask = rng.random(shape) < observed_fraction
    return mask


def apply_mask(X, mask):
    Omega = np.where(mask)
    b = X[Omega]
    return Omega, b


def create_netflix_mask(original_mask, observed_fraction, seed=None):
    rng = np.random.default_rng(seed)

    # random_mask = np.random.rand(*original_mask.shape) < observed_fraction
    random_mask = rng.random(original_mask.shape) < observed_fraction

    # only keep entries that actually exist
    final_mask = original_mask & random_mask
    return final_mask
