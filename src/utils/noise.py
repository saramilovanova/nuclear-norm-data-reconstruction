"""Noise generation helpers."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(X, sigma, normalize=False, seed=None):
    """Add Gaussian noise to the data matrix X."""
    # noise = sigma * np.random.randn(*X.shape)
    rng = np.random.default_rng(seed=seed)
    noise = sigma * rng.standard_normal(X.shape)
    X_noisy = X + noise
    if normalize:
        X_noisy = np.clip(X_noisy, 0, 1)
    else:
        X_noisy = np.clip(X_noisy, 0, 255)
    return X_noisy


def add_pairflip_noise(X, mask, prob=0.1, seed=None):
    """Flip each observed entry up or down by 1 with probability prob."""
    rng = np.random.default_rng(seed=seed)
    Y = X.copy()

    flip_mask = (rng.random(X.shape) < prob) & mask

    # choose direction: +1 or -1
    direction = rng.choice([-1, 1], size=X.shape)

    Y[flip_mask] = Y[flip_mask] + direction[flip_mask]

    # keep ratings valid
    Y = np.clip(Y, 1, 5)

    return Y, flip_mask


def add_pairflip_extreme(X, mask, prob=0.1, seed=None):
    """Flip each observed entry to its opposite extreme (1<->5, 2<->4) with probability prob."""
    rng = np.random.default_rng(seed=seed)
    Y = X.copy()

    flip_mask = (rng.random(X.shape) < prob) & mask

    mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}

    for val in [1, 2, 3, 4, 5]:
        idx = flip_mask & (X == val)
        Y[idx] = mapping[val]

    # keep ratings valid
    Y = np.clip(Y, 1, 5)

    return Y, flip_mask


def add_symmetric_noise(X, mask, prob=0.1, seed=None):
    """Flip each observed entry to a different random rating with probability prob."""
    rng = np.random.default_rng(seed=seed)
    Y = X.copy()

    flip_mask = (rng.random(X.shape) < prob) & mask

    for val in [1, 2, 3, 4, 5]:
        idx = flip_mask & (X == val)
        choices = [rating for rating in [1, 2, 3, 4, 5] if rating != val]
        Y[idx] = rng.choice(choices, size=np.sum(idx))

    # keep ratings valid
    Y = np.clip(Y, 1, 5)

    return Y, flip_mask
