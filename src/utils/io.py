"""Simple I/O helpers for numpy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.io import imread
import pandas as pd


def load_image(path: str | Path) -> np.ndarray:
    """Load image as grayscale numpy array normalized to [0, 1]."""
    X = imread(Path(path), as_gray=True)
    X = X.astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0  # Normalize to [0, 1]
    return X


def load_netflix_matrix(path):
    df = pd.read_csv(path, index_col=0)
    X = df.values.astype(np.float32)

    # normalize to [0,1] like images
    X = (X - 1.0) / 4.0

    mask = ~np.isnan(X)
    X_filled = np.nan_to_num(X, nan=0.0)

    return X_filled, mask
