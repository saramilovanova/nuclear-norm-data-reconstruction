"""Simple I/O helpers for numpy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# from skimage.io import imread
import cv2
import pandas as pd


def load_image(path: str | Path, normalize: bool = False) -> np.ndarray:
    """Load image as grayscale numpy array normalized to [0, 1]."""
    X = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE).astype(np.float64)

    # Normalize to [0, 1]
    if normalize and X.max() > 1.0:
        X /= 255.0
    return X


def load_netflix_matrix(
    path: str | Path, normalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Load Netflix data matrix from CSV, returning filled matrix and mask."""
    df = pd.read_csv(path, index_col=0)
    X = df.values.astype(np.float64)

    if normalize:
        # normalize to [0,1] like images
        X = (X - 1.0) / 4.0

    mask = ~np.isnan(X)
    X_filled = np.nan_to_num(X, nan=0.0)

    return X_filled, mask
