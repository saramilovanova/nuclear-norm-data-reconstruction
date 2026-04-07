"""Simple I/O helpers for numpy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.io import imread


def save_array(path: str | Path, array: np.ndarray) -> None:
    """Save array to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_array(path: str | Path) -> np.ndarray:
    """Load array from .npy file."""
    return np.load(Path(path))


def load_image(path: str | Path) -> np.ndarray:
    """Load image as grayscale numpy array normalized to [0, 1]."""
    X = imread(Path(path), as_gray=True)
    X = X.astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0  # Normalize to [0, 1]
    return X
