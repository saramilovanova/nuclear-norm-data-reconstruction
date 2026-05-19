from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def extract_patches(
    image: np.ndarray,
    patch_size: int | Sequence[int] = 8,
    stride: int = 1,
    flatten: bool = True,
    subtract_mean: bool = False,
    return_positions: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """
    Extract sliding patches from a 2D grayscale image.

    Returns patches as columns: shape (patch_dim, n_patches) when flatten=True.
    If subtract_mean=True, also returns the per-patch means so they can be
    restored during reconstruction.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("extract_patches currently expects a 2D grayscale image.")

    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    H, W = image.shape
    if H < ph or W < pw:
        raise ValueError("patch_size larger than the image.")

    patches = []
    patch_means = []
    positions = []

    for i in range(0, H - ph + 1, stride):
        for j in range(0, W - pw + 1, stride):
            patch = image[i : i + ph, j : j + pw].copy()
            mean = float(patch.mean())
            if subtract_mean:
                patch = patch - mean
                patch_means.append(mean)
            if flatten:
                patches.append(patch.reshape(-1))
            else:
                patches.append(patch)
            positions.append((i, j))

    patches_arr = np.stack(patches, axis=1 if flatten else 0)
    patch_means_arr = np.asarray(patch_means, dtype=float) if subtract_mean else None

    if subtract_mean and return_positions:
        return patches_arr, patch_means_arr, np.asarray(positions, dtype=int)

    if subtract_mean:
        return patches_arr, patch_means_arr

    if return_positions:
        return patches_arr, np.asarray(positions, dtype=int)

    return patches_arr


def reconstruct_from_patches(
    patches: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int | Sequence[int] = 8,
    stride: int = 1,
    positions: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    patch_means: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct an image from patches by overlap-averaging.

    Parameters
    ----------
    patches : array
        Shape (patch_dim, n_patches) if flattened, or (n_patches, ph, pw).
    image_shape : tuple
        Target image shape (H, W).
    positions : array, optional
        Patch top-left coordinates. If omitted, positions are generated using
        the same stride and patch_size convention as extract_patches.
    weights : array, optional
        Per-patch scalar weights.
    patch_means : array, optional
        Per-patch means previously returned by extract_patches when
        subtract_mean=True.

    Returns
    -------
    image : (H, W) array
    """
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    H, W = image_shape
    image = np.zeros((H, W), dtype=float)
    counts = np.zeros((H, W), dtype=float)

    patches = np.asarray(patches, dtype=float)

    flattened = patches.ndim == 2
    if flattened:
        n_patches = patches.shape[1]
    elif patches.ndim == 3:
        n_patches = patches.shape[0]
    else:
        raise ValueError(
            "patches must be either (patch_dim, n_patches) or (n_patches, ph, pw)."
        )

    if positions is None:
        pos = []
        for i in range(0, H - ph + 1, stride):
            for j in range(0, W - pw + 1, stride):
                pos.append((i, j))
        positions = np.asarray(pos, dtype=int)
    else:
        positions = np.asarray(positions, dtype=int)

    if positions.shape[0] != n_patches:
        raise ValueError("positions and patches disagree on number of patches.")

    if weights is None:
        weights = np.ones(n_patches, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != n_patches:
            raise ValueError("weights must have one value per patch.")

    if patch_means is not None:
        patch_means = np.asarray(patch_means, dtype=float).reshape(-1)
        if patch_means.shape[0] != n_patches:
            raise ValueError("patch_means must have one value per patch.")

    for idx in range(n_patches):
        i, j = positions[idx]
        w = weights[idx]

        if flattened:
            patch = patches[:, idx].reshape(ph, pw)
        else:
            patch = patches[idx]

        if patch_means is not None:
            patch = patch + patch_means[idx]

        image[i : i + ph, j : j + pw] += w * patch
        counts[i : i + ph, j : j + pw] += w

    counts[counts == 0] = 1.0
    return image / counts
