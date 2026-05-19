from __future__ import annotations

from typing import Optional

import numpy as np

from .omp import omp_batch
from .dictionary_update import update_dictionary


def _normalize_columns(D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Normalize dictionary columns to unit norm. If a column has near-zero norm, it is left unchanged to avoid numerical issues.
    norms = np.linalg.norm(D, axis=0)
    safe_norms = np.where(norms > eps, norms, 1.0)
    return D / safe_norms


def initialize_dictionary(
    Y: np.ndarray,
    n_atoms: int,
    random_state: Optional[int] = None,
    method: str = "data",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Initialize dictionary columns either from random data columns or Gaussian noise.
    """
    Y = np.asarray(Y, dtype=float)
    n, N = Y.shape
    rng = np.random.default_rng(random_state)

    if method not in {"data", "gaussian"}:
        raise ValueError("method must be 'data' or 'gaussian'.")

    if method == "data":
        replace = n_atoms > N
        idx = rng.choice(N, size=n_atoms, replace=replace)
        D = Y[:, idx].copy()
        zero_cols = np.linalg.norm(D, axis=0) < eps
        if np.any(zero_cols):
            D[:, zero_cols] = rng.standard_normal((n, int(np.sum(zero_cols))))
    else:
        D = rng.standard_normal((n, n_atoms))

    return _normalize_columns(D, eps=eps)


def ksvd(
    Y: np.ndarray,
    n_atoms: int,
    sparsity: int,
    n_iter: int = 20,
    tol: float = 1e-6,
    initial_dictionary: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    fixed_atoms: int = 0,
    normalize_masked_dictionary: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    K-SVD dictionary learning.

    Parameters
    ----------
    Y : (n, N) array
        Data matrix with samples in columns.
    n_atoms : int
        Dictionary size K.
    sparsity : int
        OMP sparsity level T0.
    n_iter : int
        Maximum number of K-SVD iterations.
    tol : float
        Stop if relative objective improvement is below this threshold.
    initial_dictionary : (n, K) array, optional
        User-supplied initial dictionary.
    masks : (n, N) array, optional
        Observation masks for sparse coding only.
    fixed_atoms : int
        Number of leading atoms kept fixed during the update step.
    normalize_masked_dictionary : bool
        Passed to OMP for masked sparse coding.
    random_state : int, optional
        RNG seed.
    verbose : bool
        Print progress.

    Returns
    -------
    D : (n, K) array
        Learned dictionary.
    X : (K, N) array
        Sparse codes.
    history : list[float]
        Objective values per iteration.
    """
    Y = np.asarray(Y, dtype=float)
    n, N = Y.shape

    if initial_dictionary is None:
        D = initialize_dictionary(Y, n_atoms, random_state=random_state, method="data")
    else:
        D = np.asarray(initial_dictionary, dtype=float).copy()
        if D.shape != (n, n_atoms):
            raise ValueError("initial_dictionary must have shape (n, n_atoms).")
        D = _normalize_columns(D)

    history: list[float] = []
    prev_obj: Optional[float] = None

    for it in range(n_iter):
        # 1. Sparse coding - compute X given current D using OMP
        X = omp_batch(
            Y,
            D,
            sparsity=sparsity,
            masks=masks,
            normalize_masked_dictionary=normalize_masked_dictionary,
        )

        # 2. Dictionary update - update each atom in D using SVD
        D, X = update_dictionary(
            Y,
            D,
            X,
            masks=masks,
            fixed_atoms=fixed_atoms,
            random_state=random_state,
        )

        # Compute objective value (squared Frobenius norm of residual)
        residual = Y - D @ X
        if masks is not None:
            residual = residual * masks

        obj = float(np.linalg.norm(residual, ord="fro") ** 2)
        history.append(obj)

        if verbose:
            if prev_obj is None:
                print(f"iter {it + 1:02d}: objective = {obj:.6e}")
            else:
                rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
                print(
                    f"iter {it + 1:02d}: objective = {obj:.6e}, rel_change = {rel:.3e}"
                )

        # Check for convergence
        if prev_obj is not None:
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
            if rel_change < tol:
                break

        prev_obj = obj

    return D, X, history
