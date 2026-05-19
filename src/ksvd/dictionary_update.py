from __future__ import annotations

from typing import Optional

import numpy as np


def _normalize_column(d: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    # Normalize a single column vector to unit norm.
    n = float(np.linalg.norm(d))
    if n < eps:
        return d.copy(), 1.0
    return d / n, n


def update_dictionary(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    fixed_atoms: int = 0,
    reinitialize_unused: bool = True,
    masks: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-SVD dictionary update.

    For each atom k, form the residual restricted to the samples that use atom k,
    then compute the best rank-1 approximation via SVD.

    Parameters
    ----------
    Y : (n, N) array
        Data matrix, signals in columns.
    D : (n, K) array
        Current dictionary.
    X : (K, N) array
        Current sparse codes.
    fixed_atoms : int
        Number of leading atoms to keep unchanged, e.g. 1 for a fixed DC atom.
    reinitialize_unused : bool
        If True, reinitialize atoms that are unused.
    masks : (n, N) array, optional
        Observation masks. Used to restrict residuals during atom updates.
    random_state : int, optional
        Seed for reinitialization fallback.
    tol : float
        Numerical tolerance.

    Returns
    -------
    D_new, X_new
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float).copy()
    X = np.asarray(X, dtype=float).copy()

    n, N = Y.shape
    n2, K = D.shape
    if n2 != n:
        raise ValueError("Y and D must have the same number of rows.")
    if X.shape != (K, N):
        raise ValueError("X must have shape (K, N).")

    rng = np.random.default_rng(random_state)

    # Loop over dictionary atoms (skip fixed leading atoms)
    for k in range(fixed_atoms, K):
        # Support indices: samples that use atom k
        omega = np.flatnonzero(np.abs(X[k, :]) > tol)

        # Handle unused atoms: reinitialize or steal from largest residual
        if omega.size == 0:
            if not reinitialize_unused:
                continue

            # Residual across all samples
            residual = Y - D @ X
            col_norms = np.linalg.norm(residual, axis=0)
            j = int(np.argmax(col_norms))

            # If no significant residual, reinitialize randomly
            if col_norms[j] <= tol:
                v = rng.standard_normal(n)
                v /= np.linalg.norm(v)
                D[:, k] = v
                X[:, j] = 0.0
                X[k, j] = 0.0
                continue

            # Steal the direction of the largest residual column
            new_atom = residual[:, j] / col_norms[j]
            new_atom, _ = _normalize_column(new_atom, eps=tol)

            D[:, k] = new_atom
            X[:, j] = 0.0
            X[k, j] = col_norms[j]
            continue

        # Compute representation error restricted to samples in omega:
        # E_k = Y_omega - D X_omega + d_k x_k,omega
        # we add back the contribution of atom k to remove its effect
        E = Y[:, omega] - D @ X[:, omega] + np.outer(D[:, k], X[k, omega])

        # Apply observation masks before SVD
        if masks is not None:
            E = E * masks[:, omega]

        # Best rank-1 approximation via SVD
        U, S, Vt = np.linalg.svd(E, full_matrices=False)

        # Update atom and corresponding coefficients
        D[:, k] = U[:, 0]  # d_k = u_1
        X[k, omega] = S[0] * Vt[0, :]  # x_k,omega = sigma_1 * v_1

        # D[:, k], scale = _normalize_column(D[:, k], eps=tol)
        # if abs(scale - 1.0) > tol:
        #     X[k, :] *= scale

    return D, X
