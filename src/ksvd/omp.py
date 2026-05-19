from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.linear_model import orthogonal_mp


def omp_single(
    y: np.ndarray,
    D: np.ndarray,
    sparsity: Optional[int] = None,
    error_goal: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    normalize_masked_dictionary: bool = True,
    max_atoms: Optional[int] = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Orthogonal Matching Pursuit for one signal.

    Parameters
    ----------
    y : (n,) array
        Signal.
    D : (n, K) array
        Dictionary with columns as atoms.
    sparsity : int, optional
        Maximum number of nonzeros to use.
    error_goal : float, optional
        Stop when ||residual||_2 <= error_goal.
    mask : (n,) bool/0-1 array, optional
        Observed entries. If provided, only those entries are used in the
        projections and least-squares fit.
    normalize_masked_dictionary : bool
        If True, the masked dictionary columns are normalized before pursuit.
        This mirrors the paper's missing-pixel experiment, where projections
        used only the observed entries.
    max_atoms : int, optional
        Hard cap on the number of selected atoms.
    tol : float
        Numerical tolerance.

    Returns
    -------
    x : (K,) array
        Sparse coefficient vector.
    """
    y = np.asarray(y).reshape(-1)
    D = np.asarray(D)
    n, K = D.shape

    # Check whether y and D are compatible
    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]}, but D has {n} rows.")

    # Determine max_atoms based on provided parameters
    if sparsity is None and error_goal is None and max_atoms is None:
        max_atoms = K
    elif max_atoms is None:
        max_atoms = sparsity if sparsity is not None else K

    # Initialize effective signal and dictionary based on mask
    if mask is not None:
        obs = np.asarray(mask, dtype=bool).reshape(-1)
        if obs.shape[0] != n:
            raise ValueError("mask must have the same length as y.")
        y_eff = y[obs]
        D_eff = D[obs, :]
        if normalize_masked_dictionary:
            col_norms = np.linalg.norm(D_eff, axis=0)
            valid = col_norms > tol
            D_work = np.zeros_like(D_eff)
            D_work[:, valid] = D_eff[:, valid] / col_norms[valid]
        else:
            col_norms = np.ones(K)
            D_work = D_eff.copy()
            valid = np.linalg.norm(D_work, axis=0) > tol
    else:  # No mask, use original y and D
        y_eff = y
        D_work = D
        col_norms = np.ones(K)
        # If not normalizing, we still want to check for near-zero norm columns to avoid numerical issues
        valid = np.linalg.norm(D_work, axis=0) > tol

    # Check for valid atoms
    if not np.any(valid):
        raise ValueError("No valid dictionary atoms (all have near-zero norm).")

    # Initialize
    residual = y_eff.copy()
    support: list[int] = []

    # Main OMP loop
    while True:
        if len(support) >= max_atoms:
            break

        # Step 1: Find atom with maximum correlation to residual

        # candidate_idx = np.flatnonzero(valid)
        # correlations = D_work[:, candidate_idx].T @ residual
        # best_local = int(np.argmax(np.abs(correlations)))
        # best_atom = int(candidate_idx[best_local])

        available = [i for i in np.flatnonzero(valid) if i not in support]
        if not available:
            break
        correlations = D_work[:, available].T @ residual
        best_atom = available[int(np.argmax(np.abs(correlations)))]

        if best_atom in support:
            break

        # Step 2: Add atom to support set
        support.append(best_atom)
        Ds = D_work[:, support]

        # Step 3: Solve least squares for all atoms in support
        coef_norm, *_ = np.linalg.lstsq(Ds, y_eff, rcond=None)

        # Step 4: Update residual
        residual = y_eff - Ds @ coef_norm

        # Step 5: Check stopping criteria
        if error_goal is not None and np.linalg.norm(residual) <= error_goal:
            break
        if sparsity is not None and len(support) >= sparsity:
            break

    # Assemble final sparse coefficient vector
    x = np.zeros(K, dtype=float)
    if support:
        coef_original = coef_norm / col_norms[support]
        x[np.array(support, dtype=int)] = coef_original

    return x


def omp_batch(
    Y: np.ndarray,
    D: np.ndarray,
    sparsity: Optional[int] = None,
    error_goal: Optional[float] = None,
    masks: Optional[np.ndarray] = None,
    normalize_masked_dictionary: bool = True,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    OMP for a batch of signals.

    Y is expected to be shaped (n, N), with signals in columns.
    """
    Y = np.asarray(Y)
    D = np.asarray(D)
    n, N = Y.shape
    _, K = D.shape

    X = np.zeros((K, N), dtype=float)

    for i in range(N):
        mask_i = None if masks is None else masks[:, i]
        X[:, i] = omp_single(
            Y[:, i],
            D,
            sparsity=sparsity,
            error_goal=error_goal,
            mask=mask_i,
            normalize_masked_dictionary=normalize_masked_dictionary,
            tol=tol,
        )

    return X


# def omp_batch(Y, D, sparsity):
#     """
#     Sparse coding using sklearn OMP.

#     Parameters
#     ----------
#     Y : (n, N)
#         Signals in columns.
#     D : (n, K)
#         Dictionary.
#     sparsity : int
#         Number of nonzero coefficients.

#     Returns
#     -------
#     X : (K, N)
#         Sparse coefficient matrix.
#     """

#     X = orthogonal_mp(
#         D,
#         Y,
#         n_nonzero_coefs=sparsity
#     )

#     return X
