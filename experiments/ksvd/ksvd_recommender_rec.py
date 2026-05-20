"""
K-SVD Recommender with Error-Goal Reconstruction
============================================================

1. Trains K-SVD on the observed ratings matrix (masked OMP only)
2. Reconstructs held-out entries using error-goal OMP (no dict update on test data)
3. Matches the paper's sparse coding + reconstruction paradigm
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
import numpy as np

from src.ksvd.ksvd import ksvd, initialize_dictionary
from src.ksvd.omp import omp_single, omp_batch


def user_mean_center(
    R: np.ndarray,
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """User-mean center ratings matrix."""
    R = np.asarray(R, dtype=float)
    M = np.asarray(M, dtype=bool)

    user_sum = (R * M).sum(axis=1)
    user_count = M.sum(axis=1)

    user_means = np.divide(
        user_sum,
        user_count,
        out=np.zeros_like(user_sum),
        where=user_count > 0,
    )

    R_centered = (R - user_means[:, None]) * M
    return R_centered, user_means


def restore_user_means(
    R_centered: np.ndarray,
    user_means: np.ndarray,
) -> np.ndarray:
    """Restore user means after reconstruction."""
    return R_centered + user_means[:, None]


def create_train_test_split(
    R: np.ndarray,
    test_fraction: float = 0.1,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hide a subset of observed ratings for evaluation.

    Returns
    -------
    R_train : ratings with test entries set to 0
    train_mask : boolean, True = used for training
    test_mask : boolean, True = held out for evaluation
    """
    rng = np.random.default_rng(random_state)
    R = np.asarray(R, dtype=float)

    observed = np.argwhere(R != 0)
    n_test = int(test_fraction * len(observed))
    test_idx = rng.choice(len(observed), size=n_test, replace=False)
    test_positions = observed[test_idx]

    train_mask = R != 0
    test_mask = np.zeros_like(train_mask, dtype=bool)

    for u, i in test_positions:
        train_mask[u, i] = False
        test_mask[u, i] = True

    R_train = R.copy()
    R_train[test_mask] = 0.0

    return R_train, train_mask, test_mask


def ksvd_recommender_error_goal(
    R: np.ndarray,
    n_atoms: int = 128,
    sparsity: int = 10,
    n_iter: int = 20,
    epsilon: float = 0.5,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-SVD recommender training with error-goal reconstruction.

    Procedure:
    1. User mean-center the ratings matrix
    2. Transpose to (n_items, n_users) for K-SVD convention
    3. Train K-SVD with masked OMP (sparsity) on observed entries
    4. Reconstruct via D @ X with full dictionary
    5. Restore user means

    Parameters
    ----------
    R : (n_users, n_items)
        Ratings matrix, missing = 0
    n_atoms : int
        Dictionary size
    sparsity : int
        Max nonzero coefficients during training
    n_iter : int
        K-SVD iterations
    epsilon : float
        Error tolerance for reconstruction OMP
        (error_goal = epsilon * sqrt(n_observed_per_user))
    random_state : int, optional
    verbose : bool

    Returns
    -------
    D : (n_items, n_atoms)
        Learned dictionary
    X : (n_atoms, n_users)
        Sparse codes
    user_means : (n_users,)
        User biases (subtracted during training)
    """
    R = np.asarray(R, dtype=float)
    M = R != 0  # observation mask (n_users, n_items)

    # Step 1: User mean-center
    R_centered, user_means = user_mean_center(R, M)

    # Step 2: Transpose to K-SVD convention: (n_items, n_users)
    Y = R_centered.T  # (n_items, n_users)
    Mask = M.T.astype(bool)  # (n_items, n_users)

    n_items, n_users = Y.shape

    # Step 3: Train K-SVD with masked OMP
    D, X, history = ksvd(
        Y,
        n_atoms=n_atoms,
        sparsity=sparsity,
        n_iter=n_iter,
        masks=Mask,
        fixed_atoms=0,  # no DC atom for ratings
        normalize_masked_dictionary=True,
        random_state=random_state,
        verbose=verbose,
    )

    return D, X, user_means, history


def reconstruct_with_error_goal(
    D: np.ndarray,
    R_test_input: np.ndarray,
    user_means: np.ndarray,
    epsilon: float = 0.5,
    max_atoms: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct ratings using error-goal OMP.

    For each held-out user:
    - Center their ratings
    - Run masked OMP with error_goal = epsilon * sqrt(n_observed)
    - Reconstruct as D @ x
    - Restore mean

    Parameters
    ----------
    D : (n_items, n_atoms)
    R_test_input : (n_users, n_items)
        Ratings with test entries = 0, observed = original values
    user_means : (n_users,)
    epsilon : float
        Error tolerance per observed rating
    max_atoms : int, optional
        Cap on nonzero coefficients (default = n_atoms)

    Returns
    -------
    R_reconstructed : (n_users, n_items)
        Full reconstructed matrix
    """
    R_test_input = np.asarray(R_test_input, dtype=float)
    n_users, n_items = R_test_input.shape
    n_atoms = D.shape[1]

    if max_atoms is None:
        max_atoms = n_atoms

    # Center test input
    R_centered = R_test_input - user_means[:, None]
    Y = R_centered.T  # (n_items, n_users)

    M = (R_test_input != 0).T.astype(bool)  # (n_items, n_users)

    X = np.zeros((n_atoms, n_users), dtype=float)

    for u in range(n_users):
        y_u = Y[:, u]
        m_u = M[:, u]
        n_obs = m_u.sum()

        if n_obs == 0:
            continue

        # Error goal based on number of observed items for this user
        error_goal_u = epsilon * np.sqrt(float(n_obs))

        X[:, u] = omp_single(
            y_u,
            D,
            error_goal=error_goal_u,
            mask=m_u,
            normalize_masked_dictionary=True,
            max_atoms=max_atoms,
        )

    # Reconstruct
    Y_hat = D @ X
    R_hat = restore_user_means(Y_hat.T, user_means)

    return R_hat


def evaluate_recommender(
    R_true: np.ndarray,
    R_pred: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """Evaluate on held-out test entries only."""
    y_true = R_true[test_mask]
    y_pred = R_pred[test_mask]

    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
