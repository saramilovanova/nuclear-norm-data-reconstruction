import numpy as np
from scipy.sparse.linalg import svds


def svt(matrix_shape, Omega, b, tau, delta, max_iter=200, tol=1e-4):
    """
    Singular Value Thresholding (SVT) algorithm

    - Finds the minimum of   tau ||X||_* + .5 || X ||_F^2
    - subject to P_Omega(X) = P_Omega(M)

    Parameters:
        matrix_shape: (n1, n2)
        Omega: indices of observed entries (tuple of arrays)
        b: observed values
        tau: threshold parameter
        delta: step size
        max_iter: maximum number of iterations
        tol: convergence tolerance
    """

    n1, n2 = matrix_shape

    # Initialize Y
    Y = np.zeros((n1, n2))
    Y[Omega] = b

    norm_b = np.linalg.norm(b) + 1e-12  # avoid division by zero

    # --- k0 initialization ---
    # k0 = int(np.ceil(tau / (delta * norm_b)))
    # Y[Omega] = k0 * delta * b

    history = {"residual": [], "rank": []}

    # r = 10  # initial rank

    for k in range(max_iter):

        # --- SVD ---
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        # --- truncated SVD ---
        # U, S, Vt = svds(Y, k=r, which="LM")

        # --- sort singular values ---
        # idx = np.argsort(S)[::-1]
        # S = S[idx]
        # U = U[:, idx]
        # Vt = Vt[idx, :]

        # --- Soft-thresholding ---
        # Shrink singular values: max(sigma_i - tau, 0)
        S_thresh = np.maximum(S - tau, 0)

        rank = np.sum(S_thresh > 0)

        # X = U @ np.diag(S_thresh) @ Vt
        # Efficient reconstruction (avoid full diag)
        # X = (U[:, :rank] * S_thresh[:rank]) @ Vt[:rank, :]

        if rank == 0:
            X = np.zeros_like(Y)
        else:
            X = (U[:, :rank] * S_thresh[:rank]) @ Vt[:rank, :]

        # --- Residual ---
        X_Omega = X[Omega]
        residual = b - X_Omega
        rel_error = np.linalg.norm(residual) / norm_b

        history["residual"].append(rel_error)
        history["rank"].append(rank)

        # Check convergence
        if rel_error < tol:
            break

        # --- Dual update ---
        Y[Omega] += delta * residual

        # --- increase rank if needed ---
        # if S[-1] > tau and r < min(n1, n2):
        #     r = min(r * 2, min(n1, n2))

    return X, history
