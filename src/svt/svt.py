import numpy as np
from scipy.sparse.linalg import svds


def svt(matrix_shape, Omega, b, tau, delta, max_iter=1000, tol=1e-4):
    """
    Singular Value Thresholding (SVT) algorithm for matrix completion and denoising.

    - Finds the minimum of   tau ||X||_* + .5 || X ||_F^2
    - subject to P_Omega(X) = P_Omega(M)

    Parameters:
        matrix_shape: tuple
            Shape of the matrix (n1, n2)
        Omega: tuple of arrays
            Indices of observed entries
        b: array
            Observed values at Omega
        tau: float
            threshold parameter
        delta: float
            step size
        max_iter: int
            maximum number of iterations
        tol: float
            convergence tolerance

    Returns:
        X: array
            Reconstructed matrix
        history: dict
            Dictionary containing convergence history (residuals, ranks)
    """

    n1, n2 = matrix_shape

    # Projection matrix of observations
    Y = np.zeros((n1, n2))
    Y[Omega] = b

    # spectral norm initialization
    norm2 = np.linalg.norm(Y, 2)
    k0 = int(np.ceil(tau / (delta * norm2)))
    Y *= k0 * delta

    norm_b = np.linalg.norm(b) + 1e-8  # to avoid division by zero in relative error

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
            X = np.zeros((n1, n2))
        else:
            X = (U[:, :rank] * S_thresh[:rank]) @ Vt[:rank, :]

        # --- Residual ---
        X_Omega = X[Omega]
        residual = b - X_Omega
        rel_error = np.linalg.norm(residual) / norm_b

        history["residual"].append(rel_error)
        history["rank"].append(rank)
        print(f"iter {k}, rank {rank}, residual {rel_error:.5f}")

        # Check convergence
        if rel_error < tol:
            break

        # --- Dual update ---
        Y[Omega] += delta * residual

        # --- increase rank if needed ---
        # if S[-1] > tau and r < min(n1, n2):
        #     r = min(r * 2, min(n1, n2))

    return X, history
