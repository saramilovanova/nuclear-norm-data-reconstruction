import numpy as np


def estimate_sigma(S, beta):
    """
    Estimate noise level using median heuristic (MP approximation).
    """
    mp_median = (
        1 + np.sqrt(beta)
    ) ** 2 * 0.56  # median approximation of MP distribution
    return np.median(S) / np.sqrt(mp_median)


def optimal_shrinkage(S, beta, sigma=None):
    """
    Gavish-Donoho optimal shrinkage (Frobenius loss).
    """
    if sigma is None:
        sigma = estimate_sigma(S, beta)

    # print(f"Estimated sigma: {sigma:.4f}")
    # print(f"Beta: {beta:.4f}")
    # print(f"Max singular value: {S[0]:.4f}")

    y = S / sigma

    term = (y**2 - beta - 1) ** 2 - 4 * beta
    term = np.maximum(term, 0)

    eta = np.sqrt(term) / y
    eta[y <= (1 + np.sqrt(beta))] = 0

    return sigma * eta


# Option 1 (follows Gavish-Donoho): Apply optimal shrinkage on a full matrix after SVT for reconstruction
def optimal_shrinkage_denoise(X, sigma=None):
    """
    Apply optimal shrinkage to a full matrix.
    """
    n1, n2 = X.shape
    beta = min(n1, n2) / max(n1, n2)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    S_shrunk = optimal_shrinkage(S, beta, sigma)

    rank = np.sum(S_shrunk > 0)
    print(f"Optimal shrinkage rank: {rank}")
    if rank == 0:
        return np.zeros_like(X)

    return (U[:, :rank] * S_shrunk[:rank]) @ Vt[:rank, :]


# Option 2: Optimal SVT - shrinkage inside loop
def optimal_svt(matrix_shape, Omega, b, delta, sigma=None, max_iter=500, tol=1e-5):
    n1, n2 = matrix_shape
    Y = np.zeros((n1, n2))

    norm_b = np.linalg.norm(b) + 1e-8
    history = {"residual": [], "rank": []}

    for k in range(max_iter):

        # --- SVD ---
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        # print(f"Iteration {k+1}: singular values = {S}")

        beta = min(n1, n2) / max(n1, n2)
        # sigma_est = estimate_sigma(S, beta)

        S_shrunk = optimal_shrinkage(S, beta, sigma)

        rank = np.sum(S_shrunk > 0)

        if rank == 0:
            X = np.zeros((n1, n2))
        else:
            X = (U[:, :rank] * S_shrunk[:rank]) @ Vt[:rank, :]

        # --- Residual ---
        residual = b - X[Omega]
        rel_error = np.linalg.norm(residual) / norm_b

        history["residual"].append(rel_error)
        history["rank"].append(rank)

        if rel_error < tol:
            break

        # dual update
        Y[Omega] += delta * residual

    print(f"Final rank: {rank}, final relative error: {rel_error:.6f}")
    return X, history
