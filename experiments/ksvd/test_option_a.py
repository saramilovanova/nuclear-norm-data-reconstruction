"""
Test: K-SVD Recommender — Option A (Error-Goal Reconstruction)
===============================================================

This is the simpler, more stable approach. Use this first.
"""

import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ksvd_recommender_option_a import (
    create_train_test_split,
    ksvd_recommender_error_goal,
    reconstruct_with_error_goal,
    evaluate_recommender,
)

from src.utils.io import load_netflix_matrix

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/netflix")
N_ATOMS = 256
SPARSITY_TRAIN = 10
N_ITER = 20
EPSILON = 0.5  # error tolerance in rating points
TEST_FRACTION = 0.1  # hold out 10% of observed ratings
SEED = 42

# ─────────────────────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────────────────────

print("Loading Netflix matrix...")
R_full, original_mask = load_netflix_matrix(
    DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
)

n_users, n_items = R_full.shape
print(f"Shape: {n_users} users × {n_items} items")
print(
    f"Observed: {original_mask.sum()} / {original_mask.size} "
    f"({100*original_mask.mean():.1f}%)"
)
print(
    f"Rating range: [{R_full[original_mask].min():.1f}, "
    f"{R_full[original_mask].max():.1f}]"
)
print()

# ─────────────────────────────────────────────────────────────
# Train/test split (within observed ratings)
# ─────────────────────────────────────────────────────────────

print(f"Creating train/test split ({TEST_FRACTION*100:.0f}% holdout)...")
R_train, train_mask, test_mask = create_train_test_split(
    R_full,
    test_fraction=TEST_FRACTION,
    random_state=SEED,
)

print(
    f"Training entries: {train_mask.sum()} "
    f"({100*train_mask.mean():.1f}% of all entries)"
)
print(
    f"Test entries: {test_mask.sum()} " f"({100*test_mask.mean():.1f}% of all entries)"
)
print()

# ─────────────────────────────────────────────────────────────
# Train K-SVD
# ─────────────────────────────────────────────────────────────

print("Training K-SVD...")
D, X_train, user_means, history = ksvd_recommender_error_goal(
    R_train,
    n_atoms=N_ATOMS,
    sparsity=SPARSITY_TRAIN,
    n_iter=N_ITER,
    epsilon=EPSILON,
    random_state=SEED,
    verbose=True,
)
print()

# ─────────────────────────────────────────────────────────────
# Convergence plot
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(history) + 1), history, marker="o", markersize=4)
plt.xlabel("Iteration")
plt.ylabel("Masked Reconstruction Error")
plt.title("K-SVD Training Convergence (Option A)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Dictionary shape:", D.shape)
print("Objective values should be monotonically decreasing (or increasing slightly")
print("only if the algorithm is unstable).")
print()

# ─────────────────────────────────────────────────────────────
# Reconstruct on test data with error-goal OMP
# ─────────────────────────────────────────────────────────────

print("Reconstructing test entries with error-goal OMP...")
R_reconstructed = reconstruct_with_error_goal(
    D,
    R_train,
    user_means,
    epsilon=EPSILON,
    max_atoms=SPARSITY_TRAIN,
)
print()

# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

metrics = evaluate_recommender(R_full, R_reconstructed, test_mask)

print("=" * 50)
print("RESULTS (Option A - Error-Goal Reconstruction)")
print("=" * 50)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"MAE:  {metrics['mae']:.6f}")
print()
print("For comparison:")
print("  SVT (0.4 missing): RMSE = 0.761, MAE = 0.595")
print("  Fill-in k-SVD (0.1 holdout): RMSE = 0.939, MAE = 0.717")
print("=" * 50)

# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────


def show_matrix_results(
    R_true,
    R_train_input,
    R_reconstructed,
    test_mask,
    n_users_show=100,
    n_items_show=256,
    title="",
):
    """Display ratings matrix slices."""
    sl_u = slice(0, n_users_show)
    sl_i = slice(0, n_items_show)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    ims = [
        (R_true[sl_u, sl_i], "Ground truth"),
        (
            R_train_input[sl_u, sl_i],
            f"Training input\n({100*(1-TEST_FRACTION):.0f}% observed)",
        ),
        (R_reconstructed[sl_u, sl_i], "Reconstructed"),
        (
            np.abs((R_true - R_reconstructed) * test_mask)[sl_u, sl_i],
            "Error (test only)",
        ),
    ]

    for ax, (mat, ttl) in zip(axes, ims):
        vmin, vmax = (1, 5) if "error" not in ttl.lower() else (0, 2)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Items")
        ax.set_ylabel("Users")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


show_matrix_results(
    R_full,
    R_train,
    R_reconstructed,
    test_mask,
    title=f"K-SVD Netflix — Option A (K={N_ATOMS}, T={SPARSITY_TRAIN}, ε={EPSILON})",
)

# Error distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

test_errors = np.abs((R_full - R_reconstructed)[test_mask])
axes[0].hist(test_errors, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(
    metrics["rmse"],
    color="crimson",
    linestyle="--",
    label=f"RMSE = {metrics['rmse']:.3f}",
)
axes[0].axvline(
    metrics["mae"], color="orange", linestyle="--", label=f"MAE = {metrics['mae']:.3f}"
)
axes[0].set_xlabel("Absolute error (rating points)")
axes[0].set_ylabel("Count")
axes[0].set_title("Error distribution on test entries")
axes[0].legend()

axes[1].plot(range(1, len(history) + 1), history, marker="o", color="steelblue")
axes[1].set_xlabel("K-SVD iteration")
axes[1].set_ylabel("Masked Reconstruction Error")
axes[1].set_title("Training convergence")

plt.tight_layout()
plt.show()
