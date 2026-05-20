"""
K-SVD Denoising
==================================================================
Removed user mean-centering, which was reintroducing noise.

For denoising (unlike reconstruction), we should:
1. Train K-SVD directly on the noisy matrix (no preprocessing)
2. Minimize reconstruction error on all entries
3. Clip reconstructed values to valid range
4. Evaluate against clean ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ksvd.ksvd import ksvd, initialize_dictionary
from src.utils.io import load_netflix_matrix
from src.utils.noise import add_symmetric_noise

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/netflix")
OUTPUT_PATH = Path(
    "nuclear-norm-data-reconstruction/results/ksvd/recommender/denoising"
)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Denoising parameters
NOISE_LEVEL = 0.1
NOISE_TYPE = "symmetric"

# K-SVD parameters
N_ATOMS = 1024  # Increased from 512 (4× overcomplete)
SPARSITY_TRAIN = 20  # Increased from 10
N_ITER = 20  # Slightly more iterations
SEED = 42

print("=" * 70)
print("K-SVD DENOISING")
print("=" * 70)
print()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


print("Loading clean Netflix matrix...")
R_clean, original_mask = load_netflix_matrix(
    DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
)

n_users, n_items = R_clean.shape
print(f"Shape: {n_users} users x {n_items} items")
print(
    f"Observed: {original_mask.sum()} / {original_mask.size} "
    f"({100*original_mask.mean():.1f}%)"
)
print(
    f"Clean rating range: [{R_clean[original_mask].min():.1f}, "
    f"{R_clean[original_mask].max():.1f}]"
)
print()

print(f"Adding {NOISE_TYPE} noise (p={NOISE_LEVEL})...")
if NOISE_TYPE == "symmetric":
    R_noisy, flip_mask = add_symmetric_noise(
        R_clean, mask=original_mask, prob=NOISE_LEVEL, seed=SEED
    )
else:
    R_noisy = R_clean + np.random.RandomState(SEED).normal(
        0, NOISE_LEVEL, R_clean.shape
    )
    R_noisy = np.clip(R_noisy, 1, 5)

print(
    f"Noisy rating range: [{R_noisy[original_mask].min():.1f}, "
    f"{R_noisy[original_mask].max():.1f}]"
)

# Compute baseline error
rmse_corrupted_before = np.sqrt(np.mean((R_clean[flip_mask] - R_noisy[flip_mask]) ** 2))
print("Corrupted entries RMSE (before denoising):", rmse_corrupted_before)
rmse_noisy = rmse(R_clean[original_mask], R_noisy[original_mask])
mae_noisy = mae(R_clean[original_mask], R_noisy[original_mask])
print(f"Noise RMSE: {rmse_noisy:.4f}, MAE: {mae_noisy:.4f}")
print()

print("Preparing data for K-SVD...")
print()

# Transpose to K-SVD convention: (n_items, n_users)
Y = R_noisy.T  # (256, n_users)

print(f"Y shape: {Y.shape}")
print(f"Y value range: [{Y.min():.2f}, {Y.max():.2f}]")
print()

print("Initializing dictionary...")
D = initialize_dictionary(
    Y,
    n_atoms=N_ATOMS,
    random_state=SEED,
    method="data",
)
print(f"Dictionary shape: {D.shape}")
print()


print("Training K-SVD on noisy data...")
print(f"  n_atoms={N_ATOMS}, sparsity={SPARSITY_TRAIN}, n_iter={N_ITER}")
print()

D, X, history = ksvd(
    Y,
    n_atoms=N_ATOMS,
    sparsity=SPARSITY_TRAIN,
    n_iter=N_ITER,
    initial_dictionary=D,
    masks=original_mask.T.astype(bool),  # (n_items, n_users)
    fixed_atoms=0,
    random_state=SEED,
    verbose=True,
)
print()

print("Reconstructing (denoising)...")
Y_denoised = D @ X

# Transpose back to user × item format
R_denoised = Y_denoised.T  # (n_users, n_items)

# Clip to valid rating range [1, 5]
R_denoised = np.clip(R_denoised, 1, 5)

print(f"Denoised rating range: [{R_denoised.min():.1f}, {R_denoised.max():.1f}]")
print()


print("=" * 70)
print("DENOISING RESULTS")
print("=" * 70)

# Evaluate on all observed entries
mask_obs = original_mask
clean_mask = original_mask & (~flip_mask)

rmse_corrupted_after = np.sqrt(
    np.mean((R_clean[flip_mask] - R_denoised[flip_mask]) ** 2)
)
print("Corrupted entries RMSE (after denoising):", rmse_corrupted_after)

rmse_clean_after = np.sqrt(np.mean((R_clean[clean_mask] - R_denoised[clean_mask]) ** 2))
print("Clean entries RMSE (after denoising):", rmse_clean_after)

rmse_denoised = rmse(R_clean[mask_obs], R_denoised[mask_obs])
mae_denoised = mae(R_clean[mask_obs], R_denoised[mask_obs])

# Improvement metrics
rmse_improvement = (rmse_noisy - rmse_denoised) / rmse_noisy * 100
mae_improvement = (mae_noisy - mae_denoised) / mae_noisy * 100

print(f"\nInput (noisy):")
print(f"  RMSE: {rmse_noisy:.4f}")
print(f"  MAE:  {mae_noisy:.4f}")

print(f"\nOutput (denoised):")
print(f"  RMSE: {rmse_denoised:.4f}")
print(f"  MAE:  {mae_denoised:.4f}")

print(f"\nImprovement:")
print(f"  RMSE: {rmse_improvement:+.1f}%")
print(f"  MAE:  {mae_improvement:+.1f}%")


user_error_noisy = np.sqrt(np.mean((R_clean - R_noisy) ** 2, axis=1))
user_error_denoised = np.sqrt(np.mean((R_clean - R_denoised) ** 2, axis=1))

improvement_per_user = (user_error_noisy - user_error_denoised) / user_error_noisy * 100

print("\nPer-user improvement statistics:")
print(f"  Mean improvement: {improvement_per_user.mean():+.1f}%")
print(f"  Median improvement: {np.median(improvement_per_user):+.1f}%")
print(f"  Min improvement: {improvement_per_user.min():+.1f}%")
print(f"  Max improvement: {improvement_per_user.max():+.1f}%")
print(f"  Users with improvement: {(improvement_per_user > 0).sum()} / {n_users}")
print()

# Convergence plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Objective
axes[0].plot(
    range(1, len(history) + 1), history, marker="o", markersize=4, color="steelblue"
)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Reconstruction Error")
axes[0].set_title("K-SVD Training Convergence")
axes[0].grid(True, alpha=0.3)

# Error comparison
methods = ["Noisy Input", "K-SVD Denoised"]
rmse_vals = [rmse_noisy, rmse_denoised]
mae_vals = [mae_noisy, mae_denoised]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = axes[1].bar(
    x_pos - width / 2, rmse_vals, width, label="RMSE", color="steelblue"
)
bars2 = axes[1].bar(x_pos + width / 2, mae_vals, width, label="MAE", color="orange")

axes[1].set_ylabel("Error")
axes[1].set_title("Denoising Quality Comparison")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(methods)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

for bar in bars1:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Improvement histogram
axes[2].hist(improvement_per_user, bins=50, color="steelblue", edgecolor="white")
axes[2].axvline(
    improvement_per_user.mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {improvement_per_user.mean():+.1f}%",
)
axes[2].set_xlabel("Per-User RMSE Improvement (%)")
axes[2].set_ylabel("Number of Users")
axes[2].set_title("Distribution of Improvements")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "convergence_and_errors.png", dpi=150)
plt.show()


def show_denoising_results(
    R_clean, R_noisy, R_denoised, mask, n_users_show=100, n_items_show=256, title=""
):
    """Display clean / noisy / denoised matrix slices."""
    sl_u = slice(0, n_users_show)
    sl_i = slice(0, n_items_show)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    ims = [
        (R_clean[sl_u, sl_i], "Clean Original"),
        (R_noisy[sl_u, sl_i], f"Noisy Input (σ={NOISE_LEVEL})"),
        (R_denoised[sl_u, sl_i], "K-SVD Denoised (Corrected)"),
        (np.abs((R_clean - R_denoised))[sl_u, sl_i], "Abs Error (denoised)"),
    ]

    for ax, (mat, ttl) in zip(axes, ims):
        vmin, vmax = (1, 5) if "error" not in ttl.lower() else (0, 2)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Items (Movies)")
        ax.set_ylabel("Users")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "denoising_matrices.png", dpi=150)
    plt.show()


show_denoising_results(
    R_clean,
    R_noisy,
    R_denoised,
    original_mask,
    title=f"K-SVD Denoising (Corrected: no mean-centering)\n"
    f"K={N_ATOMS}, T={SPARSITY_TRAIN}, σ={NOISE_LEVEL}",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

error_noisy = np.abs(R_clean[mask_obs] - R_noisy[mask_obs])
error_denoised = np.abs(R_clean[mask_obs] - R_denoised[mask_obs])

axes[0].hist(
    error_noisy, bins=50, alpha=0.6, label="Noisy Input", color="red", edgecolor="white"
)
axes[0].hist(
    error_denoised,
    bins=50,
    alpha=0.6,
    label="K-SVD Denoised",
    color="green",
    edgecolor="white",
)
axes[0].axvline(
    rmse_noisy,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Noisy RMSE={rmse_noisy:.3f}",
)
axes[0].axvline(
    rmse_denoised,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Denoised RMSE={rmse_denoised:.3f}",
)
axes[0].set_xlabel("Absolute Error (rating points)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Error Distribution Comparison")
axes[0].legend(fontsize=9)

# Per-user RMSE
axes[1].plot(user_error_noisy, alpha=0.5, label="Noisy", linewidth=1, color="red")
axes[1].plot(
    user_error_denoised, alpha=0.5, label="Denoised", linewidth=1, color="green"
)
axes[1].axhline(rmse_noisy, color="red", linestyle="--", alpha=0.5)
axes[1].axhline(rmse_denoised, color="green", linestyle="--", alpha=0.5)
axes[1].set_xlabel("User ID")
axes[1].set_ylabel("RMSE")
axes[1].set_title("Per-User Denoising Performance")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "error_distribution.png", dpi=150)
plt.show()


print()
print(f"Results saved to {OUTPUT_PATH}")
print("=" * 70)
