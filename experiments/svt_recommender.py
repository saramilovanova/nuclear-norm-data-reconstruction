"""Run SVT experiments on recommender matrices."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.svt.svt import svt
from src.utils.masking import create_netflix_mask, apply_mask
from src.utils.noise import add_gaussian_noise, add_pairflip_noise, add_pairflip_extreme
from src.utils.io import load_netflix_matrix
from src.utils.metrics import mse, rmse, psnr

from experiments.config import *

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/netflix")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/svt/recommender")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def run_reconstruction_experiment_netflix(X, original_mask, sparsity=0.4):

    observed_fraction = 1 - sparsity
    mask = create_netflix_mask(original_mask, observed_fraction, seed=SEED)
    Omega, b = apply_mask(X, mask)

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    # tau = TAU_FACTOR * max(n1, n2)
    delta = DELTA_FACTOR / observed_fraction

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # evaluate ONLY where data exists but was hidden
    test_mask = original_mask & (~mask)

    # X_rec = X_rec * 4 + 1
    # X_rec = np.clip(X_rec, 1, 5)

    mse_val = np.mean((X[test_mask] - X_rec[test_mask]) ** 2)
    psnr_val = psnr(X[test_mask], X_rec[test_mask], data_range=X.max() - X.min())
    rmse = np.sqrt(np.mean((X[test_mask] - X_rec[test_mask]) ** 2))
    nrmse = rmse / (X.max() - X.min())  # divide by 4
    mae = np.mean(np.abs(X[test_mask] - X_rec[test_mask]))

    print(
        "Reconstruction for {}% observed entries:".format(int(observed_fraction * 100))
    )
    print("MSE:", mse_val)
    print("PSNR:", psnr_val)
    print("RMSE:", rmse)
    print("NRMSE:", nrmse)
    print("MAE:", mae)
    return mse_val, psnr_val, rmse, nrmse, mae


def run_denoising_experiment_netflix(X, original_mask, sigma=0.1):

    # Y = add_gaussian_noise(X, sigma)
    Y, flip_mask = add_pairflip_noise(X, original_mask, prob=sigma)

    # only use observed entries
    # Omega = np.where(original_mask)
    Omega = np.where(original_mask & (~flip_mask))
    b = Y[Omega]

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    delta = DELTA_FACTOR

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # --- evaluation ---
    mse_val = np.mean(
        (X[original_mask] - X_rec[original_mask]) ** 2
    )  # overall MSE on observed entries
    mse_corrupted = np.mean(
        (X[flip_mask] - X_rec[flip_mask]) ** 2
    )  # did we fix the corrupted entries?
    clean_mask = original_mask & (~flip_mask)
    mse_clean = np.mean(
        (X[clean_mask] - X_rec[clean_mask]) ** 2
    )  # did we damage correct entries?

    psnr_val = psnr(
        X[original_mask], X_rec[original_mask], data_range=X.max() - X.min()
    )
    rmse = np.sqrt(mse_val)
    nrmse = rmse / (X.max() - X.min())  # divide by 4
    rmse_clean = np.sqrt(mse_clean)
    rmse_corrupted = np.sqrt(mse_corrupted)

    mae = np.mean(np.abs(X[original_mask] - X_rec[original_mask]))

    print("Denoising with sigma={}:".format(sigma))
    print("MSE:", mse_val)
    print("MSE on corrupted entries:", mse_corrupted)
    print("MSE on clean entries:", mse_clean)
    print("PSNR:", psnr_val)
    print("RMSE:", rmse)
    print("NRMSE:", nrmse)
    print("RMSE on corrupted entries:", rmse_corrupted)
    print("RMSE on clean entries:", rmse_clean)
    print("MAE:", mae)
    return (
        mse_val,
        mse_corrupted,
        mse_clean,
        psnr_val,
        rmse,
        rmse_corrupted,
        rmse_clean,
        nrmse,
        mae,
    )


def main():

    X, original_mask = load_netflix_matrix(
        DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
    )

    reconstruction_results = []
    denoising_results = []

    # --- Reconstruction ---
    for s in SPARSITY_LEVELS:
        m, p, r, n, a = run_reconstruction_experiment_netflix(X, original_mask, s)

        reconstruction_results.append(
            {"sparsity": s, "MSE": m, "PSNR": p, "RMSE": r, "NRMSE": n, "MAE": a}
        )

    # --- Denoising ---
    for sigma in NOISE_LEVELS:
        m, mc, mclean, p, r, rc, rclean, n, a = run_denoising_experiment_netflix(
            X, original_mask, sigma
        )

        denoising_results.append(
            {
                "sigma": sigma,
                "MSE": m,
                "MSE_Corrupted": mc,
                "MSE_Clean": mclean,
                "PSNR": p,
                "RMSE": r,
                "RMSE_Corrupted": rc,
                "RMSE_Clean": rclean,
                "NRMSE": n,
                "MAE": a,
            }
        )

    pd.DataFrame(reconstruction_results).to_csv(
        RESULTS_PATH / "netflix_reconstruction_notnorm.csv", index=False
    )
    pd.DataFrame(denoising_results).to_csv(
        RESULTS_PATH / "netflix_denoising_notnorm.csv", index=False
    )


if __name__ == "__main__":
    main()
