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
from src.utils.noise import (
    add_gaussian_noise,
    add_pairflip_noise,
    add_pairflip_extreme,
    add_symmetric_noise,
)
from src.utils.io import load_netflix_matrix
from src.utils.metrics import mse, rmse, psnr

from experiments.config import *

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/netflix")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/svt/recommender")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def run_reconstruction_experiment_netflix(X, original_mask, sparsity=0.4, seed=None):

    n1, n2 = X.shape
    observed_fraction = 1 - sparsity
    tau = TAU_FACTOR * max(n1, n2)
    delta = DELTA_FACTOR / observed_fraction

    mask = create_netflix_mask(original_mask, observed_fraction, seed=seed)
    Omega, b = apply_mask(X, mask)

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)
    X_rec = np.clip(X_rec, 1, 5)

    # evaluate ONLY where data exists but was hidden
    test_mask = original_mask & (~mask)

    mse_val = np.mean((X[test_mask] - X_rec[test_mask]) ** 2)
    psnr_val = psnr(X[test_mask], X_rec[test_mask], data_range=4.0)
    rmse = np.sqrt(np.mean((X[test_mask] - X_rec[test_mask]) ** 2))
    nrmse = rmse / 4.0  # divide by 4
    mae = np.mean(np.abs(X[test_mask] - X_rec[test_mask]))

    return mse_val, psnr_val, rmse, nrmse, mae


def run_denoising_experiment_netflix(X, original_mask, sigma=0.1, seed=None):

    n1, n2 = X.shape
    TAU_FACTOR = 7
    tau = TAU_FACTOR * max(n1, n2)
    delta = DELTA_FACTOR

    Y, flip_mask = add_symmetric_noise(X, original_mask, prob=sigma, seed=seed)
    # Omega = np.where(original_mask & (~flip_mask))
    Omega = np.where(original_mask)  # use observed entries
    b = Y[Omega]

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)
    X_rec = np.clip(X_rec, 1, 5)

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

    psnr_val = psnr(X[original_mask], X_rec[original_mask], data_range=4.0)
    rmse = np.sqrt(mse_val)
    nrmse = rmse / 4.0  # divide by 4
    rmse_clean = np.sqrt(mse_clean)
    rmse_corrupted = np.sqrt(mse_corrupted)

    mae = np.mean(np.abs(X[original_mask] - X_rec[original_mask]))

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

    N_TRIALS = 5

    # --- Reconstruction ---
    for s in SPARSITY_LEVELS:
        rmses = []
        maes = []
        psnrs = []

        for trial in range(N_TRIALS):

            seed = SEED + trial
            m, p, r, n, a = run_reconstruction_experiment_netflix(
                X, original_mask, sparsity=s, seed=seed
            )

            rmses.append(r)
            maes.append(a)
            psnrs.append(p)

        reconstruction_results.append(
            {
                "sparsity": s,
                "RMSE_mean": np.mean(rmses),
                "RMSE_std": np.std(rmses, ddof=1),
                "MAE_mean": np.mean(maes),
                "MAE_std": np.std(maes, ddof=1),
                "PSNR_mean": np.mean(psnrs),
                "PSNR_std": np.std(psnrs, ddof=1),
            }
        )

    # --- Denoising ---
    for sigma in NOISE_LEVELS:
        mses = []
        mses_corrupted = []
        mses_clean = []
        rmses = []
        rmses_corrupted = []
        rmses_clean = []
        nrmses = []
        maes = []
        psnrs = []

        for trial in range(N_TRIALS):
            seed = SEED + trial
            m, mc, mclean, p, r, rc, rclean, n, a = run_denoising_experiment_netflix(
                X, original_mask, sigma, seed=seed
            )

            rmses.append(r)
            maes.append(a)
            psnrs.append(p)
            mses.append(m)
            mses_corrupted.append(mc)
            mses_clean.append(mclean)
            rmses_corrupted.append(rc)
            rmses_clean.append(rclean)
            nrmses.append(n)

        denoising_results.append(
            {
                "sigma": sigma,
                "MSE": np.mean(mses),
                "MSE_Corrupted": np.mean(mses_corrupted),
                "MSE_Clean": np.mean(mses_clean),
                "PSNR_mean": np.mean(psnrs),
                "PSNR_std": np.std(psnrs, ddof=1),
                "RMSE_mean": np.mean(rmses),
                "RMSE_std": np.std(rmses, ddof=1),
                "RMSE_Corrupted": np.mean(rmses_corrupted),
                "RMSE_Clean": np.mean(rmses_clean),
                "NRMSE": np.mean(nrmses),
                "MAE_mean": np.mean(maes),
                "MAE_std": np.std(maes, ddof=1),
            }
        )

    pd.DataFrame(reconstruction_results).to_csv(
        RESULTS_PATH / "netflix_reconstruction_v2_notround.csv", index=False
    )
    pd.DataFrame(denoising_results).to_csv(
        RESULTS_PATH / "netflix_denoising_v2_notround.csv", index=False
    )


if __name__ == "__main__":
    main()
