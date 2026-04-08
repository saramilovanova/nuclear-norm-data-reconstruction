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
from skimage.io import imread

from src.svt.svt import svt
from src.utils.masking import create_netflix_mask, apply_mask
from src.utils.noise import add_gaussian_noise
from src.utils.io import load_netflix_matrix
from src.utils.metrics import mse, rmse, psnr

from experiments.config import *

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/netflix")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/svt/recommender")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def run_reconstruction_experiment_netflix(X, original_mask, observed_fraction=0.4):

    mask = create_netflix_mask(original_mask, observed_fraction, seed=SEED)
    Omega, b = apply_mask(X, mask)

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    delta = DELTA_FACTOR / observed_fraction

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # evaluate ONLY where data exists but was hidden
    test_mask = original_mask & (~mask)

    mse_val = np.mean((X[test_mask] - X_rec[test_mask]) ** 2)
    psnr_val = psnr(X[test_mask], X_rec[test_mask])

    print(
        "Reconstruction for {}% observed entries:".format(int(observed_fraction * 100))
    )
    print("MSE:", mse_val)
    print("PSNR:", psnr_val)

    return mse_val, psnr_val


def run_denoising_experiment_netflix(X, original_mask, sigma=0.1):

    Y = add_gaussian_noise(X, sigma)

    # only use observed entries
    Omega = np.where(original_mask)
    b = Y[Omega]

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    delta = DELTA_FACTOR

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # evaluate only observed entries
    mse_val = np.mean((X[original_mask] - X_rec[original_mask]) ** 2)
    psnr_val = psnr(X[original_mask], X_rec[original_mask])

    print("Denoising with sigma={}:".format(sigma))
    print("MSE:", mse_val)
    print("PSNR:", psnr_val)

    return mse_val, psnr_val


def main():

    X, original_mask = load_netflix_matrix(
        DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
    )

    reconstruction_results = []
    denoising_results = []

    # --- Reconstruction ---
    for s in SPARSITY_LEVELS:
        m, p = run_reconstruction_experiment_netflix(X, original_mask, s)

        reconstruction_results.append({"sparsity": s, "MSE": m, "PSNR": p})

    # --- Denoising ---
    for sigma in NOISE_LEVELS:
        m, p = run_denoising_experiment_netflix(X, original_mask, sigma)

        denoising_results.append({"sigma": sigma, "MSE": m, "PSNR": p})

    pd.DataFrame(reconstruction_results).to_csv(
        RESULTS_PATH / "netflix_reconstruction.csv", index=False
    )
    pd.DataFrame(denoising_results).to_csv(
        RESULTS_PATH / "netflix_denoising.csv", index=False
    )


if __name__ == "__main__":
    main()
