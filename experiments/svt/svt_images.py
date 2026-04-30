"""Run SVT experiments on image data."""

import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.svt.svt import svt
from src.utils.masking import create_mask, apply_mask
from src.utils.noise import add_gaussian_noise
from src.utils.io import load_image
from src.utils.metrics import mse, psnr, nmse

from experiments.config import *

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/CBSD68")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/svt/images")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def run_reconstruction_experiment(X, sparsity=0.2):

    observed_fraction = 1 - sparsity
    n1, n2 = X.shape
    tau = 8 * max(n1, n2)
    # delta = DELTA_FACTOR / observed_fraction # -> caused divergence for higher sparsity levels
    delta = DELTA_FACTOR

    mask = create_mask(X.shape, observed_fraction, seed=SEED)
    Omega, b = apply_mask(X, mask)

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)
    X_rec = np.clip(X_rec, 0, 255)
    return mse(X, X_rec), nmse(X, X_rec, data_range=255), psnr(X, X_rec, data_range=255)


def run_denoising_experiment(X, sigma=0.1):

    Y = add_gaussian_noise(X, sigma, normalize=True)

    # FULL observation
    Omega = np.where(np.ones_like(X, dtype=bool))
    b = Y[Omega]

    n1, n2 = X.shape
    tau = TAU_FACTOR * max(n1, n2)
    delta = DELTA_FACTOR

    X_denoised, _ = svt((n1, n2), Omega, b, tau, delta)
    X_denoised = np.clip(X_denoised, 0, 1)
    return (
        mse(X, X_denoised),
        nmse(X, X_denoised, data_range=1.0),
        psnr(X, X_denoised, data_range=1.0),
    )


def main():
    reconstruction_results = []
    denoising_results = []
    # Load image
    for img_path in sorted(DATA_PATH.glob("*.png")):
        X = load_image(img_path)
        X = X[:256, :256]

        # --- Reconstruction ---
        for s in SPARSITY_LEVELS:
            m, n, p = run_reconstruction_experiment(X, s)

            reconstruction_results.append(
                {"image": img_path.name, "sparsity": s, "MSE": m, "NMSE": n, "PSNR": p}
            )

        X = load_image(img_path, normalize=True)
        X = X[:256, :256]

        # --- Denoising ---
        for sigma in NOISE_LEVELS:
            m, n, p = run_denoising_experiment(X, sigma)

            denoising_results.append(
                {"image": img_path.name, "sigma": sigma, "MSE": m, "NMSE": n, "PSNR": p}
            )

    # Save results
    pd.DataFrame(reconstruction_results).to_csv(
        RESULTS_PATH / "reconstruction.csv", index=False
    )

    pd.DataFrame(denoising_results).to_csv(RESULTS_PATH / "denoising.csv", index=False)


if __name__ == "__main__":
    # check synthetic image
    # X = np.random.rand(128, 128)
    # print("Reconstruction results:")
    # run_reconstruction_experiment(X)
    # print("Denoising results:")
    # run_denoising_experiment(X)
    main()
