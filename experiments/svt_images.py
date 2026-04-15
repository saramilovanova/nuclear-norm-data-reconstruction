"""Run SVT experiments on image data."""

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
from src.utils.masking import create_mask, apply_mask
from src.utils.noise import add_gaussian_noise
from src.utils.io import load_image
from src.utils.metrics import mse, psnr

from experiments.config import *

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/CBSD68")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/svt/images")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def run_reconstruction_experiment(X, sparsity=0.2):

    observed_fraction = 1 - sparsity
    mask = create_mask(X.shape, observed_fraction, seed=SEED)
    Omega, b = apply_mask(X, mask)

    # print("---- DEBUG ----")
    # print("Omega size:", len(Omega[0]))
    # print("b mean/std:", np.mean(b), np.std(b))
    # print("Input min/max:", X.min(), X.max())

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    delta = DELTA_FACTOR / observed_fraction

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # print(
    #     "Reconstruction for {}% observed entries:".format(int(observed_fraction * 100))
    # )
    print("MSE:", mse(X, X_rec))
    print("PSNR:", psnr(X, X_rec))
    return mse(X, X_rec), psnr(X, X_rec)


def run_denoising_experiment(X, sigma=0.1):

    Y = add_gaussian_noise(X, sigma)

    # FULL observation
    Omega = np.where(np.ones_like(X, dtype=bool))
    b = Y[Omega]

    # print("---- DEBUG ----")
    # print("Omega size:", len(Omega[0]))
    # print("b mean/std:", np.mean(b), np.std(b))
    # print("Input min/max:", X.min(), X.max())

    n1, n2 = X.shape
    tau = TAU_FACTOR * np.sqrt(n1 * n2)
    delta = DELTA_FACTOR

    X_rec, _ = svt((n1, n2), Omega, b, tau, delta)

    # print("Noisy difference:", np.linalg.norm(X - X_rec))

    # print("\nDenoising with Gaussian noise (σ = {}):".format(sigma))
    print("MSE:", mse(X, X_rec))
    print("PSNR:", psnr(X, X_rec))
    return mse(X, X_rec), psnr(X, X_rec)


def main():
    reconstruction_results = []
    denoising_results = []
    # Load image
    for img_path in sorted(DATA_PATH.glob("*.png")):
        X = load_image(img_path)

        # --- Reconstruction ---
        for s in SPARSITY_LEVELS:
            m, p = run_reconstruction_experiment(X, s)

            reconstruction_results.append(
                {"image": img_path.name, "sparsity": s, "MSE": m, "PSNR": p}
            )

        # --- Denoising ---
        for sigma in NOISE_LEVELS:
            m, p = run_denoising_experiment(X, sigma)

            denoising_results.append(
                {"image": img_path.name, "sigma": sigma, "MSE": m, "PSNR": p}
            )

    # Save results
    pd.DataFrame(reconstruction_results).to_csv(
        RESULTS_PATH / "reconstruction.csv", index=False
    )

    pd.DataFrame(denoising_results).to_csv(RESULTS_PATH / "denoising.csv", index=False)


if __name__ == "__main__":
    # check synthetic image
    # X = np.random.rand(128, 128)

    # X = load_image(DATA_PATH / "0000.png")

    # print("Reconstruction results:")
    # run_reconstruction_experiment(X)
    # print("Denoising results:")
    # run_denoising_experiment(X)
    main()
