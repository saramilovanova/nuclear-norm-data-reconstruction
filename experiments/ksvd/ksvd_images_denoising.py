# NOISE_LEVELS = [12.75, 25.5, 51.0]  # Corresponding to 5%, 10%, 20% of pixel range [0, 255]

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ksvd.ksvd import ksvd, initialize_dictionary
from src.ksvd.patches import (
    extract_patches,
    reconstruct_from_patches,
)

from src.utils.io import load_image
from src.utils.noise import add_gaussian_noise
from src.utils.metrics import mse, nrmse, psnr

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/CBSD68")
RESULTS_PATH = Path("nuclear-norm-data-reconstruction/results/ksvd/images/denoising")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [12.75, 25.5, 51.0]

SPARSITY_BY_SIGMA = {
    12.75: 4,
    25.5: 3,
    51.0: 2,
}

N_ATOMS = 256
N_ITER = 10
PATCH_SIZE = 8
STRIDE = 1
SEED = 42

results = []

for img_path in sorted(DATA_PATH.glob("*.png")):

    print(f"\nProcessing {img_path.name}")

    X = load_image(img_path)
    X = X[:256, :256]

    for sigma in NOISE_LEVELS:

        sparsity = SPARSITY_BY_SIGMA[sigma]

        print(f"  sigma={sigma}, sparsity={sparsity}")

        # Add Gaussian noise
        X_noisy = add_gaussian_noise(X, sigma=sigma)

        # Extract overlapping patches
        patches = extract_patches(
            X_noisy,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
        )

        # Initialize dictionary with fixed DC atom
        initial_dictionary = initialize_dictionary(
            patches,
            n_atoms=N_ATOMS,
            random_state=SEED,
        )

        initial_dictionary[:, 0] = np.ones(initial_dictionary.shape[0]) / np.sqrt(
            initial_dictionary.shape[0]
        )

        D, codes, history = ksvd(
            patches,
            n_atoms=N_ATOMS,
            sparsity=sparsity,
            n_iter=N_ITER,
            initial_dictionary=initial_dictionary,
            fixed_atoms=1,
            random_state=SEED,
            verbose=False,
        )

        reconstructed_patches = D @ codes

        X_rec = reconstruct_from_patches(
            reconstructed_patches,
            image_shape=X.shape,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
        )

        X_rec = np.clip(X_rec, 0, 255)

        results.append(
            {
                "image": img_path.name,
                "sigma": sigma,
                "sparsity": sparsity,
                "NRMSE": nrmse(X, X_rec, data_range=255),
                "PSNR": psnr(X, X_rec, data_range=255),
            }
        )

df = pd.DataFrame(results)

df.to_csv(RESULTS_PATH / "denoising_results.csv", index=False)

print(df.groupby("sigma")[["PSNR", "NRMSE"]].mean())
