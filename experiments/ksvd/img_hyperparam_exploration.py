from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports from the src package work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import pandas as pd
import numpy as np

from src.ksvd.ksvd import ksvd, initialize_dictionary
from src.ksvd.patches import (
    extract_patches,
    reconstruct_from_patches,
)
from src.utils.io import load_image
from src.utils.noise import add_gaussian_noise
from src.utils.metrics import psnr, nrmse

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/CBSD68")

DICT_SIZES = [128, 256, 441, 512]
SPARSITIES = [2, 4, 6, 8]

results = []

# Representative image
X = load_image(DATA_PATH / "0047.png")
X = X[:256, :256]

# Add noise
sigma = 25.5
X_noisy = add_gaussian_noise(X, sigma=sigma)

# Extract overlapping patches
patches = extract_patches(
    X_noisy,
    patch_size=8,
    stride=1,
)

n_patches = patches.shape[1]

for K in DICT_SIZES:
    for sparsity in SPARSITIES:

        print(f"\nK={K}, sparsity={sparsity}")

        # Initialize dictionary
        initial_dictionary = initialize_dictionary(
            patches,
            n_atoms=K,
            random_state=42,
        )

        # Insert fixed DC atom
        initial_dictionary[:, 0] = np.ones(initial_dictionary.shape[0]) / np.sqrt(
            initial_dictionary.shape[0]
        )

        start = time.time()

        D, codes, history = ksvd(
            patches,
            n_atoms=K,
            sparsity=sparsity,
            n_iter=10,
            initial_dictionary=initial_dictionary,
            fixed_atoms=1,
            random_state=42,
            verbose=False,
        )

        elapsed = time.time() - start

        reconstructed_patches = D @ codes

        X_rec = reconstruct_from_patches(
            reconstructed_patches,
            image_shape=X.shape,
            patch_size=8,
            stride=1,
        )

        X_rec = np.clip(X_rec, 0, 255)

        p = psnr(X, X_rec, data_range=255)
        n = nrmse(X, X_rec, data_range=255)

        # Approximate operation estimate
        ops_estimate = n_patches * K * sparsity

        results.append(
            {
                "dictionary_size": K,
                "sparsity": sparsity,
                "PSNR": p,
                "NRMSE": n,
                "runtime_sec": elapsed,
                "n_patches": n_patches,
                "ops_estimate": ops_estimate,
            }
        )

df = pd.DataFrame(results)

print(df.sort_values("PSNR", ascending=False))

df.to_csv("ksvd_hyperparameter_search.csv", index=False)
