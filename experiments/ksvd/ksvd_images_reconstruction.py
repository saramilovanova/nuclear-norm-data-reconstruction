"""
Paper-faithful K-SVD image reconstruction / inpainting benchmark.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ksvd.ksvd import (
    ksvd,
    initialize_dictionary,
)

from src.ksvd.omp import omp_batch

from src.ksvd.patches import (
    extract_patches,
    reconstruct_from_patches,
)

from src.utils.io import load_image
from src.utils.masking import create_mask
from src.utils.metrics import (
    mse,
    nrmse,
    psnr,
)

DATA_PATH = Path("nuclear-norm-data-reconstruction/data/CBSD68")

RESULTS_PATH = Path(
    "nuclear-norm-data-reconstruction/results/ksvd/images/reconstruction"
)

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Missing-entry fractions
MISSING_FRACTIONS = [0.2, 0.4, 0.6]

# K-SVD parameters
PATCH_SIZE = 8
STRIDE = 8  # non-overlapping patches

N_ATOMS = 441
SPARSITY = 10
N_ITER = 10

SEED = 42

# Optional runtime reduction
# MAX_PATCHES = 20000


# ============================================================
def train_universal_dictionary():

    rng = np.random.default_rng(SEED)

    all_patches = []

    image_paths = sorted(DATA_PATH.glob("*.png"))

    print("\nCollecting training patches...")

    for img_path in image_paths:

        print(f"  {img_path.name}")

        X = load_image(img_path)
        X = X[:256, :256]

        patches = extract_patches(
            X,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
        )

        all_patches.append(patches)

    all_patches = np.concatenate(all_patches, axis=1)

    print(f"\nTotal patches before subsampling: {all_patches.shape[1]}")

    # Random patch subsampling
    # if all_patches.shape[1] > MAX_PATCHES:

    #     idx = rng.choice(
    #         all_patches.shape[1],
    #         size=MAX_PATCHES,
    #         replace=False,
    #     )

    #     all_patches = all_patches[:, idx]

    print(f"Training patches used: {all_patches.shape[1]}")

    # Initialize dictionary
    D0 = initialize_dictionary(
        all_patches,
        n_atoms=N_ATOMS,
        random_state=SEED,
    )

    D0[:, 0] = np.ones(D0.shape[0]) / np.sqrt(D0.shape[0])

    print("\nTraining universal dictionary...")

    D, _, history = ksvd(
        all_patches,
        n_atoms=N_ATOMS,
        sparsity=SPARSITY,
        n_iter=N_ITER,
        initial_dictionary=D0,
        fixed_atoms=1,
        random_state=SEED,
        verbose=True,
    )

    return D, history


def reconstruct_image_with_mask(
    image_corrupted,
    mask,
    D,
):

    patches, positions = extract_patches(
        image_corrupted,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        return_positions=True,
    )

    patch_masks = extract_patches(
        mask.astype(float),
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    ).astype(bool)

    codes = omp_batch(
        patches,
        D,
        sparsity=SPARSITY,
        masks=patch_masks,
        normalize_masked_dictionary=True,
    )

    reconstructed_patches = D @ codes

    image_rec = reconstruct_from_patches(
        reconstructed_patches,
        image_shape=image_corrupted.shape,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        positions=positions,
    )

    image_rec = mask * image_corrupted + (1 - mask) * image_rec

    image_rec = np.clip(image_rec, 0, 255)

    return image_rec


def main():

    D, history = train_universal_dictionary()

    np.save(
        RESULTS_PATH / "universal_dictionary.npy",
        D,
    )

    results = []

    image_paths = sorted(DATA_PATH.glob("*.png"))

    for img_path in image_paths:

        print(f"\nTesting image: {img_path.name}")

        X = load_image(img_path)
        X = X[:256, :256]

        for missing_fraction in MISSING_FRACTIONS:

            observed_fraction = 1 - missing_fraction

            print(f"  Missing fraction: {missing_fraction}")

            mask = create_mask(
                X.shape,
                observed_fraction=observed_fraction,
                seed=SEED,
            )

            X_corrupted = X.copy()
            X_corrupted[~mask] = 0

            X_rec = reconstruct_image_with_mask(
                X_corrupted,
                mask,
                D,
            )

            results.append(
                {
                    "image": img_path.name,
                    "missing_fraction": missing_fraction,
                    "NRMSE": nrmse(
                        X,
                        X_rec,
                        data_range=255,
                    ),
                    "PSNR": psnr(
                        X,
                        X_rec,
                        data_range=255,
                    ),
                }
            )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_PATH / "reconstruction_results.csv",
        index=False,
    )

    print("\n================================================")
    print("Average reconstruction performance")
    print("================================================")

    print(df.groupby("missing_fraction")[["PSNR", "NRMSE"]].mean())

    print("\nResults saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()
