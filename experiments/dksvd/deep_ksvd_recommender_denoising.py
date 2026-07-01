"""
Deep K-SVD denoising for recommender system data using 1D signals.

Each user vector (256 movie ratings) is treated as a 1D signal.
The model extracts overlapping 1D windows (patches) of length PATCH_SIZE,
runs T learned ISTA iterations with a shared dictionary, and folds
the reconstructed patches back into a denoised user vector.

Note: unlike image patches, 1D windows of movie ratings have no
meaningful spatial structure (movie ordering is arbitrary). This
experiment is included for completeness and methodological comparison
rather than expected strong performance.

Mirrors the structure of the classical K-SVD Netflix denoising
experiment (symmetric noise at p=0.1, same metrics).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from src.utils.io import load_netflix_matrix
from src.utils.noise import add_symmetric_noise

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH    = Path("/d/hpc/home/sm79111/thesis/data/netflix")
RESULTS_PATH = Path(
    "/d/hpc/home/sm79111/thesis/results/deep_ksvd/recommender/denoising"
)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────

SIGNAL_LENGTH = 256   # number of movies per user vector
PATCH_SIZE    = 8     # 1D window length (same as image case for comparability)
N_ATOMS       = 64    # 8× overcomplete dictionary
T             = 7     # ISTA unrolling iterations (same as image case)
N_EPOCHS      = 200
LR            = 1e-4
NOISE_LEVEL   = 0.1   # fraction of observed ratings flipped
SEED          = 42

# Map ratings [1, 5] → [-1, 1]  (same normalisation logic as image case)
RATING_MEAN = 3.0
RATING_STD  = 2.0
RATING_MIN  = 1.0
RATING_MAX  = 5.0


# ── 1D overcomplete DCT dictionary ───────────────────────────────────────────

def init_dct_1d(patch_size: int, n_atoms: int) -> np.ndarray:
    """
    Overcomplete 1D DCT dictionary of shape (patch_size, n_atoms).

    Each column is a DCT-II basis function sampled at patch_size points,
    normalised to unit L2 norm. Analogous to init_dct() in Deep_KSVD.py
    but for 1D signals instead of 2D image patches.
    """
    D = np.zeros((patch_size, n_atoms), dtype=np.float32)
    for k in range(n_atoms):
        for n in range(patch_size):
            D[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_atoms))
    norms = np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-8)
    return D / norms


# ── 1D unrolled ISTA model ────────────────────────────────────────────────────

class DenoisingNet_1D(nn.Module):
    """
    Unrolled ISTA denoising network for 1D signals.

    Architecture mirrors DenoisingNet_MLP from Deep_KSVD.py:
      - Unfold: extract overlapping 1D patches via nn.Unfold(kernel=(1, P))
      - T iterations of learned ISTA with shared dictionary D and step c
      - Per-patch learned threshold via a small MLP (content-dependent)
      - Fold: average overlapping reconstructed patches back to signal

    Input/output shape: (1, 1, signal_length)
    """

    def __init__(self, signal_length, patch_size, n_atoms, T,
                 Dict_init, c_init, device):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size    = patch_size
        self.n_atoms       = n_atoms
        self.T             = T
        self.device        = device

        n_patches = signal_length - patch_size + 1  # with stride=1

        # 1D unfold/fold implemented as 2D with height=1
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=(1, 1))
        self.fold   = nn.Fold(
            output_size=(1, signal_length),
            kernel_size=(1, patch_size),
            stride=(1, 1),
        )

        # Precompute overlap-add normalisation (not learnable)
        ones = torch.ones(1, patch_size, n_patches)
        self.register_buffer("count", self.fold(ones))  # (1, 1, 1, signal_length)

        # Learnable dictionary: (patch_size, n_atoms)
        self.D = nn.Parameter(torch.FloatTensor(Dict_init))

        # Learnable step-size parameter (initialised to squared spectral norm)
        self.c = nn.Parameter(c_init.clone().detach())

        # Per-element weight applied to patches before the threshold MLP
        self.w = nn.Parameter(torch.ones(patch_size))

        # MLP: maps one weighted patch → one positive threshold scalar
        # Smaller than image case (patch_size=8 not 64)
        self.W = nn.Sequential(
            nn.Linear(patch_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        for m in self.W.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        """
        x : (1, 1, signal_length)
        returns denoised signal of the same shape
        """
        # Add spatial height=1 for 2D Unfold: (1, 1, 1, signal_length)
        x_2d = x.unsqueeze(2)

        # Extract 1D patches: (1, patch_size, n_patches)
        P = self.unfold(x_2d)
        n_patches = P.shape[2]

        # Normalise dictionary columns at each forward pass
        D = F.normalize(self.D, p=2, dim=0)  # (patch_size, n_atoms)

        # Compute per-patch threshold from patch content (once, before ISTA loop)
        # P[0].T : (n_patches, patch_size)
        lam = torch.abs(self.W(P[0].T * self.w))  # (n_patches, 1)
        lam = lam.T.unsqueeze(0)                   # (1, 1, n_patches)

        # ISTA: z_{t+1} = S_{λ/c}( z_t + (1/c) · D^T (P - D z_t) )
        Z = torch.zeros(1, self.n_atoms, n_patches, device=self.device)
        for _ in range(self.T):
            residual = P - torch.einsum("pa,ban->bpn", D, Z)
            Z = Z + torch.einsum("pa,bpn->ban", D, residual) / self.c
            Z = torch.sign(Z) * F.relu(torch.abs(Z) - lam / self.c)

        # Reconstruct patches from sparse codes
        P_rec = torch.einsum("pa,ban->bpn", D, Z)  # (1, patch_size, n_patches)

        # Fold and normalise by overlap count
        x_rec = self.fold(P_rec) / self.count       # (1, 1, 1, signal_length)

        return x_rec.squeeze(2)  # (1, 1, signal_length)


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    rng = np.random.RandomState(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading Netflix matrix...")
    R_clean, original_mask = load_netflix_matrix(
        DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
    )
    n_users, n_items = R_clean.shape
    print(f"Shape    : {n_users} users × {n_items} movies")
    print(f"Observed : {original_mask.sum()} / {original_mask.size} "
          f"({100 * original_mask.mean():.1f}%)\n")

    # ── Noise ─────────────────────────────────────────────────────────────────
    print(f"Adding symmetric noise  (p={NOISE_LEVEL})...")
    R_noisy, flip_mask = add_symmetric_noise(
        R_clean, mask=original_mask, prob=NOISE_LEVEL, seed=SEED
    )
    rmse_noisy = rmse(R_clean[original_mask], R_noisy[original_mask])
    print(f"Noise RMSE : {rmse_noisy:.4f}")

    # Mean-fill unobserved entries so every user vector is fully defined.
    # Missing entries → global rating mean → normalises to 0, a neutral value
    # that the ISTA loop does not need to explain.
    global_mean = float(R_clean[original_mask].mean())
    R_input = R_noisy.copy()
    R_input[~original_mask] = global_mean

    # Normalise to [-1, 1]
    R_input_norm = (R_input  - RATING_MEAN) / RATING_STD
    R_clean_norm = (R_clean  - RATING_MEAN) / RATING_STD

    # ── Model ─────────────────────────────────────────────────────────────────
    Dict_np = init_dct_1d(PATCH_SIZE, N_ATOMS)
    c_val   = float(linalg.norm(Dict_np, ord=2) ** 2)

    model = DenoisingNet_1D(
        signal_length=SIGNAL_LENGTH,
        patch_size=PATCH_SIZE,
        n_atoms=N_ATOMS,
        T=T,
        Dict_init=Dict_np,
        c_init=torch.FloatTensor([c_val]),
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {N_EPOCHS} epochs over {n_users} users...\n")

    user_indices = np.arange(n_users)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        rng.shuffle(user_indices)
        epoch_loss = 0.0
        n_active   = 0

        for i in user_indices:
            obs = original_mask[i]
            if obs.sum() < PATCH_SIZE:
                continue  # skip users with fewer observations than one patch

            x = (torch.from_numpy(R_input_norm[i])
                 .float().unsqueeze(0).unsqueeze(0).to(device))    # (1, 1, 256)
            target = (torch.from_numpy(R_clean_norm[i])
                      .float().to(device))                          # (256,)

            optimizer.zero_grad()
            pred = model(x)[0, 0]                                   # (256,)

            # Loss on observed entries only
            loss = criterion(pred[obs], target[obs])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_active   += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{N_EPOCHS}  "
                  f"avg loss = {epoch_loss / max(n_active, 1):.5f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating on full dataset...")
    model.eval()
    R_denoised_norm = np.zeros_like(R_clean_norm)

    with torch.no_grad():
        for i in range(n_users):
            x = (torch.from_numpy(R_input_norm[i])
                 .float().unsqueeze(0).unsqueeze(0).to(device))
            R_denoised_norm[i] = model(x)[0, 0].cpu().numpy()

    # Denormalise and clip to valid rating range
    R_denoised = R_denoised_norm * RATING_STD + RATING_MEAN
    R_denoised = np.clip(R_denoised, RATING_MIN, RATING_MAX)

    obs           = original_mask
    clean_entries = original_mask & ~flip_mask

    results = {
        "noise_level":            NOISE_LEVEL,
        "patch_size":             PATCH_SIZE,
        "n_atoms":                N_ATOMS,
        "T":                      T,
        "n_epochs":               N_EPOCHS,
        "rmse_noisy":             rmse_noisy,
        "rmse_all_observed":      rmse(R_clean[obs],           R_denoised[obs]),
        "mae_all_observed":       mae( R_clean[obs],           R_denoised[obs]),
        "rmse_corrupted_entries": rmse(R_clean[flip_mask],     R_denoised[flip_mask]),
        "rmse_clean_entries":     rmse(R_clean[clean_entries], R_denoised[clean_entries]),
    }

    print("\n" + "=" * 55)
    print("DENOISING RESULTS")
    print("=" * 55)
    print(f"Noise RMSE (before)               : {rmse_noisy:.4f}")
    print(f"RMSE all observed  (after)        : {results['rmse_all_observed']:.4f}")
    print(f"MAE  all observed  (after)        : {results['mae_all_observed']:.4f}")
    print(f"RMSE corrupted entries (after)    : {results['rmse_corrupted_entries']:.4f}")
    print(f"RMSE clean entries (after)        : {results['rmse_clean_entries']:.4f}")

    out = RESULTS_PATH / "denoising_results.csv"
    pd.DataFrame([results]).to_csv(out, index=False)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
