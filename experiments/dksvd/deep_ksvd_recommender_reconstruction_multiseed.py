"""
Deep K-SVD reconstruction for recommender system data — multi-seed evaluation.

Each user vector is partially masked (simulating missing entries on top of
the Netflix sparsity pattern). The model is trained to predict the full
vector from the partially observed input, then evaluated on the held-out
reconstruction entries.

Mirrors the image reconstruction experiment structure:
  - 80/20 user-level train/test split (same fixed split as denoising)
  - Training: random masks per sample per epoch
  - Evaluation: fixed seeded mask, measured on held-out entries only

Usage:
    python deep_ksvd_recommender_reconstruction_multiseed.py --missing_fraction 0.4
"""

import argparse
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

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH    = Path("/d/hpc/home/sm79111/thesis/data/netflix")
RESULTS_PATH = Path(
    "/d/hpc/home/sm79111/thesis/results/deep_ksvd/recommender/reconstruction"
)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ── Fixed hyperparameters ─────────────────────────────────────────────────────

SIGNAL_LENGTH = 256
PATCH_SIZE    = 8
N_ATOMS       = 64
T             = 7
N_EPOCHS      = 200
LR            = 1e-4
N_SEEDS       = 5
TRAIN_RATIO   = 0.8
SPLIT_SEED    = 0    # same fixed split as denoising experiment
EVAL_SEED     = 42   # fixed evaluation mask, consistent with classical K-SVD

RATING_MEAN = 3.0
RATING_STD  = 2.0
RATING_MIN  = 1.0
RATING_MAX  = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def init_dct_1d(patch_size, n_atoms):
    D = np.zeros((patch_size, n_atoms), dtype=np.float32)
    for k in range(n_atoms):
        for n in range(patch_size):
            D[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_atoms))
    norms = np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-8)
    return D / norms

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def psnr(y_true, y_pred, data_range=4.0):
    mse_val = np.mean((y_true - y_pred) ** 2)
    if mse_val == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse_val))

def make_reconstruction_mask(observed_mask, missing_fraction, rng):
    """
    From the observed entries, hold out `missing_fraction` as the
    reconstruction target. Returns a boolean mask of the same shape:
    True = visible during reconstruction, False = held out.
    """
    recon_mask = observed_mask.copy()
    obs_indices = np.where(observed_mask)[0]
    n_holdout   = int(len(obs_indices) * missing_fraction)
    holdout     = rng.choice(obs_indices, size=n_holdout, replace=False)
    recon_mask[holdout] = False
    return recon_mask   # True = visible, False = held out (to reconstruct)


# ── Model (identical to denoising version) ────────────────────────────────────

class DenoisingNet_1D(nn.Module):

    def __init__(self, signal_length, patch_size, n_atoms, T, Dict_init, c_init, device):
        super().__init__()
        self.n_atoms = n_atoms
        self.T       = T
        self.device  = device

        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=(1, 1))
        self.fold   = nn.Fold(
            output_size=(1, signal_length),
            kernel_size=(1, patch_size),
            stride=(1, 1),
        )
        ones = torch.ones(1, patch_size, signal_length - patch_size + 1)
        self.register_buffer("count", self.fold(ones))

        self.D = nn.Parameter(torch.FloatTensor(Dict_init))
        self.c = nn.Parameter(c_init.clone().detach())
        self.w = nn.Parameter(torch.ones(patch_size))
        self.W = nn.Sequential(
            nn.Linear(patch_size, 16), nn.ReLU(),
            nn.Linear(16, 8),          nn.ReLU(),
            nn.Linear(8, 1),
        )
        for m in self.W.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x_2d = x.unsqueeze(2)
        P    = self.unfold(x_2d)
        D    = F.normalize(self.D, p=2, dim=0)
        lam  = torch.abs(self.W(P[0].T * self.w)).T.unsqueeze(0)
        Z    = torch.zeros(1, self.n_atoms, P.shape[2], device=self.device)
        for _ in range(self.T):
            residual = P - torch.einsum("pa,ban->bpn", D, Z)
            Z = Z + torch.einsum("pa,bpn->ban", D, residual) / self.c
            Z = torch.sign(Z) * F.relu(torch.abs(Z) - lam / self.c)
        return (self.fold(torch.einsum("pa,ban->bpn", D, Z)) / self.count).squeeze(2)


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(R_clean, original_mask, train_idx, test_idx,
               missing_fraction, seed, device):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    global_mean = float(R_clean[original_mask].mean())

    # Fixed evaluation masks for test users (seeded independently of run seed)
    # so the same entries are always held out at evaluation time
    eval_rng = np.random.RandomState(EVAL_SEED)
    eval_recon_masks = {}
    for i in test_idx:
        eval_recon_masks[i] = make_reconstruction_mask(
            original_mask[i], missing_fraction, eval_rng
        )

    # Normalise
    R_clean_norm = (R_clean - RATING_MEAN) / RATING_STD

    # Build model
    Dict_np = init_dct_1d(PATCH_SIZE, N_ATOMS)
    c_val   = float(linalg.norm(Dict_np, ord=2) ** 2)
    model   = DenoisingNet_1D(
        signal_length=SIGNAL_LENGTH, patch_size=PATCH_SIZE,
        n_atoms=N_ATOMS, T=T,
        Dict_init=Dict_np, c_init=torch.FloatTensor([c_val]),
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    train_idx_shuffled = train_idx.copy()

    # ── Training on train users only ──────────────────────────────────────────
    for _ in range(N_EPOCHS):
        model.train()
        rng.shuffle(train_idx_shuffled)

        for i in train_idx_shuffled:
            obs = original_mask[i]
            if obs.sum() < PATCH_SIZE:
                continue

            # Random reconstruction mask: hold out missing_fraction of observed
            recon_mask = make_reconstruction_mask(obs, missing_fraction, rng)

            # Input: mean-fill held-out and originally unobserved entries
            x_np = R_clean[i].copy()
            x_np[~recon_mask] = global_mean
            x_norm = (x_np - RATING_MEAN) / RATING_STD

            x      = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            target = torch.from_numpy(R_clean_norm[i]).float().to(device)

            optimizer.zero_grad()
            # Loss on visible entries only (recon_mask = True entries)
            loss = criterion(model(x)[0, 0][recon_mask], target[recon_mask])
            loss.backward()
            optimizer.step()

    # ── Evaluation on test users only ─────────────────────────────────────────
    model.eval()
    R_reconstructed_norm = np.zeros_like(R_clean_norm)

    with torch.no_grad():
        for i in test_idx:
            recon_mask = eval_recon_masks[i]
            x_np = R_clean[i].copy()
            x_np[~recon_mask] = global_mean
            x_norm = (x_np - RATING_MEAN) / RATING_STD

            x = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            R_reconstructed_norm[i] = model(x)[0, 0].cpu().numpy()

    R_rec = np.clip(R_reconstructed_norm * RATING_STD + RATING_MEAN, RATING_MIN, RATING_MAX)

    # Evaluate on held-out entries only (the entries to be reconstructed)
    held_out_rows, held_out_cols = [], []
    for i in test_idx:
        recon_mask  = eval_recon_masks[i]
        holdout_idx = np.where(original_mask[i] & ~recon_mask)[0]
        for j in holdout_idx:
            held_out_rows.append(i)
            held_out_cols.append(j)

    held_out_rows = np.array(held_out_rows)
    held_out_cols = np.array(held_out_cols)

    true_vals = R_clean[held_out_rows, held_out_cols]
    pred_vals = R_rec[held_out_rows, held_out_cols]

    return {
        "rmse": rmse(true_vals, pred_vals),
        "mae":  mae( true_vals, pred_vals),
        "psnr": psnr(true_vals, pred_vals, data_range=4.0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--missing_fraction", type=float, default=0.4,
                   help="Fraction of observed entries to hold out for reconstruction.")
    p.add_argument("--start_seed", type=int, default=0,
                   help="First seed index. Use to resume after a wall-time kill.")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help="Number of seeds to run from start_seed.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds  = list(range(args.start_seed, args.start_seed + args.n_seeds))
    mf_tag = f"mf{int(args.missing_fraction * 100):03d}"

    print(f"Device           : {device}")
    print(f"Missing fraction : {args.missing_fraction}")
    print(f"Seeds            : {seeds}\n")

    print("Loading Netflix matrix...")
    R_clean, original_mask = load_netflix_matrix(
        DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
    )
    n_users = R_clean.shape[0]
    print(f"Shape : {n_users} users × {R_clean.shape[1]} movies")

    # Fixed split — identical to the denoising experiment
    rng_split = np.random.RandomState(SPLIT_SEED)
    perm      = rng_split.permutation(n_users)
    n_train   = int(TRAIN_RATIO * n_users)
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]
    print(f"Train users : {len(train_idx)}  |  Test users : {len(test_idx)}\n")

    records = []
    for i, seed in enumerate(seeds):
        print(f"── Run {i + 1}/{len(seeds)}  (seed={seed}) ──────────────────")
        result = run_single(R_clean, original_mask, train_idx, test_idx,
                            args.missing_fraction, seed=seed, device=device)
        records.append({**result, "seed": seed})
        print(f"   RMSE: {result['rmse']:.4f}  "
              f"MAE: {result['mae']:.4f}  "
              f"PSNR: {result['psnr']:.4f} dB")

    per_run_path = RESULTS_PATH / f"reconstruction_results_per_run_{mf_tag}.csv"
    df_new = pd.DataFrame(records)
    if per_run_path.exists() and args.start_seed > 0:
        df_runs = pd.concat([pd.read_csv(per_run_path), df_new], ignore_index=True)
    else:
        df_runs = df_new
    df_runs.to_csv(per_run_path, index=False)

    means = df_runs.mean(numeric_only=True)
    stds  = df_runs.std(numeric_only=True)

    print("\n" + "=" * 60)
    print(f"RESULTS  mf={args.missing_fraction}  "
          f"(mean ± std over {len(df_runs)} seeds)")
    print("=" * 60)
    for col in ["rmse", "mae", "psnr"]:
        print(f"  {col:<6}: {means[col]:.4f} ± {stds[col]:.4f}")

    summary = pd.DataFrame({"metric": list(means.index),
                             "mean": means.values, "std": stds.values})
    out = RESULTS_PATH / f"reconstruction_results_summary_{mf_tag}.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
