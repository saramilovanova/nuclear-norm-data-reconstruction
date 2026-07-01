"""
Deep K-SVD denoising for recommender system data — multi-seed evaluation.

Users are split 80/20 into train/test (fixed split, seeded independently
of the noise seed). The model is trained on training users only and
evaluated exclusively on held-out test users, matching the train/test
separation used in the image experiments.

Usage:
    python deep_ksvd_recommender_denoising_multiseed.py --noise_level 0.05
    python deep_ksvd_recommender_denoising_multiseed.py --noise_level 0.10
    python deep_ksvd_recommender_denoising_multiseed.py --noise_level 0.20
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
from src.utils.noise import add_symmetric_noise

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH    = Path("/d/hpc/home/sm79111/thesis/data/netflix")
RESULTS_PATH = Path(
    "/d/hpc/home/sm79111/thesis/results/deep_ksvd/recommender/denoising"
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
SPLIT_SEED    = 0     # fixed independently of noise seed — same split across all runs
 
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
 
 
# ── Model ─────────────────────────────────────────────────────────────────────
 
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
 
def run_single(R_clean, original_mask, train_idx, test_idx, noise_level, seed, device):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
 
    R_noisy, flip_mask = add_symmetric_noise(
        R_clean, mask=original_mask, prob=noise_level, seed=seed
    )
 
    global_mean  = float(R_clean[original_mask].mean())
    R_input      = R_noisy.copy()
    R_input[~original_mask] = global_mean
    R_input_norm = (R_input - RATING_MEAN) / RATING_STD
    R_clean_norm = (R_clean - RATING_MEAN) / RATING_STD
 
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
            x      = torch.from_numpy(R_input_norm[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            target = torch.from_numpy(R_clean_norm[i]).float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x)[0, 0][obs], target[obs])
            loss.backward()
            optimizer.step()
 
    # ── Evaluation on test users only ─────────────────────────────────────────
    model.eval()
    R_denoised_norm = np.zeros_like(R_clean_norm)
    with torch.no_grad():
        for i in test_idx:
            x = torch.from_numpy(R_input_norm[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            R_denoised_norm[i] = model(x)[0, 0].cpu().numpy()
 
    R_denoised = np.clip(R_denoised_norm * RATING_STD + RATING_MEAN, RATING_MIN, RATING_MAX)
 
    # All masks restricted to test users
    obs_test   = original_mask[test_idx]
    flip_test  = flip_mask[test_idx]
    clean_test = obs_test & ~flip_test
 
    return {
        "rmse_all_observed":      rmse(R_clean[test_idx][obs_test],   R_denoised[test_idx][obs_test]),
        "mae_all_observed":       mae( R_clean[test_idx][obs_test],   R_denoised[test_idx][obs_test]),
        "psnr":                   psnr(R_clean[test_idx][obs_test],   R_denoised[test_idx][obs_test], data_range=4.0),
        "rmse_corrupted_entries": rmse(R_clean[test_idx][flip_test],  R_denoised[test_idx][flip_test]),
        "rmse_clean_entries":     rmse(R_clean[test_idx][clean_test], R_denoised[test_idx][clean_test]),
    }
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noise_level", type=float, required=True,
                   help="Symmetric noise probability. Use 0.05, 0.10, or 0.20.")
    p.add_argument("--start_seed", type=int, default=0,
                   help="First seed index to run. Use to resume after a wall-time kill.")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help="Number of seeds to run from start_seed.")
    return p.parse_args()
 
 
def main():
    args       = parse_args()
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p_tag      = f"p{int(args.noise_level * 100):03d}"
    seeds      = list(range(args.start_seed, args.start_seed + args.n_seeds))
 
    print(f"Device      : {device}")
    print(f"Noise level : {args.noise_level}")
    print(f"Seeds       : {seeds}\n")
 
    print("Loading Netflix matrix...")
    R_clean, original_mask = load_netflix_matrix(
        DATA_PATH / "netflix_dense_mtx_0_925_256_movies.csv"
    )
    n_users = R_clean.shape[0]
    print(f"Shape : {n_users} users × {R_clean.shape[1]} movies")
 
    # Fixed 80/20 split — same across all seeds and noise levels
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
                            args.noise_level, seed=seed, device=device)
        records.append({**result, "seed": seed})
        print(f"   RMSE: {result['rmse_all_observed']:.4f}  "
              f"MAE: {result['mae_all_observed']:.4f}  "
              f"PSNR: {result['psnr']:.4f} dB")
 
    # Append to existing CSV if resuming, otherwise create fresh
    per_run_path = RESULTS_PATH / f"denoising_results_per_run_{p_tag}.csv"
    df_new = pd.DataFrame(records)
    if per_run_path.exists() and args.start_seed > 0:
        df_runs = pd.concat([pd.read_csv(per_run_path), df_new], ignore_index=True)
    else:
        df_runs = df_new
    df_runs.to_csv(per_run_path, index=False)
 
    means = df_runs.mean(numeric_only=True)
    stds  = df_runs.std(numeric_only=True)
 
    print("\n" + "=" * 60)
    print(f"RESULTS  p={args.noise_level}  (mean ± std over {len(df_runs)} seeds)")
    print("=" * 60)
    for col in ["rmse_all_observed", "mae_all_observed", "psnr",
                "rmse_corrupted_entries", "rmse_clean_entries"]:
        print(f"  {col:<30}: {means[col]:.4f} ± {stds[col]:.4f}")
 
    summary = pd.DataFrame({"metric": list(means.index),
                             "mean": means.values, "std": stds.values})
    out = RESULTS_PATH / f"denoising_results_summary_{p_tag}.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved → {out}")
 
 
if __name__ == "__main__":
    main()