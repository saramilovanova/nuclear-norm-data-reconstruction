"""
Evaluate trained Deep K-SVD reconstruction models on CBSD68.

Each model was trained for a specific missing fraction (0.2, 0.4, 0.6)
with masked inputs, so inference is a single forward pass followed by
a data-consistency step — no PnP iterations needed.

Output CSV matches the format of the classical K-SVD reconstruction
results for direct comparison.

Usage:
    python experiments/Deep-K-SVD/evaluate_deep_ksvd_reconstruction.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy import linalg
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import Deep_KSVD

from src.utils.masking import create_mask
from src.utils.metrics import nrmse, psnr

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH      = Path("/d/hpc/home/sm79111/thesis/data/CBSD68")
CHECKPOINT_DIR = Path("/d/hpc/home/sm79111/thesis/results/deep_ksvd/checkpoints_recon")
RESULTS_PATH   = Path("/d/hpc/home/sm79111/thesis/results/deep_ksvd/images/reconstruction_trained")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────

MISSING_FRACTIONS = [0.2, 0.4, 0.6]
SEED = 42


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(device):
    patch_size = 8
    m = 16
    Dict_init = Deep_KSVD.init_dct(patch_size, m).to(device)
    c_val = float(linalg.norm(Dict_init.cpu().numpy(), ord=2) ** 2)
    c_init = torch.FloatTensor([c_val]).to(device)
    w_init = torch.ones(patch_size ** 2).float().to(device)
    return Deep_KSVD.DenoisingNet_MLP(
        patch_size=patch_size, D_in=patch_size ** 2,
        H_1=128, H_2=64, H_3=32, D_out_lam=1, T=7,
        min_v=-1, max_v=1,
        Dict_init=Dict_init, c_init=c_init, w_init=w_init, device=device,
    )


def load_model(missing_fraction, device):
    return load_model_for_checkpoint(missing_fraction, device)


def load_model_for_checkpoint(missing_fraction, device, checkpoint_path=None):
    mf_tag = f"mf{int(missing_fraction * 100)}"
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            model = build_model(device)
            raw = torch.load(p, map_location=device)
            model.load_state_dict(raw["model"] if "model" in raw else raw)
            model.to(device).eval()
            print(f"Loaded {p.name}  (iter={raw.get('iter', '?')})")
            return model
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt_dir = CHECKPOINT_DIR / mf_tag
    for name in [f"model_{mf_tag}_final.pth", f"model_{mf_tag}_latest.pth"]:
        p = ckpt_dir / name
        if p.exists():
            model = build_model(device)
            raw = torch.load(p, map_location=device)
            model.load_state_dict(raw["model"] if "model" in raw else raw)
            model.to(device).eval()
            print(f"Loaded {p.name}  (iter={raw.get('iter', '?')})")
            return model
    raise FileNotFoundError(
        f"No checkpoint found for missing_fraction={missing_fraction} in {ckpt_dir}"
    )


def load_gray(path: Path) -> np.ndarray:
    """Load an image (color or gray) and return float32 in [0, 255]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    # img = img[:256, :256] # below when loading it is cropped
    return img.astype(np.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

def reconstruct(model, X, mask, device):
    """
    Single forward pass through the reconstruction-trained network,
    followed by a data-consistency step to restore observed pixels exactly.
    """
    mean, std = 127.5, 127.5

    X_masked = X.copy()
    X_masked[~mask] = 0.0

    X_norm = (X_masked - mean) / std
    t = torch.from_numpy(X_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(t)[0, 0].cpu().numpy()

    pred = pred_norm * std + mean

    # Restore observed pixels exactly
    X_rec = mask * X + (1 - mask) * pred
    return np.clip(X_rec, 0, 255)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a specific checkpoint .pth file to load",
    )
    parser.add_argument(
        "--missing_fraction",
        type=float,
        default=None,
        help="If using --checkpoint_path, evaluate only this missing fraction",
    )
    args = parser.parse_args()

    img_paths = sorted(DATA_PATH.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {DATA_PATH}")
    print(f"Found {len(img_paths)} images in {DATA_PATH}\n")

    results = []

    if args.checkpoint_path is not None:
        if args.missing_fraction is not None:
            fractions_to_eval = [args.missing_fraction]
        else:
            match = re.search(r"mf(\d+)", str(args.checkpoint_path))
            if not match:
                raise ValueError(
                    "When using --checkpoint_path, also pass --missing_fraction "
                    "or use a checkpoint path containing 'mfXX'."
                )
            fractions_to_eval = [int(match.group(1)) / 100.0]
    else:
        fractions_to_eval = MISSING_FRACTIONS

    for mf in fractions_to_eval:
        model = load_model_for_checkpoint(mf, device, checkpoint_path=args.checkpoint_path)

        print(f"\n=== missing_fraction={mf} ===")

        for img_path in img_paths:
            X = load_gray(img_path)
            X = X[:256, :256]

            mask = create_mask(X.shape, observed_fraction=1 - mf, seed=SEED)
            X_rec = reconstruct(model, X, mask, device)

            results.append({
                "image":            img_path.name,
                "missing_fraction": mf,
                "NRMSE":            nrmse(X, X_rec, data_range=255),
                "PSNR":             psnr(X, X_rec,  data_range=255),
            })
            print(f"  {img_path.name:<30}  PSNR={results[-1]['PSNR']:.2f} dB")

    df = pd.DataFrame(results)
    out = RESULTS_PATH / "reconstruction_results.csv"
    df.to_csv(out, index=False)

    print("\n================================================")
    print("Average reconstruction performance")
    print("================================================")
    print(df.groupby("missing_fraction")[["PSNR", "NRMSE"]].mean().round(4))
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
