"""
Evaluate trained Deep K-SVD models on CBSD68 (or any folder of images).
Outputs a CSV matching the format of the classical k-SVD results, so
the two can be compared directly.

Usage (after training is complete):
    python evaluate_deep_ksvd.py \
        --data_dir /path/to/CBSD68 \
        --checkpoint_dir ./checkpoints \
        --output_dir ../../results/deep_ksvd/images/denoising

The script looks for:
    <checkpoint_dir>/sigma<N>/model_sigma<N>_final.pth
  or, if training was interrupted:
    <checkpoint_dir>/sigma<N>/model_sigma<N>_latest.pth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent))
import Deep_KSVD


# ──────────────────────────── Helpers ────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    """Load an image (color or gray) and return float32 in [0, 255]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    img = img[:256, :256]
    return img.astype(np.float32)


def psnr(img_ref: np.ndarray, img_test: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((img_ref.astype(np.float64) - img_test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def nrmse(img_ref: np.ndarray, img_test: np.ndarray, data_range: float = 255.0) -> float:
    return np.sqrt(np.mean((img_ref - img_test) ** 2)) / data_range


def build_model(device) -> torch.nn.Module:
    patch_size = 8
    m = 16
    Dict_init = Deep_KSVD.init_dct(patch_size, m).to(device)
    c_val = float(linalg.norm(Dict_init.cpu().numpy(), ord=2) ** 2)
    c_init = torch.FloatTensor([c_val]).to(device)
    w_init = torch.ones(patch_size ** 2).float().to(device)  # neutral init for eval

    model = Deep_KSVD.DenoisingNet_MLP(
        patch_size=patch_size,
        D_in=patch_size ** 2,
        H_1=128, H_2=64, H_3=32,
        D_out_lam=1,
        T=7,
        min_v=-1, max_v=1,
        Dict_init=Dict_init,
        c_init=c_init,
        w_init=w_init,
        device=device,
    )
    return model


def denoise(model: torch.nn.Module, img_np: np.ndarray,
            sigma: float, device, seed: int = 42) -> np.ndarray:
    """Add noise and denoise a single grayscale image (float32, 0-255 scale)."""
    mean, std = 127.5, 127.5

    np.random.seed(seed)
    noisy = img_np + sigma * np.random.randn(*img_np.shape).astype(np.float32)

    # Normalise to [-1, 1]
    noisy_norm = (noisy - mean) / std
    tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(tensor)[0, 0].cpu().numpy()

    # Denormalise back to [0, 255]
    pred = pred_norm * std + mean
    pred = np.clip(pred, 0, 255)
    return pred


# ──────────────────────────── Main ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CBSD68 image folder")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory containing sigma<N> subdirectories")
    p.add_argument("--sigmas", type=float, nargs="+", default=[15.0, 25.0, 50.0])
    p.add_argument("--output_dir", type=str,
                   default="./results/deep_ksvd/images/denoising")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}\n")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    print(f"Found {len(img_paths)} images in {data_dir}\n")

    results = []

    for sigma in args.sigmas:
        ckpt_dir = Path(args.checkpoint_dir) / f"sigma{int(sigma)}"
        final_ckpt = ckpt_dir / f"model_sigma{int(sigma)}_final.pth"
        latest_ckpt = ckpt_dir / f"model_sigma{int(sigma)}_latest.pth"

        if final_ckpt.exists():
            ckpt_path = final_ckpt
        elif latest_ckpt.exists():
            ckpt_path = latest_ckpt
            print(f"[sigma={sigma}] NOTE: using latest (not final) checkpoint.")
        else:
            print(f"[sigma={sigma}] No checkpoint found — skipping.")
            continue

        # Load model
        model = build_model(device)
        raw = torch.load(ckpt_path, map_location=device)
        state_dict = raw["model"] if "model" in raw else raw
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        n_iter = raw.get("iter", "?")
        print(f"=== sigma={sigma}  checkpoint: {ckpt_path.name}  (iter={n_iter}) ===")

        for img_path in img_paths:
            X = load_gray(img_path)
            X_rec = denoise(model, X, sigma=sigma, device=device)

            row = {
                "image": img_path.name,
                "sigma": sigma,
                "NRMSE": nrmse(X, X_rec, data_range=255),
                "PSNR": psnr(X, X_rec, data_range=255),
            }
            results.append(row)
            print(f"  {img_path.name:<30}  PSNR={row['PSNR']:.2f} dB")

    if not results:
        print("No results — did training complete?")
        return

    df = pd.DataFrame(results)
    csv_path = out_dir / "denoising_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    print(df.groupby("sigma")[["PSNR", "NRMSE"]].mean().round(4))


if __name__ == "__main__":
    main()
