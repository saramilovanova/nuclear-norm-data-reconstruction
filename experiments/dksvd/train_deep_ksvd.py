"""
Adapted Deep K-SVD training script for HPC (SLURM + V100).

Changes from the original DKSVD_train_model.py:
  - sigma passed as CLI argument (train all 3 models in parallel)
  - per-batch test evaluation removed → evaluate every --eval_every steps
    (this was the main bottleneck: original code ran full-image inference
    every single gradient step, ~29M times per model)
  - Mixed precision (torch.cuda.amp) for ~2x speedup on V100
  - Checkpoint saving with sigma in filename, plus resume support
  - num_workers > 0 for faster data loading on HPC
  - Configurable paths

Usage:
    python train_deep_ksvd.py --sigma 25 --data_dir ./gray \
        --train_list train_gray.txt --test_list test_gray.txt \
        --output_dir ./checkpoints --epochs 3 --amp
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import linalg
import sys

# Deep_KSVD.py must be in the same directory (cloned from the original repo)
sys.path.insert(0, str(Path(__file__).parent))
import Deep_KSVD


# ──────────────────────────── CLI ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Deep K-SVD denoiser")
    p.add_argument("--sigma", type=float, required=True,
                   help="Noise std in [0,255] pixel scale. Use 15, 25, or 50.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=10_000,
                   help="Run test-set PSNR every N gradient steps")
    p.add_argument("--save_every", type=int, default=100_000,
                   help="Save checkpoint every N gradient steps")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory containing grayscale BSDS images")
    p.add_argument("--train_list", type=str, default="train_gray.txt")
    p.add_argument("--test_list", type=str, default="test_gray.txt")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Automatic Mixed Precision (recommended on V100)")
    p.add_argument("--n_test_images", type=int, default=10,
                   help="Number of test images to use for PSNR during training")
    return p.parse_args()


# ──────────────────────────── Model init ────────────────────────────

def build_model(device):
    """Initialise DenoisingNet_MLP exactly as in the original code."""
    patch_size = 8
    m = 16  # init_dct(8, 16) creates a 64×256 overcomplete DCT dictionary

    Dict_init = Deep_KSVD.init_dct(patch_size, m).to(device)
    c_init_val = float(linalg.norm(Dict_init.cpu().numpy(), ord=2) ** 2)
    c_init = torch.FloatTensor([c_init_val]).to(device)
    w_init = torch.normal(mean=1.0, std=0.1 * torch.ones(patch_size ** 2)).float().to(device)

    model = Deep_KSVD.DenoisingNet_MLP(
        patch_size=patch_size,
        D_in=patch_size ** 2,   # 64
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


# ──────────────────────────── Eval helper ────────────────────────────

def evaluate(model, test_loader, device):
    """Return mean PSNR over test images (in standard 0-255 scale)."""
    model.eval()
    psnrs = []
    with torch.no_grad():
        for img_clean, img_noisy in test_loader:
            img_clean = img_clean[0, 0].to(device)   # (H, W)
            img_noisy = img_noisy.to(device)          # (1, 1, H, W)
            pred = model(img_noisy)[0, 0]             # (H, W)
            # Images are normalised to [-1, 1]; 4/MSE gives standard PSNR in dB
            mse = torch.mean((img_clean - pred) ** 2)
            psnr = 10.0 * torch.log10(torch.tensor(4.0) / mse)
            psnrs.append(psnr.item())
    model.train()
    return float(np.mean(psnrs))


# ──────────────────────────── Main ────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and (device.type == "cuda")
    print(f"Device : {device}")
    print(f"AMP    : {use_amp}")
    print(f"sigma  : {args.sigma}")

    # ── Output directory (one per sigma) ──────────────────────────
    out_dir = Path(args.output_dir) / f"sigma{int(args.sigma)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    with open(args.train_list) as f:
        train_names = [l.strip() for l in f if l.strip()]
    with open(args.test_list) as f:
        test_names = [l.strip() for l in f if l.strip()]

    mean, std = 255 / 2, 255 / 2   # normalise [0,255] → [-1,1]
    transform = transforms.Compose([
        Deep_KSVD.Normalize(mean=mean, std=std),
        Deep_KSVD.ToTensor(),
    ])

    train_dataset = Deep_KSVD.SubImagesDataset(
        root_dir=args.data_dir,
        image_names=train_names,
        sub_image_size=128,
        sigma=args.sigma,
        transform=transform,
    )
    test_dataset = Deep_KSVD.FullImagesDataset(
        root_dir=args.data_dir,
        image_names=test_names[: args.n_test_images],
        sigma=args.sigma,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train samples : {len(train_dataset):,}")
    print(f"Test images   : {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(device).to(device)

    # Resume from checkpoint if one exists
    latest_ckpt = out_dir / f"model_sigma{int(args.sigma)}_latest.pth"
    start_iter = 0
    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iter", 0)
        print(f"Resumed from iter {start_iter:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Log file ──────────────────────────────────────────────────
    log_path = out_dir / f"training_log_sigma{int(args.sigma)}.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("iter,train_loss,test_psnr_dB,elapsed_s\n")

    # ── Training loop ─────────────────────────────────────────────
    global_iter = start_iter
    max_iter = args.epochs * len(train_loader)
    running_loss = 0.0
    t_start = time.time()

    print(f"\nStarting training for {max_iter:,} iterations ({args.epochs} epochs).\n")

    for epoch in range(args.epochs):
        for clean, noisy in train_loader:
            if global_iter >= max_iter:
                break

            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            global_iter += 1

            # ── Periodic evaluation ────────────────────────────────
            if global_iter % args.eval_every == 0:
                test_psnr = evaluate(model, test_loader, device)
                avg_loss = running_loss / args.eval_every
                elapsed = time.time() - t_start
                msg = (f"[{global_iter:>8,}/{max_iter:,}] "
                       f"loss={avg_loss:.5f}  PSNR={test_psnr:.2f} dB  "
                       f"elapsed={elapsed/3600:.1f}h")
                print(msg)
                with open(log_path, "a") as f:
                    f.write(f"{global_iter},{avg_loss:.6f},{test_psnr:.4f},{elapsed:.1f}\n")
                running_loss = 0.0

            # ── Checkpoint ────────────────────────────────────────
            if global_iter % args.save_every == 0:
                ckpt_path = out_dir / f"model_sigma{int(args.sigma)}_iter{global_iter}.pth"
                state = {"model": model.state_dict(), "iter": global_iter,
                         "optimizer": optimizer.state_dict()}
                torch.save(state, ckpt_path)
                torch.save(state, latest_ckpt)
                print(f"  → Checkpoint saved: {ckpt_path.name}")

        if global_iter >= max_iter:
            break

    # ── Final save ────────────────────────────────────────────────
    final_path = out_dir / f"model_sigma{int(args.sigma)}_final.pth"
    torch.save({"model": model.state_dict(), "iter": global_iter}, final_path)
    torch.save({"model": model.state_dict(), "iter": global_iter}, latest_ckpt)
    total_time = (time.time() - t_start) / 3600
    print(f"\nDone. Total time: {total_time:.2f}h")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
