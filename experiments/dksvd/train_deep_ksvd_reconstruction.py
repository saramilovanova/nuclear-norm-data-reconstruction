"""
Deep K-SVD training for image reconstruction (inpainting).

Identical to the denoising training script except the noise model is
replaced with a random binary mask. One model is trained per missing
fraction. The architecture and loss are unchanged.

Usage:
    python train_deep_ksvd_reconstruction.py \
        --missing_fraction 0.4 \
        --data_dir experiments/Deep-K-SVD/gray \
        --train_list experiments/Deep-K-SVD/train_gray.txt \
        --test_list  experiments/Deep-K-SVD/test_gray.txt \
        --output_dir results/deep_ksvd/checkpoints_recon \
        --epochs 3 --amp
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import linalg
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent))
import Deep_KSVD


# ── Dataset ───────────────────────────────────────────────────────────────────

class MaskedSubImagesDataset(Dataset):
    """
    Same as SubImagesDataset but corrupts patches with a random binary mask
    instead of Gaussian noise. Each __getitem__ call draws a fresh random
    mask so the model sees varied mask patterns across training.
    """

    def __init__(self, root_dir, image_names, sub_image_size,
                 missing_fraction, transform=None):
        self.images = [
            cv2.imread(str(Path(root_dir) / name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            for name in image_names
        ]
        self.sub_image_size = sub_image_size
        self.missing_fraction = missing_fraction
        self.transform = transform

        w, h = self.images[0].shape
        self.number_sub_images = (w - sub_image_size + 1) * (h - sub_image_size + 1)
        self.number_images = len(self.images)

    def __len__(self):
        return self.number_images * self.number_sub_images

    def __getitem__(self, idx):
        # Deterministic crop position (same as original SubImagesDataset)
        img_idx, sub_idx = divmod(idx, self.number_sub_images)
        np.random.seed(idx)
        image = self.images[img_idx]
        w, h = image.shape
        i = np.random.randint(0, w - self.sub_image_size + 1)
        j = np.random.randint(0, h - self.sub_image_size + 1)
        clean = image[i: i + self.sub_image_size, j: j + self.sub_image_size].copy()
        clean = clean.reshape(1, self.sub_image_size, self.sub_image_size)

        # Fresh random mask each call (not seeded) → varied patterns per epoch
        mask = (np.random.random(clean.shape) > self.missing_fraction).astype(np.float32)
        masked = clean * mask

        if self.transform:
            clean = self.transform(clean)
            masked = self.transform(masked)

        return clean, masked


class MaskedFullImagesDataset(Dataset):
    """
    Full images with a fixed seeded mask for consistent PSNR tracking
    during training.
    """

    def __init__(self, root_dir, image_names, missing_fraction, transform=None):
        self.images = [
            cv2.imread(str(Path(root_dir) / name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            for name in image_names
        ]
        self.missing_fraction = missing_fraction
        self.transform = transform

        # Pre-generate fixed masks (seeded) for reproducible test PSNR
        self.masks = []
        for k, img in enumerate(self.images):
            rng = np.random.RandomState(seed=k)
            self.masks.append(
                (rng.random(img.shape) > missing_fraction).astype(np.float32)
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx].copy().reshape(1, *self.images[idx].shape)
        masked = clean * self.masks[idx]

        if self.transform:
            clean = self.transform(clean)
            masked = self.transform(masked)

        return clean, masked


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(device):
    patch_size = 8
    m = 16
    Dict_init = Deep_KSVD.init_dct(patch_size, m).to(device)
    c_val = float(linalg.norm(Dict_init.cpu().numpy(), ord=2) ** 2)
    c_init = torch.FloatTensor([c_val]).to(device)
    w_init = torch.normal(mean=1.0, std=0.1 * torch.ones(patch_size ** 2)).float().to(device)
    return Deep_KSVD.DenoisingNet_MLP(
        patch_size=patch_size, D_in=patch_size ** 2,
        H_1=128, H_2=64, H_3=32, D_out_lam=1, T=7,
        min_v=-1, max_v=1,
        Dict_init=Dict_init, c_init=c_init, w_init=w_init, device=device,
    )


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, test_loader, device):
    model.eval()
    psnrs = []
    with torch.no_grad():
        for clean, masked in test_loader:
            clean = clean[0, 0].to(device)
            masked = masked.to(device)
            pred = model(masked)[0, 0]
            mse = torch.mean((clean - pred) ** 2)
            psnrs.append((10.0 * torch.log10(torch.tensor(4.0) / mse)).item())
    model.train()
    return float(np.mean(psnrs))


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--missing_fraction", type=float, required=True,
                   help="Fraction of missing pixels. Use 0.2, 0.4, or 0.6.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--save_every", type=int, default=100_000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_list", type=str, default="train_gray.txt")
    p.add_argument("--test_list", type=str, default="test_gray.txt")
    p.add_argument("--output_dir", type=str, default="./checkpoints_recon")
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--n_test_images", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mf_tag = f"mf{int(args.missing_fraction * 100)}"  # e.g. "mf40"
    use_amp = args.amp and (device.type == "cuda")

    print(f"Device           : {device}")
    print(f"AMP              : {use_amp}")
    print(f"Missing fraction : {args.missing_fraction}")

    out_dir = Path(args.output_dir) / mf_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_list) as f:
        train_names = [l.strip() for l in f if l.strip()]
    with open(args.test_list) as f:
        test_names = [l.strip() for l in f if l.strip()]

    mean, std = 127.5, 127.5
    transform = transforms.Compose([
        Deep_KSVD.Normalize(mean=mean, std=std),
        Deep_KSVD.ToTensor(),
    ])

    train_dataset = MaskedSubImagesDataset(
        root_dir=args.data_dir,
        image_names=train_names,
        sub_image_size=128,
        missing_fraction=args.missing_fraction,
        transform=transform,
    )
    test_dataset = MaskedFullImagesDataset(
        root_dir=args.data_dir,
        image_names=test_names[: args.n_test_images],
        missing_fraction=args.missing_fraction,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train samples : {len(train_dataset):,}")
    print(f"Test images   : {len(test_dataset)}")

    model = build_model(device).to(device)

    # Resume if checkpoint exists
    latest_ckpt = out_dir / f"model_{mf_tag}_latest.pth"
    start_iter = 0
    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iter", 0)
        print(f"Resumed from iter {start_iter:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    log_path = out_dir / f"training_log_{mf_tag}.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("iter,train_loss,test_psnr_dB,elapsed_s\n")

    global_iter = start_iter
    max_iter = args.epochs * len(train_loader)
    running_loss = 0.0
    t_start = time.time()

    print(f"\nStarting training for {max_iter:,} iterations ({args.epochs} epochs).\n")

    for epoch in range(args.epochs):
        for clean, masked in train_loader:
            if global_iter >= max_iter:
                break

            clean  = clean.to(device, non_blocking=True)
            masked = masked.to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(masked)
                    loss = criterion(pred, clean)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(masked)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            global_iter  += 1

            if global_iter % args.eval_every == 0:
                test_psnr = evaluate(model, test_loader, device)
                avg_loss  = running_loss / args.eval_every
                elapsed   = time.time() - t_start
                print(f"[{global_iter:>8,}/{max_iter:,}] "
                      f"loss={avg_loss:.5f}  PSNR={test_psnr:.2f} dB  "
                      f"elapsed={elapsed/3600:.1f}h")
                with open(log_path, "a") as f:
                    f.write(f"{global_iter},{avg_loss:.6f},{test_psnr:.4f},{elapsed:.1f}\n")
                running_loss = 0.0

            if global_iter % args.save_every == 0:
                ckpt_path = out_dir / f"model_{mf_tag}_iter{global_iter}.pth"
                state = {"model": model.state_dict(), "iter": global_iter,
                         "optimizer": optimizer.state_dict()}
                torch.save(state, ckpt_path)
                torch.save(state, latest_ckpt)
                print(f"  → Checkpoint: {ckpt_path.name}")

        if global_iter >= max_iter:
            break

    final_path = out_dir / f"model_{mf_tag}_final.pth"
    torch.save({"model": model.state_dict(), "iter": global_iter}, final_path)
    torch.save({"model": model.state_dict(), "iter": global_iter}, latest_ckpt)
    print(f"\nDone. Final model: {final_path}")


if __name__ == "__main__":
    main()
