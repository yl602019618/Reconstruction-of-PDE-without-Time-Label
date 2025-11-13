#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation for F_x, F_y prediction using PermInvUNet_attn.

- EXACT same normalization as training:
  trajectories * 1e10, F * 1e12, then standardize with TRAIN stats.
- Robust checkpoint loading (handles 'module.' prefix and formats).
- Evaluate indices [--start, --end], inclusive.
- Save per-sample npy (denormalized), and PNGs for Fx/Fy comparisons.
- Write metrics.csv with relative L2 errors (optional, helpful).
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv

from NIOModules import NIOFP2D, NIOFP2D_FNO, NIOFP2D_FNO_attn
# ------------------------------- Constants ------------------------------- #
TRAJ_SCALE = 1e10      # trajectories multiplier used during training
F_SCALE    = 1e12      # F multiplier (for both Fx and Fy) used during training

DEFAULT_TRAIN  = "/home/ubuntu/unlabelPDE_official/2d_force/dataset_2D_drift.npz"
DEFAULT_TEST   = "/home/ubuntu/unlabelPDE_official/2d_force/test_dataset_2D_drift.npz"
DEFAULT_CKPT   = "/home/ubuntu/unlabelPDE_official/2d_force/result_2d_nio/model_checkpoint_best_0.049776.pt"
DEFAULT_OUTDIR = "result_fig/fno"

# ------------------------------- Utils ---------------------------------- #
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_train_stats(train_npz_path: str):
    """
    Compute mean/std from TRAIN file with the SAME scaling/axes as training.
    Returns dict with traj_mean/std (shape (1,1,Nx,Ny)) and F_mean/std (shape (1,Nx,Ny,2)).
    """
    data = np.load(train_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32) * TRAJ_SCALE  # (M, T, Nx, Ny)
    F            = np.array(data["F"],           dtype=np.float32) * F_SCALE      # (M, 2, Nx, Ny)

    # traj stats across (samples, time)
    traj_mean = trajectories.mean(axis=(0, 1), keepdims=True)   # (1,1,Nx,Ny)
    traj_std  = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

    # F stats across samples; keep channel dimension
    # training里标准化是逐点逐通道： (F - F_mean)/F_std
    F_mean = F.mean(axis=0, keepdims=True)                      # (1, 2, Nx, Ny)
    F_std  = F.std(axis=0, keepdims=True) + 1e-8                # (1, 2, Nx, Ny)

    # 为了和训练时 DataLoader 输出 (Nx,Ny,2) 一致，后续会做通道维变换
    # 这里仍保留 (1,2,Nx,Ny)，便于逻辑清晰
    return {
        "traj_mean": traj_mean, "traj_std": traj_std,
        "F_mean": F_mean, "F_std": F_std,
    }

def load_test_sample(test_npz_path: str, index: int):
    """
    Load one sample from TEST file (ORIGINAL units, before any scaling).
    Returns:
        traj_raw: (T, Nx, Ny)
        Fx_raw:   (Nx, Ny)
        Fy_raw:   (Nx, Ny)
    """
    data = np.load(test_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32)  # (M, T, Nx, Ny)
    F = np.array(data["F"], dtype=np.float32)                        # (M, 2, Nx, Ny)

    if index < 0 or index >= trajectories.shape[0]:
        raise IndexError(f"Index {index} out of range [0, {trajectories.shape[0]-1}]")

    traj_raw = trajectories[index]        # (T, Nx, Ny)
    Fx_raw   = F[index, 0]               # (Nx, Ny)
    Fy_raw   = F[index, 1]               # (Nx, Ny)
    return traj_raw, Fx_raw, Fy_raw

def normalize_input(traj_raw: np.ndarray, stats: dict):
    """
    Normalize one trajectory using TRAIN stats & scaling; return tensor (1, T, Nx, Ny).
    """
    traj_scaled = traj_raw * TRAJ_SCALE
    traj_norm = (traj_scaled - stats["traj_mean"].squeeze(0)) / stats["traj_std"].squeeze(0)
    return torch.tensor(traj_norm[None, ...], dtype=torch.float32)

def denormalize_output(pred_norm: torch.Tensor, stats: dict):
    """
    Denormalize model outputs to ORIGINAL units.
    Accepts pred_norm shape (B, Nx, Ny, 2) or (B, 2, Nx, Ny).
    Returns:
        Fx_pred, Fy_pred as numpy arrays (Nx, Ny) in ORIGINAL units.
    """
    if pred_norm.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {pred_norm.shape}")

    if pred_norm.shape[-1] == 2:
        Fx_norm = pred_norm[0, ..., 0].cpu().numpy()
        Fy_norm = pred_norm[0, ..., 1].cpu().numpy()
    elif pred_norm.shape[1] == 2:
        Fx_norm = pred_norm[0, 0, ...].cpu().numpy()
        Fy_norm = pred_norm[0, 1, ...].cpu().numpy()
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {pred_norm.shape}")

    # stats["F_mean"], stats["F_std"]: (1,2,Nx,Ny) -> to (Nx,Ny,2) for convenience
    F_mean_chlast = np.transpose(stats["F_mean"].squeeze(0), (1, 2, 0))  # (Nx,Ny,2)
    F_std_chlast  = np.transpose(stats["F_std"].squeeze(0),  (1, 2, 0))  # (Nx,Ny,2)

    Fx_scaled = Fx_norm * F_std_chlast[..., 0] + F_mean_chlast[..., 0]
    Fy_scaled = Fy_norm * F_std_chlast[..., 1] + F_mean_chlast[..., 1]

    Fx_orig = Fx_scaled / F_SCALE
    Fy_orig = Fy_scaled / F_SCALE
    return Fx_orig, Fy_orig

def load_checkpoint_robust(ckpt_path: str, device="cpu"):
    """
    Robustly load a checkpoint:
    - supports raw state_dict or dict with 'state_dict' key
    - strips 'module.' prefix if present
    """
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], (dict, OrderedDict)):
        state_dict = raw["state_dict"]
    elif isinstance(raw, (dict, OrderedDict)):
        state_dict = raw
    else:
        raise RuntimeError(f"Unrecognized checkpoint format at {ckpt_path}")

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd

def rel_l2(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    """Relative L2 error ||a-b||_2 / ||b||_2 with small eps for stability."""
    num = np.linalg.norm((a - b).ravel(), 2)
    den = np.linalg.norm(b.ravel(), 2) + eps
    return float(num / den)

def save_comparison_fig(pair_true: np.ndarray, pair_pred: np.ndarray,
                        title_true: str, title_pred: str, out_path: str):
    """
    Save side-by-side comparison figure with shared, symmetric color limits.
    """
    vmax = float(np.max(np.abs([pair_true, pair_pred])))
    vmin = -vmax
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(pair_true, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(title_true)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pair_pred, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(title_pred)
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle(out_path.split(os.sep)[-1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Batch evaluate UNet for Fx, Fy prediction.")
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN,  help="TRAIN npz path (for stats).")
    parser.add_argument("--test_data",  type=str, default=DEFAULT_TEST,   help="TEST  npz path (evaluation set).")
    parser.add_argument("--ckpt",       type=str, default=DEFAULT_CKPT,   help="Model checkpoint (.pt).")
    parser.add_argument("--outdir",     type=str, default=DEFAULT_OUTDIR, help="Output dir for figures and npy.")
    parser.add_argument("--start",      type=int, default=0,              help="Start index (inclusive).")
    parser.add_argument("--end",        type=int, default=32,             help="End index (inclusive).")
    parser.add_argument("--nx",         type=int, default=80,             help="Grid Nx used in model.")
    parser.add_argument("--ny",         type=int, default=80,             help="Grid Ny used in model.")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device, e.g., 'cuda' or 'cpu'.")
    parser.add_argument("--strict",     action="store_true",
                        help="Use strict=True for load_state_dict (default False).")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # 1) TRAIN stats
    print(f"[Info] Loading TRAIN stats from: {args.train_data}")
    stats = compute_train_stats(args.train_data)

    # 2) Build model once
    print(f"[Info] Building model PermInvUNet_attn(depth=5) and loading checkpoint: {args.ckpt}")
    input_dimensions_trunk = 2
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 12
    modes = 32
    output_dim = 2
    nx = 80
    ny = 80

    model = NIOFP2D_FNO(input_dimensions_trunk,
                        n_hidden_layers,
                        neurons,
                        n_basis,
                        fno_layers,
                        width,
                        modes,
                        output_dim)

    model = model.to(args.device)
    model.eval()

    # 3) Load checkpoint robustly once
    sd = load_checkpoint_robust(args.ckpt, device="cpu")
    ret = model.load_state_dict(sd, strict=args.strict)
    if ret is not None and hasattr(ret, "missing_keys") and (ret.missing_keys or ret.unexpected_keys):
        print("[Warn] Incompatible keys when loading:")
        if ret.missing_keys:
            print("  Missing keys:", ret.missing_keys)
        if ret.unexpected_keys:
            print("  Unexpected keys:", ret.unexpected_keys)
    Nx, Ny = 80, 80
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx, dtype=np.float32),
                                np.linspace(-1, 1, Ny, dtype=np.float32),
                                indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=2)
    grid = torch.tensor(grid, dtype=torch.float32).to(args.device)
    # 4) Prepare metrics file
    metrics_path = os.path.join(args.outdir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["index", "rel_l2_Fx", "rel_l2_Fy"])

        # 5) Loop over indices
        for idx in range(args.start, args.end + 1):
            print(f"\n[Info] Processing sample #{idx}")
            try:
                traj_raw, Fx_true_raw, Fy_true_raw = load_test_sample(args.test_data, idx)
            except IndexError as e:
                print(f"[Skip] {e}")
                continue

            # Normalize input
            inputs = normalize_input(traj_raw, stats)  # (1, T, Nx, Ny)

            # Forward
            with torch.no_grad():
                x = inputs.to(args.device)
                pred = model(x,grid)  # expect (B, Nx, Ny, 2) or (B, 2, Nx, Ny)
                if pred.ndim == 4 and pred.shape[1] == 2:
                    pred = pred.permute(0, 2, 3, 1).contiguous()

            # Denormalize prediction
            Fx_pred_raw, Fy_pred_raw = denormalize_output(pred, stats)

            # Denormalize GT via explicit math using TRAIN stats (symmetry & ensuring same transform)
            F_mean = stats["F_mean"].squeeze(0)   # (2,Nx,Ny)
            F_std  = stats["F_std"].squeeze(0)    # (2,Nx,Ny)

            Fx_true_scaled = Fx_true_raw * F_SCALE
            Fy_true_scaled = Fy_true_raw * F_SCALE

            Fx_true_raw_dn = ((Fx_true_scaled - F_mean[0]) / F_std[0] * F_std[0] + F_mean[0]) / F_SCALE
            Fy_true_raw_dn = ((Fy_true_scaled - F_mean[1]) / F_std[1] * F_std[1] + F_mean[1]) / F_SCALE

            # Save per-sample npy (denormalized, original units)
            npy_path = os.path.join(args.outdir, f"sample_{idx:04d}_predictions.npy")
            np.save(npy_path, {
                "index": idx,
                "Fx_pred": Fx_pred_raw.astype(np.float32),
                "Fy_pred": Fy_pred_raw.astype(np.float32),
                "Fx_true": Fx_true_raw_dn.astype(np.float32),
                "Fy_true": Fy_true_raw_dn.astype(np.float32),
            })
            print(f"[OK] Saved predictions to: {npy_path}")

            # Save figures (Fx and Fy), with symmetric color scales
            fx_fig_path = os.path.join(args.outdir, f"sample_{idx:04d}_Fx.png")
            fy_fig_path = os.path.join(args.outdir, f"sample_{idx:04d}_Fy.png")
            save_comparison_fig(Fx_true_raw_dn, Fx_pred_raw, "Fx (True)", "Fx (Pred)", fx_fig_path)
            print(f"[OK] Saved figure: {fx_fig_path}")
            save_comparison_fig(Fy_true_raw_dn, Fy_pred_raw, "Fy (True)", "Fy (Pred)", fy_fig_path)
            print(f"[OK] Saved figure: {fy_fig_path}")

            # Metrics
            rel_fx = rel_l2(Fx_pred_raw, Fx_true_raw_dn)
            rel_fy = rel_l2(Fy_pred_raw, Fy_true_raw_dn)
            writer.writerow([idx, rel_fx, rel_fy])
            print(f"[Metrics] index={idx}  rel_l2_Fx={rel_fx:.6f}  rel_l2_Fy={rel_fy:.6f}")

    print("\n[Done] Batch evaluation complete.")
    print(f"[Info] Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
