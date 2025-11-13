#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation of a trained UNet model on the 2D drift-diffusion test set.

- Uses EXACT same normalization as training (scales + train-set stats).
- Robust checkpoint loading (handles 'module.' prefix and various formats).
- Loops from --start to --end (inclusive), saving per-sample npy and figures.
- Writes a metrics.csv with per-sample relative L2 errors for drift/diffusion.

Author: you :)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv

from NIOModules import NIOFP2D, NIOFP2D_FNO, NIOFP2D_FNO_attn # ensure module is importable

# ------------------------------- Constants ------------------------------- #
TRAJ_SCALE = 1e10      # trajectories multiplier used during training
DRIFT_SCALE = 1e21     # training's "potential" -> drift
DIFFUSION_SCALE = 1e6  # training's "drag" -> diffusion

DEFAULT_TRAIN = "/home/ubuntu/unlabelPDE_official/2d_diffusion/dataset_2D_drift_diffusion.npz"
DEFAULT_TEST  = "/home/ubuntu/unlabelPDE_official/2d_diffusion/test_dataset_2D_drift_diffusion.npz"
DEFAULT_CKPT  = "/home/ubuntu/unlabelPDE_official/2d_diffusion/result_2d_nio/model_checkpoint_best_0.115960.pt"
DEFAULT_OUTDIR = "result_fig/nio"

# ------------------------------- Utils ---------------------------------- #
def compute_train_stats(train_npz_path):
    """Compute mean/std from TRAIN file with the SAME scaling/axes as training."""
    data = np.load(train_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32) * TRAJ_SCALE  # (M, 100, Nx, Ny)
    drift       = np.array(data["potential"],   dtype=np.float32) * DRIFT_SCALE   # (M, Nx, Ny)
    diffusion   = np.array(data["drag"],        dtype=np.float32) * DIFFUSION_SCALE

    traj_mean = trajectories.mean(axis=(0, 1), keepdims=True)
    traj_std  = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

    drift_mean = drift.mean(axis=0, keepdims=True)
    drift_std  = drift.std(axis=0, keepdims=True) + 1e-8

    diff_mean  = diffusion.mean(axis=0, keepdims=True)
    diff_std   = diffusion.std(axis=0, keepdims=True) + 1e-8

    return {
        "traj_mean": traj_mean, "traj_std": traj_std,
        "drift_mean": drift_mean, "drift_std": drift_std,
        "diff_mean": diff_mean, "diff_std": diff_std,
    }

def load_test_sample(test_npz_path, index):
    """Load raw ORIGINAL-units arrays for a given test index."""
    data = np.load(test_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32)  # (M, 100, Nx, Ny)
    drift = np.array(data["potential"], dtype=np.float32)            # (M, Nx, Ny)
    diffusion = np.array(data["drag"], dtype=np.float32)             # (M, Nx, Ny)
    if index < 0 or index >= trajectories.shape[0]:
        raise IndexError(f"Index {index} out of range [0, {trajectories.shape[0]-1}]")
    return trajectories[index], drift[index], diffusion[index]

def normalize_input(traj_raw, stats):
    """Normalize one test trajectory using TRAIN stats & scaling; returns (1, 100, Nx, Ny) tensor."""
    traj_scaled = traj_raw * TRAJ_SCALE
    traj_norm = (traj_scaled - stats["traj_mean"].squeeze(0)) / stats["traj_std"].squeeze(0)
    return torch.tensor(traj_norm[None, ...], dtype=torch.float32)

def denormalize_output(pred_norm, stats):
    """
    Denormalize model outputs to ORIGINAL units.
    Accepts (B, Nx, Ny, 2) or (B, 2, Nx, Ny); returns (drift_pred, diffusion_pred) as (Nx, Ny) numpy.
    """
    if pred_norm.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {pred_norm.shape}")

    if pred_norm.shape[-1] == 2:
        drift_norm = pred_norm[0, ..., 0].cpu().numpy()
        diff_norm  = pred_norm[0, ..., 1].cpu().numpy()
    elif pred_norm.shape[1] == 2:
        drift_norm = pred_norm[0, 0, ...].cpu().numpy()
        diff_norm  = pred_norm[0, 1, ...].cpu().numpy()
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {pred_norm.shape}")

    drift_mean = stats["drift_mean"].squeeze(0)
    drift_std  = stats["drift_std"].squeeze(0)
    diff_mean  = stats["diff_mean"].squeeze(0)
    diff_std   = stats["diff_std"].squeeze(0)

    drift_scaled = drift_norm * drift_std + drift_mean
    diff_scaled  = diff_norm  * diff_std  + diff_mean

    drift_orig = drift_scaled / DRIFT_SCALE
    diff_orig  = diff_scaled  / DIFFUSION_SCALE
    return drift_orig, diff_orig

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def load_checkpoint_robust(ckpt_path, device="cpu"):
    """
    Load a checkpoint robustly:
    - supports a raw state_dict or a dict with 'state_dict' key
    - strips leading 'module.' from keys if present
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

def rel_l2(a, b, eps=1e-12):
    """Relative L2 error ||a-b||_2 / ||b||_2 with small eps for stability."""
    num = np.linalg.norm((a - b).ravel(), 2)
    den = np.linalg.norm(b.ravel(), 2) + eps
    return float(num / den)

# ------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Batch evaluate UNet on 2D drift-diffusion test set.")
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN, help="Path to TRAIN npz (for stats).")
    parser.add_argument("--test_data",  type=str, default=DEFAULT_TEST,  help="Path to TEST npz (evaluation set).")
    parser.add_argument("--ckpt",       type=str, default=DEFAULT_CKPT,  help="Path to model checkpoint (.pt).")
    parser.add_argument("--outdir",     type=str, default=DEFAULT_OUTDIR,help="Output dir for figures and npy.")
    parser.add_argument("--start",      type=int, default=33,             help="Start index (inclusive).")
    parser.add_argument("--end",        type=int, default=64,            help="End index (inclusive).")
    parser.add_argument("--nx",         type=int, default=61,            help="Grid Nx.")
    parser.add_argument("--ny",         type=int, default=61,            help="Grid Ny.")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--strict",     action="store_true",
                        help="Use strict=True when loading state_dict (default False).")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # 1) TRAIN stats
    print(f"[Info] Loading TRAIN stats from: {args.train_data}")
    stats = compute_train_stats(args.train_data)

    # 2) Build model once
    print(f"[Info] Building model PermInvUNet_attn and loading checkpoint: {args.ckpt}")
    input_dimensions_trunk = 2
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 12
    modes = 32
    output_dim = 2
    nx = 61
    ny = 61
    model = NIOFP2D(input_dimensions_trunk,
                n_hidden_layers,
                neurons,
                n_basis,
                fno_layers,
                width,
                modes,
                output_dim)
    model = model.to(args.device)
    model.eval()
    Nx, Ny = 61, 61
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx, dtype=np.float32),
                                np.linspace(-1, 1, Ny, dtype=np.float32),
                                indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=2)
    grid = torch.tensor(grid, dtype=torch.float32).to(args.device)

    # 3) Load checkpoint robustly once
    sd = load_checkpoint_robust(args.ckpt, device="cpu")
    ret = model.load_state_dict(sd, strict=args.strict)
    if ret is not None and hasattr(ret, "missing_keys") and (ret.missing_keys or ret.unexpected_keys):
        print("[Warn] Incompatible keys when loading:")
        if ret.missing_keys:
            print("  Missing keys:", ret.missing_keys)
        if ret.unexpected_keys:
            print("  Unexpected keys:", ret.unexpected_keys)

    # 4) Prepare metrics file
    metrics_path = os.path.join(args.outdir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["index", "rel_l2_drift", "rel_l2_diffusion"])

        # 5) Loop over indices
        for idx in range(args.start, args.end + 1):
            print(f"\n[Info] Processing sample #{idx}")
            try:
                traj_raw, drift_true_raw, diff_true_raw = load_test_sample(args.test_data, idx)
            except IndexError as e:
                print(f"[Skip] {e}")
                continue

            # Normalize input & forward
            inputs = normalize_input(traj_raw, stats)  # (1, 100, Nx, Ny)
            with torch.no_grad():
                x = inputs.to(args.device)
                pred = model(x,grid)  # (B, 2, Nx, Ny) or (B, Nx, Ny, 2)
                if pred.ndim == 4 and pred.shape[1] == 2:
                    pred = pred.permute(0, 2, 3, 1).contiguous()

            # Denormalize pred
            drift_pred_raw, diff_pred_raw = denormalize_output(pred, stats)

            # Denormalize GT via explicit math using TRAIN stats (symmetry)
            drift_true_scaled = drift_true_raw * DRIFT_SCALE
            diff_true_scaled  = diff_true_raw  * DIFFUSION_SCALE
            drift_true_raw_dn = ((drift_true_scaled - stats["drift_mean"].squeeze(0)) / stats["drift_std"].squeeze(0)
                                * stats["drift_std"].squeeze(0) + stats["drift_mean"].squeeze(0)) / DRIFT_SCALE
            diff_true_raw_dn  = ((diff_true_scaled  - stats["diff_mean"].squeeze(0))  / stats["diff_std"].squeeze(0)
                                * stats["diff_std"].squeeze(0)  + stats["diff_mean"].squeeze(0))  / DIFFUSION_SCALE

            # Save npy per sample
            npy_path = os.path.join(args.outdir, f"sample_{idx:04d}_predictions.npy")
            np.save(npy_path, {
                "index": idx,
                "drift_pred": drift_pred_raw.astype(np.float32),
                "diffusion_pred": diff_pred_raw.astype(np.float32),
                "drift_true": drift_true_raw_dn.astype(np.float32),
                "diffusion_true": diff_true_raw_dn.astype(np.float32),
            })
            print(f"[OK] Saved predictions to: {npy_path}")

            # Figures per sample
            # Drift
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im0 = axes[0].imshow(drift_true_raw_dn, origin="lower")
            axes[0].set_title("Drift (True)")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(drift_pred_raw, origin="lower")
            axes[1].set_title("Drift (Pred)")
            plt.colorbar(im1, ax=axes[1])

            fig.suptitle(f"Sample #{idx} — Drift (True vs Pred)")
            drift_fig_path = os.path.join(args.outdir, f"sample_{idx:04d}_drift.png")
            plt.tight_layout()
            plt.savefig(drift_fig_path, dpi=150)
            plt.close()
            print(f"[OK] Saved figure: {drift_fig_path}")

            # Diffusion
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im0 = axes[0].imshow(diff_true_raw_dn, origin="lower")
            axes[0].set_title("Diffusion (True)")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(diff_pred_raw, origin="lower")
            axes[1].set_title("Diffusion (Pred)")
            plt.colorbar(im1, ax=axes[1])

            fig.suptitle(f"Sample #{idx} — Diffusion (True vs Pred)")
            diff_fig_path = os.path.join(args.outdir, f"sample_{idx:04d}_diffusion.png")
            plt.tight_layout()
            plt.savefig(diff_fig_path, dpi=150)
            plt.close()
            print(f"[OK] Saved figure: {diff_fig_path}")

            # Metrics (relative L2)
            rel_drift = rel_l2(drift_pred_raw, drift_true_raw_dn)
            rel_diff  = rel_l2(diff_pred_raw,  diff_true_raw_dn)
            writer.writerow([idx, rel_drift, rel_diff])
            print(f"[Metrics] index={idx}  rel_l2_drift={rel_drift:.6f}  rel_l2_diffusion={rel_diff:.6f}")

    print("\n[Done] Batch evaluation complete.")
    print(f"[Info] Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
