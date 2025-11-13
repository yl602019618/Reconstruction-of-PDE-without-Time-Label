#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate NIO / NIO-FNO / UNet on a batch of test samples, propagate density with
a Fokker–Planck solver, and compute time-averaged relative L2 errors of density.

Key points:
- EXACTLY the same normalization as training:
  trajectories * 1e10, F * 1e12, then standardize with TRAIN stats.
- Robust checkpoint loading (handles 'module.' prefix and raw dict/state_dict).
- Support three models: UNet, NIO, NIO-FNO (NIOModules must be importable).
- For each test index:
    * infer Fx,Fy for each model (denormalized back to original units)
    * build reference density trajectory using TRUE force from test set
    * propagate density for each predicted force
    * compute time-averaged relative L2 error wrt the reference density
    * save npy, optional figures, and write a metrics_all.csv row per model

Usage (example):
python evaluate_all_models.py \
  --train_data /home/ubuntu/unlabel_PDE_official/2dforce/dataset_2D_drift.npz \
  --test_data  /home/ubuntu/unlabel_PDE_official/2dforce/test_dataset_2D_drift.npz \
  --ckpt_unet  /home/ubuntu/unlabel_PDE_official/2dforce/result_unet/model_checkpoint_best_0.044678.pt \
  --ckpt_nio   /home/ubuntu/unlabel_PDE_official/2dforce/result_2d_nio/model_checkpoint_best_0.049776.pt \
  --ckpt_fno   /home/ubuntu/unlabel_PDE_official/2dforce/result_2d_nio/model_checkpoint_best_0.049776.pt \
  --start 0 --end 99 \
  --outdir result_fig \
  --save_fig
"""

import os
import csv
import argparse
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt

# import your modules (must be available in PYTHONPATH)
from NIOModules import PermInvUNet_attn, NIOFP2D, NIOFP2D_FNO

# Fokker-Planck solver and utils (as in your env)
from fplanck import fokker_planck, boundary, gaussian_pdf
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
# ------------------------------- Constants ------------------------------- #
TRAJ_SCALE = 1e10      # trajectories multiplier used during training
F_SCALE    = 1e12      # F multiplier (for both Fx and Fy) used during training

DEFAULT_TRAIN  = "/home/ubuntu/unlabel_PDE_official/2dforce/dataset_2D_drift.npz"
DEFAULT_TEST   = "/home/ubuntu/unlabel_PDE_official/2dforce/test_dataset_2D_drift.npz"
DEFAULT_UNET   = "/home/ubuntu/unlabel_PDE_official/2dforce/result_unet/model_checkpoint_best_0.044678.pt"
DEFAULT_NIO    = "/home/ubuntu/unlabel_PDE_official/2dforce/result_2d_nio/model_checkpoint_best_0.049776.pt"
DEFAULT_FNO    = "/home/ubuntu/unlabel_PDE_official/2dforce/result_2d_fno/model_checkpoint_best_0.067350.pt"

# ------------------------------- Utils ---------------------------------- #
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_train_stats(train_npz_path: str):
    """
    Compute mean/std from TRAIN file with the SAME scaling/axes as training.
    Returns dict with traj_mean/std (shape (1,1,Nx,Ny)) and F_mean/std (shape (1,2,Nx,Ny)).
    """
    data = np.load(train_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32) * TRAJ_SCALE  # (M,T,Nx,Ny)
    F            = np.array(data["F"],           dtype=np.float32) * F_SCALE      # (M,2,Nx,Ny)

    traj_mean = trajectories.mean(axis=(0,1), keepdims=True)   # (1,1,Nx,Ny)
    traj_std  = trajectories.std(axis=(0,1), keepdims=True) + 1e-8
    F_mean    = F.mean(axis=0, keepdims=True)                  # (1,2,Nx,Ny)
    F_std     = F.std(axis=0, keepdims=True) + 1e-8            # (1,2,Nx,Ny)
    return {"traj_mean": traj_mean, "traj_std": traj_std, "F_mean": F_mean, "F_std": F_std}

def load_test_sample(test_npz_path: str, index: int):
    """
    Load one sample from TEST file (ORIGINAL units).
    Returns: traj_raw (T,Nx,Ny), Fx_raw (Nx,Ny), Fy_raw (Nx,Ny)
    """
    data = np.load(test_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32)  # (M,T,Nx,Ny)
    F = np.array(data["F"], dtype=np.float32)                        # (M,2,Nx,Ny)
    if index < 0 or index >= trajectories.shape[0]:
        raise IndexError(f"Index {index} out of range [0, {trajectories.shape[0]-1}]")
    traj_raw = trajectories[index]
    Fx_raw   = F[index, 0]
    Fy_raw   = F[index, 1]
    return traj_raw, Fx_raw, Fy_raw

def normalize_input(traj_raw: np.ndarray, stats: dict):
    """
    Normalize one trajectory using TRAIN stats & scaling; return tensor (1,T,Nx,Ny).
    """
    traj_scaled = traj_raw * TRAJ_SCALE
    traj_norm = (traj_scaled - stats["traj_mean"].squeeze(0)) / stats["traj_std"].squeeze(0)
    return torch.tensor(traj_norm[None, ...], dtype=torch.float32)

def denormalize_output(pred_norm: torch.Tensor, stats: dict):
    """
    Denormalize model outputs to ORIGINAL units.
    Accepts pred_norm (B,Nx,Ny,2) or (B,2,Nx,Ny).
    Returns Fx_orig, Fy_orig as (Nx,Ny) numpy arrays.
    """
    if pred_norm.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {pred_norm.shape}")
    if pred_norm.shape[-1] == 2:
        Fx_norm = pred_norm[0, ..., 0].detach().cpu().numpy()
        Fy_norm = pred_norm[0, ..., 1].detach().cpu().numpy()
    elif pred_norm.shape[1] == 2:
        Fx_norm = pred_norm[0, 0, ...].detach().cpu().numpy()
        Fy_norm = pred_norm[0, 1, ...].detach().cpu().numpy()
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {pred_norm.shape}")

    F_mean = stats["F_mean"].squeeze(0)  # (2,Nx,Ny)
    F_std  = stats["F_std"].squeeze(0)   # (2,Nx,Ny)

    Fx_scaled = Fx_norm * F_std[0] + F_mean[0]
    Fy_scaled = Fy_norm * F_std[1] + F_mean[1]
    Fx_orig   = Fx_scaled / F_SCALE
    Fy_orig   = Fy_scaled / F_SCALE
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

def potential_from_data(grid, data):
    """
    Build an interpolator f(x,y) from a grid and a 2D array 'data'.
    'grid' can be (X,Y) stacked or (x_axis, y_axis) 1D arrays.
    Returns a callable f(x,y).
    """
    # Case 1: one-dimensional axes
    if isinstance(grid, (list, tuple)) and all(np.ndim(g) == 1 for g in grid):
        axes = list(grid)
        arr = np.asarray(data)
        for d, ax in enumerate(axes):
            if np.any(np.diff(ax) == 0):
                raise ValueError(f"Axis {d} has repeated points.")
            if ax[0] > ax[-1]:
                axes[d] = ax[::-1]
                arr = np.flip(arr, axis=d)
        f = RegularGridInterpolator(tuple(axes), arr, bounds_error=False, fill_value=None)
    else:
        # Case 2: stacked grid (2, Nx, Ny)
        G = np.asarray(grid)
        if G.shape[0] != 2:
            raise ValueError("Only 2D grids supported for stacked grid.")
        X, Y = G[0], G[1]
        arr = np.asarray(data)
        n1, n2 = arr.shape

        x_try, y_try = X[:,0], Y[0,:]
        if not (x_try.size == n1 and y_try.size == n2):
            x_try2, y_try2 = X[0,:], Y[:,0]
            if x_try2.size == n1 and y_try2.size == n2:
                x_try, y_try = x_try2, y_try2
            else:
                raise ValueError("Grid shape incompatible with data.")

        def ensure_monotonic(ax, axis):
            nonlocal arr
            if np.any(np.diff(ax) == 0):
                raise ValueError("Axis has repeated points.")
            if ax[0] > ax[-1]:
                ax = ax[::-1]
                arr = np.flip(arr, axis=axis)
            return ax

        x_axis = ensure_monotonic(x_try, axis=0)
        y_axis = ensure_monotonic(y_try, axis=1)
        f = RegularGridInterpolator((x_axis, y_axis), arr, bounds_error=False, fill_value=None)

    def wrapper(x, y):
        pts = np.stack([x, y], axis=-1) if np.ndim(x) == np.ndim(y) else np.stack(np.broadcast_arrays(x, y), axis=-1)
        return f(pts)
    return wrapper

def make_grid_tensor_for_model(nx, ny, device):
    """
    Create normalized coordinate tensor on [-1,1]^2 for NIO/NIO-FNO models.
    Returns torch.FloatTensor of shape (Nx,Ny,2) on 'device'.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1, 1, nx, dtype=np.float32),
        np.linspace(-1, 1, ny, dtype=np.float32),
        indexing="ij"
    )
    grid = np.stack([grid_x, grid_y], axis=2)
    return torch.tensor(grid, dtype=torch.float32, device=device)

def build_models(args):
    """
    Build three models with your hyperparams; return dict of model_name -> model.
    """
    device = args.device
    nx, ny = args.nx, args.ny

    # UNet
    unet = PermInvUNet_attn(in_ch=1, out_ch=2, base_ch=1, depth=5, input_size=(nx, ny)).to(device)
    unet.eval()

    # NIO baseline
    nio = NIOFP2D(
        input_dimensions_trunk=2,
        n_hidden_layers=3,
        neurons=100,
        n_basis=25,
        fno_layers=3,
        width=12,
        modes=32,
        output_dim=2,
    ).to(device)
    nio.eval()

    # NIO-FNO
    nio_fno = NIOFP2D_FNO(
        input_dimensions_trunk=2,
        n_hidden_layers=3,
        neurons=100,
        n_basis=25,
        fno_layers=3,
        width=12,
        modes=32,
        output_dim=2,
    ).to(device)
    nio_fno.eval()

    return {"unet": unet, "nio": nio, "nio_fno": nio_fno}

def load_all_checkpoints(models, args):
    """
    Load ckpts for three models (robust).
    """
    name2ckpt = {"unet": args.ckpt_unet, "nio": args.ckpt_nio, "nio_fno": args.ckpt_fno}
    for name, model in models.items():
        sd = load_checkpoint_robust(name2ckpt[name], device="cpu")
        ret = model.load_state_dict(sd, strict=args.strict)
        if ret is not None and hasattr(ret, "missing_keys") and (ret.missing_keys or ret.unexpected_keys):
            print(f"[Warn] {name}: missing={ret.missing_keys}, unexpected={ret.unexpected_keys}")

def build_fokker_planck(args, force_field_func):
    """
    Construct a Fokker-Planck simulator with given force function.
    Uses the same default physical params & grid settings as in the example code.
    """
    nm = 1e-9
    viscosity = args.viscosity
    radius = args.radius_nm * nm
    drag = 6 * np.pi * viscosity * radius
    extent = [args.extent_nm * nm, args.extent_nm * nm]
    resolution = args.resolution_nm * nm

    sim = fokker_planck(
        temperature=args.temperature,
        drag=drag,
        extent=extent,
        resolution=resolution,
        boundary=boundary.reflecting,
        force=force_field_func,
    )
    return sim

def force_from_array(grid, Fx_np, Fy_np):
    """
    Build vector force function F(x,y) from arrays sampled on 'grid'.
    """
    Fx_func = potential_from_data(grid, Fx_np)
    Fy_func = potential_from_data(grid, Fy_np)
    def F(x, y):
        Fx_val = Fx_func(x, y)
        Fy_val = Fy_func(x, y)
        return np.array([Fx_val, Fy_val])
    return F

def propagate_density_with_force(args, Fx, Fy):
    """
    Given Fx,Fy arrays on the simulator grid, propagate density and return (time, Pt).
    """
    # Build a temporary simulator first to get its grid
    # We create a dummy sim with a zero force to fetch grid shape, then rebuild with actual force.
    sim_dummy = build_fokker_planck(args, force_field_func=lambda x, y: np.array([0*x, 0*y]))
    grid = sim_dummy.grid
    F_func = force_from_array(grid, Fx, Fy)

    sim = build_fokker_planck(args, force_field_func=F_func)

    # Initial PDF: Gaussian (consistent with your example)
    nm = 1e-9
    pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)

    Nsteps = args.nsteps
    dt = args.dt
    time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
    return grid, time, Pt

def time_averaged_relative_l2(Pt_pred, Pt_ref, eps=1e-12):
    """
    Discrete-time version of time-averaged relative L2 error.
    Pt_* has shape (Nt, Nx, Ny).
    """
    assert Pt_pred.shape == Pt_ref.shape, "Pred/ref density shapes must match."
    Nt = Pt_pred.shape[0]
    errs = []
    for t in range(Nt):
        num = np.linalg.norm((Pt_pred[t] - Pt_ref[t]).ravel(), 2)
        den = np.linalg.norm(Pt_ref[t].ravel(), 2) + eps
        errs.append(num / den)
    return float(np.mean(errs))

def save_field_fig(pair_true, pair_pred, extent_xy, title_true, title_pred, out_path):
    vmax = float(np.max(np.abs([pair_true, pair_pred])))
    vmin = -vmax
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(pair_true, origin="lower", extent=extent_xy, vmin=vmin, vmax=vmax)
    axes[0].set_title(title_true)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pair_pred, origin="lower", extent=extent_xy, vmin=vmin, vmax=vmax)
    axes[1].set_title(title_pred)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_density_last_frame(Pt, extent_xy, title, out_path):
    plt.figure(figsize=(5,4))
    im = plt.imshow(Pt[-1], origin="lower", extent=extent_xy)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Batch evaluate (UNet/NIO/NIO-FNO) with density propagation and time-averaged relative L2 error.")
    # data & ckpt
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN)
    parser.add_argument("--test_data",  type=str, default=DEFAULT_TEST)
    parser.add_argument("--ckpt_unet",  type=str, default=DEFAULT_UNET)
    parser.add_argument("--ckpt_nio",   type=str, default=DEFAULT_NIO)
    parser.add_argument("--ckpt_fno",   type=str, default=DEFAULT_FNO)
    # eval range & grid
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=99)
    parser.add_argument("--nx",    type=int, default=80)
    parser.add_argument("--ny",    type=int, default=80)
    # device & strict
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    # output
    parser.add_argument("--outdir", type=str, default="result_fig")
    parser.add_argument("--save_fig", action="store_true", help="Save comparison figures and last-frame density.")
    # FP physics & numerics (match your example)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--viscosity",   type=float, default=8e-4)
    parser.add_argument("--radius_nm",   type=float, default=50.0)
    parser.add_argument("--extent_nm",   type=float, default=800.0)
    parser.add_argument("--resolution_nm", type=float, default=10.0)
    parser.add_argument("--nsteps",      type=int, default=500)
    parser.add_argument("--dt",          type=float, default=10e-3)
    args = parser.parse_args()

    # prepare dirs
    base_out = args.outdir
    out_unet = os.path.join(base_out, "unet")
    out_nio  = os.path.join(base_out, "nio")
    out_fno  = os.path.join(base_out, "fno")
    for p in [base_out, out_unet, out_nio, out_fno]:
        ensure_outdir(p)

    # 1) stats
    print(f"[Info] Loading TRAIN stats from: {args.train_data}")
    stats = compute_train_stats(args.train_data)

    # 2) build models + checkpoints
    print("[Info] Building models...")
    models = build_models(args)
    print("[Info] Loading checkpoints...")
    load_all_checkpoints(models, args)

    device = args.device
    nx, ny = args.nx, args.ny

    # Precompute model grid for NIO/NIO-FNO
    grid_tensor = make_grid_tensor_for_model(nx, ny, device=device)  # (Nx,Ny,2)

    # prepare metrics csv
    metrics_csv = os.path.join(base_out, "metrics_all.csv")
    write_header = not os.path.exists(metrics_csv)
    f_csv = open(metrics_csv, "a", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["index", "model", "rel_l2_Fx", "rel_l2_Fy", "ErrL2_density"])

    # loop over test indices
    for idx in tqdm(range(args.start, args.end + 1)):
        print(f"\n[Info] Processing index #{idx}")

        # load test sample (ORIGINAL units)
        try:
            traj_raw, Fx_true_raw, Fy_true_raw = load_test_sample(args.test_data, idx)
        except IndexError as e:
            print(f"[Skip] {e}")
            continue

        # ---- Normalize input trajectory
        x_in = normalize_input(traj_raw, stats).to(device)  # (1,T,Nx,Ny)

        # ---- True force arrays (for reference density) are already ORIGINAL units
        Fx_true = Fx_true_raw.astype(np.float32)
        Fy_true = Fy_true_raw.astype(np.float32)

        # ---- Build reference density trajectory once (using true force)
        #      We also grab sim.grid to reuse its extent for figures.
        grid_ref, time_ref, Pt_ref = propagate_density_with_force(args, Fx_true, Fy_true)

        # extent for imshow
        x_min, x_max = grid_ref[0].min(), grid_ref[0].max()
        y_min, y_max = grid_ref[1].min(), grid_ref[1].max()
        extent_xy = (x_min, x_max, y_min, y_max)

        # ---------------------- Evaluate each model ---------------------- #
        # Small helper to run model, denorm, save, propagate, compute errors
        def run_one_model(model_name: str, model, save_dir: str):
            with torch.no_grad():
                if model_name == "unet":
                    pred = model(x_in)   # expect (B,2,Nx,Ny) or (B,Nx,Ny,2)
                else:
                    # NIO / NIO-FNO need grid input
                    pred = model(x_in, grid_tensor)

                if pred.ndim == 4 and pred.shape[1] == 2:
                    pred = pred.permute(0,2,3,1).contiguous()  # to (B,Nx,Ny,2)

            # denormalize back to ORIGINAL units
            Fx_pred, Fy_pred = denormalize_output(pred, stats)

            # save per-sample npy
            # npy_path = os.path.join(save_dir, f"sample_{idx:04d}_predictions.npy")
            # np.save(npy_path, {
            #     "index": idx,
            #     "Fx_pred": Fx_pred.astype(np.float32),
            #     "Fy_pred": Fy_pred.astype(np.float32),
            #     "Fx_true": Fx_true.astype(np.float32),
            #     "Fy_true": Fy_true.astype(np.float32),
            # })
            # print(f"[OK] {model_name}: saved predictions to {npy_path}")

            # field-wise rel L2 (optional but useful)
            rel_fx = rel_l2(Fx_pred, Fx_true)
            rel_fy = rel_l2(Fy_pred, Fy_true)

            # propagate density with predicted force
            _, time_pred, Pt_pred = propagate_density_with_force(args, Fx_pred, Fy_pred)

            # time-averaged relative L2 error for density trajectories
            err_l2 = time_averaged_relative_l2(Pt_pred, Pt_ref)

            # write metrics row
            writer.writerow([idx, model_name, rel_fx, rel_fy, err_l2])
            print(f"[Metrics] idx={idx} model={model_name}  relL2(Fx)={rel_fx:.6f}  relL2(Fy)={rel_fy:.6f}  ErrL2(density)={err_l2:.6f}")

            # # optional figures
            # if args.save_fig:
            #     fx_fig = os.path.join(save_dir, f"sample_{idx:04d}_Fx.png")
            #     fy_fig = os.path.join(save_dir, f"sample_{idx:04d}_Fy.png")
            #     save_field_fig(Fx_true, Fx_pred, extent_xy, "Fx (True)", f"Fx (Pred) - {model_name}", fx_fig)
            #     save_field_fig(Fy_true, Fy_pred, extent_xy, "Fy (True)", f"Fy (Pred) - {model_name}", fy_fig)
            #     save_density_last_frame(Pt_pred, extent_xy, f"P_t Last ({model_name})", os.path.join(save_dir, f"Pt_{idx:04d}.png"))

            # # also save Pt full trajectory for further analysis
            # np.save(os.path.join(save_dir, f"Pt_{idx:04d}.npy"), Pt_pred)

        # run three models
        run_one_model("unet",   models["unet"],   out_unet)
        run_one_model("nio",    models["nio"],    out_nio)
        run_one_model("nio_fno",models["nio_fno"],out_fno)

        # Optional: also save reference density last frame (once per idx)
        # if args.save_fig:
        #     save_density_last_frame(Pt_ref, extent_xy, "P_t Last (reference)", os.path.join(base_out, f"Pt_ref_{idx:04d}.png"))
        # np.save(os.path.join(base_out, f"Pt_ref_{idx:04d}.npy"), Pt_ref)

    f_csv.close()
    print(f"\n[Done] All metrics written to: {metrics_csv}")

if __name__ == "__main__":
    main()
