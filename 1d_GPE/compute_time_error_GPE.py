#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==============================
# 路径配置（按需修改）
# ==============================
TRAIN_FILE = "/home/ubuntu/unlabel_PDE_official/1dGPE/training_data_GPE.npy"
TEST_FILE  = "/home/ubuntu/unlabel_PDE_official/1dGPE/test_data_GPE.npy"

CKPT_NIO   = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_GPE_nio/model_checkpoint_best_0.040607.pt"
CKPT_FNO   = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_GPE_fno/model_checkpoint_best_0.028811.pt"
CKPT_UNET  = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_GPE_unet/model_checkpoint_best_0.027363.pt"

OUT_ROOT   = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE"
os.makedirs(OUT_ROOT, exist_ok=True)

# ==============================
# 导入模型
# ==============================
from NIOModules import NIOFP_schrodinger, NIOFP_FNO, PermInvUNet_attn1D_bag_GPE

# ==============================
# 数据与缩放（与训练严格一致：只除以max）
# ==============================
def load_data_np(path):
    d = np.load(path, allow_pickle=True).item()
    for k in ["y", "V", "g", "kappa"]:
        if k not in d:
            raise KeyError(f"{path} 缺少键: {k}")
    return d

def compute_train_scalers(train_dict):
    y_max     = train_dict["y"].max() / 3.0
    V_max     = train_dict["V"].max() / 3.0
    g_max     = train_dict["g"].max()
    kappa_max = train_dict["kappa"].max()
    return {"y_max": y_max, "V_max": V_max, "g_max": g_max, "kappa_max": kappa_max}

def normalize_with_train_scalers(d, s):
    return {
        "y":     d["y"]     / s["y_max"],
        "V":     d["V"]     / s["V_max"],
        "g":     d["g"]     / s["g_max"],
        "kappa": d["kappa"] / s["kappa_max"],
    }

# ==============================
# 三个模型构建 & 加载
# ==============================
def build_model_nio(device):
    input_dimensions_trunk = 1
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 20
    modes = 40
    output_dim = 1
    model = NIOFP_schrodinger(
        input_dimensions_trunk, n_hidden_layers, neurons, n_basis,
        fno_layers, width, modes, output_dim, device
    ).to(device)
    ckpt = torch.load(CKPT_NIO, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()
    return model

def build_model_fno(device):
    fno_layers = 3
    width = 20
    modes = 40
    output_dim = 1
    model = NIOFP_FNO(fno_layers, width, modes, output_dim, device).to(device)
    ckpt = torch.load(CKPT_FNO, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()
    return model

def build_model_unet(device):
    # 与训练时一致
    model = PermInvUNet_attn1D_bag_GPE(
        in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=128, device=device, width=20, modes=40
    ).to(device)
    ckpt = torch.load(CKPT_UNET, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()
    return model

# ==============================
# GPE 求解器（与您给出的保持一致）
# ==============================
def get_initial_condition(ic, x):
    if ic == 1:
        return np.exp(-x**2/10)
    elif ic == 2:
        return 2 * np.sin(x) / (np.exp(x) + np.exp(-x))
    elif ic == 3:
        return 2 * np.cos(x) / (np.exp(x) + np.exp(-x))
    else:
        raise ValueError("初值索引必须为 1, 2 或 3")

def step_linear(psi, dt, k):
    psi_hat = np.fft.fft(psi)
    psi_hat = np.exp(-1j * dt * 0.5 * (k**2)) * psi_hat
    return np.fft.ifft(psi_hat)

def step_nonlinear(psi, dt, V, g, kappa):
    phase = np.exp(-1j * dt * (V + g * np.abs(psi)**2 + kappa * np.abs(psi)**4))
    return phase * psi

def step_strang(psi, dt, k, V, g, kappa):
    psi = step_nonlinear(psi, dt/2, V, g, kappa)
    psi = step_linear(psi, dt, k)
    psi = step_nonlinear(psi, dt/2, V, g, kappa)
    return psi

def step_fourth_order(psi, dt, k, V, g, kappa):
    c  = 2 - 2**(1/3)
    a1 = 1.0 / c
    a2 = - 2**(1/3) / c
    b1 = 1.0 / c
    b2 = - 2**(1/3) / c
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)
    psi = step_linear(psi, a1*dt, k)
    psi = step_nonlinear(psi, b2*dt, V, g, kappa)
    psi = step_linear(psi, a2*dt, k)
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)
    psi = step_linear(psi, a2*dt, k)
    psi = step_nonlinear(psi, b2*dt, V, g, kappa)
    psi = step_linear(psi, a1*dt, k)
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)
    return psi

def solve_GPE_custom(init_func, x, dt, t_final, order, g, kappa, V):
    Nx = len(x)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    Nt = int(t_final/dt) + 1
    t = np.linspace(0, t_final, Nt)
    psi = init_func(x).astype(complex)
    psi_record = np.zeros((Nt, Nx), dtype=complex)
    psi_record[0, :] = psi
    for n in range(1, Nt):
        if order == 2:
            psi = step_strang(psi, dt, k, V, g, kappa)
        elif order == 4:
            psi = step_fourth_order(psi, dt, k, V, g, kappa)
        else:
            raise ValueError("仅支持 order=2 或 4")
        psi_record[n, :] = psi
    return t, psi_record

# ==============================
# 时间平均相对 L2 误差（你要的公式）
# ==============================
def time_averaged_L2_error(time_ref, rho_ref, time_pred, rho_pred, grid, eps=1e-12):
    """
    Err_L2 = 1/T ∫_0^T ( ||rho_pred(·,t) - rho_ref(·,t)||_2 / ||rho_ref(·,t)||_2 ) dt
    """
    if rho_ref.shape != rho_pred.shape:
        raise ValueError(
            f"rho_ref shape {rho_ref.shape} != rho_pred shape {rho_pred.shape}, "
            "请检查时间步数和空间网格是否一致"
        )

    # 处理 grid：支持 (Nx,) 或 (1, Nx)
    if isinstance(grid, (list, tuple)):
        if len(grid) != 1:
            raise ValueError(f"不支持高维 grid: len(grid) = {len(grid)}")
        x = np.asarray(grid[0])
    else:
        grid = np.asarray(grid)
        if grid.ndim == 1:
            x = grid
        elif grid.ndim == 2 and grid.shape[0] == 1:
            x = grid[0]
        else:
            raise ValueError(f'不支持的 grid 形状: {grid.shape}，期望 (Nx,) 或 (1, Nx)')

    # 空间积分（先平方再积分）
    sq_diff = (rho_pred - rho_ref) ** 2
    sq_ref  = (rho_ref) ** 2
    diff_L2_sq = np.trapz(sq_diff, x=x, axis=1)
    ref_L2_sq  = np.trapz(sq_ref,  x=x, axis=1)

    diff_L2 = np.sqrt(np.maximum(diff_L2_sq, 0.0))
    ref_L2  = np.sqrt(np.maximum(ref_L2_sq,  0.0))
    rel_L2_t = diff_L2 / (ref_L2 + eps)

    if not np.allclose(time_ref, time_pred):
        raise ValueError("time_ref 与 time_pred 不一致，请检查 GPE 调用参数")

    t = time_ref
    dt = np.diff(t)
    time_integral = np.sum(0.5 * (rel_L2_t[:-1] + rel_L2_t[1:]) * dt)
    T = t[-1] - t[0]
    return float(time_integral / T)

# ==============================
# 主流程
# ==============================
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 加载数据 & 缩放因子
    train_raw = load_data_np(TRAIN_FILE)
    test_raw  = load_data_np(TEST_FILE)
    scalers   = compute_train_scalers(train_raw)
    test_norm = normalize_with_train_scalers(test_raw, scalers)

    # 基本尺寸
    num_total = test_norm["y"].shape[0]
    T_len     = test_norm["y"].shape[1]
    Nx        = test_norm["y"].shape[2]
    assert Nx == 128, "当前脚本假定 Nx=128（与训练一致）"

    # 2) 随机抽样
    num_samples = min(args.num_samples, num_total)
    if args.seed is not None:
        np.random.seed(args.seed)
    indices = np.random.choice(num_total, size=num_samples, replace=False)
    print(f"Randomly picked {num_samples} samples. First few indices: {indices[:10]}")

    # 3) 模型
    nio  = build_model_nio(device)
    fno  = build_model_fno(device)
    unet = build_model_unet(device)
    grid_norm = torch.linspace(0.0, 1.0, Nx).unsqueeze(-1).to(device)

    # 4) GPE 求解参数（与你生成代码一致）
    x = np.linspace(-10, 10, Nx)
    order   = 2
    dt      = 0.005
    t_final = 5.0
    init_ic = 2  # 初值统一使用 2 号

    # 5) 误差容器
    errs = {"nio": [], "fno": [], "unet": []}

    # 6) 循环评估
    for idx in indices:
        # --- 准备输入：归一化轨道（与训练一致，仅除最大值）
        traj_norm = test_norm["y"][idx]   # (T, Nx)
        inp = torch.tensor(traj_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Nx)

        # 真值 V（反归一化）
        true_V = test_norm["V"][idx] * scalers["V_max"]          # (Nx,)
        # g, kappa 用原始物理尺度（来自 raw）
        g_true     = float(np.atleast_1d(test_raw["g"][idx])[0])
        kappa_true = float(np.atleast_1d(test_raw["kappa"][idx])[0])

        # --- Reference 轨道（真值势）
        t_ref, psi_ref = solve_GPE_custom(lambda xx: get_initial_condition(init_ic, xx),
                                          x, dt, t_final, order, g_true, kappa_true, true_V)
        rho_ref = np.abs(psi_ref)  # (Nt, Nx)

        # --- NIO 预测 V
        with torch.no_grad():
            pred_nio = nio(inp, grid_norm).squeeze(0)  # (Nx, C) 或 (Nx,)
        if pred_nio.dim() == 2:
            pred_V_nio = pred_nio[:, 0].detach().cpu().numpy()
        else:
            pred_V_nio = pred_nio.detach().cpu().numpy()
        pred_V_nio = pred_V_nio * scalers["V_max"]  # 反归一化

        t_nio, psi_nio = solve_GPE_custom(lambda xx: get_initial_condition(init_ic, xx),
                                          x, dt, t_final, order, g_true, kappa_true, pred_V_nio)
        rho_nio = np.abs(psi_nio)
        err_nio = time_averaged_L2_error(t_ref, rho_ref, t_nio, rho_nio, x)
        errs["nio"].append(err_nio)

        # --- FNO 预测 V
        with torch.no_grad():
            pred_fno = fno(inp, grid_norm).squeeze(0)
        if pred_fno.dim() == 2:
            pred_V_fno = pred_fno[:, 0].detach().cpu().numpy()
        else:
            pred_V_fno = pred_fno.detach().cpu().numpy()
        pred_V_fno = pred_V_fno * scalers["V_max"]

        t_fno, psi_fno = solve_GPE_custom(lambda xx: get_initial_condition(init_ic, xx),
                                          x, dt, t_final, order, g_true, kappa_true, pred_V_fno)
        rho_fno = np.abs(psi_fno)
        err_fno = time_averaged_L2_error(t_ref, rho_ref, t_fno, rho_fno, x)
        errs["fno"].append(err_fno)

        # --- UNet 预测 V
        with torch.no_grad():
            pred_unet = unet(inp).squeeze(0)
        if pred_unet.dim() == 2:
            pred_V_unet = pred_unet[:, 0].detach().cpu().numpy()
        else:
            pred_V_unet = pred_unet.detach().cpu().numpy()
        pred_V_unet = pred_V_unet * scalers["V_max"]

        t_unet, psi_unet = solve_GPE_custom(lambda xx: get_initial_condition(init_ic, xx),
                                            x, dt, t_final, order, g_true, kappa_true, pred_V_unet)
        rho_unet = np.abs(psi_unet)
        err_unet = time_averaged_L2_error(t_ref, rho_ref, t_unet, rho_unet, x)
        errs["unet"].append(err_unet)

        # 可选：保存这一样本的 V 与误差
        for name, Vhat in [("nio", pred_V_nio), ("fno", pred_V_fno), ("unet", pred_V_unet)]:
            out_dir = os.path.join(OUT_ROOT, name)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"sample_{idx}_V_and_err.npy"),
                    {"x": x, "true_V": true_V, "pred_V": Vhat,
                     "g": g_true, "kappa": kappa_true,
                     "Err_L2_rel": errs[name][-1]})

        print(f"[idx {idx}]  Err_L2_rel  NIO={errs['nio'][-1]:.4e} | FNO={errs['fno'][-1]:.4e} | UNet={errs['unet'][-1]:.4e}")

    # 7) 汇总统计并保存
    for name in ["nio", "fno", "unet"]:
        arr = np.array(errs[name], dtype=float)
        mean_v = float(arr.mean())
        std_v  = float(arr.std())
        np.save(os.path.join(OUT_ROOT, f"ErrL2_relative_{name}_Nsamples_{num_samples}.npy"), arr)
        print(f"\n[{name.upper()}] over {num_samples} samples:")
        print(f"  mean Err_L2_rel = {mean_v:.4e}")
        print(f"  std  Err_L2_rel = {std_v:.4e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="随机选择测试样本数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()
    main(args)
