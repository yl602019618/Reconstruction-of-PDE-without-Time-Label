#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====== 路径配置（与你单测脚本一致）======
TRAIN_FILE = "/home/ubuntu/unlabel_PDE_official/1dGPE/training_data_Schrodinger.npy"
TEST_FILE  = "/home/ubuntu/unlabel_PDE_official/1dGPE/test_data_Schrodinger.npy"

CKPT_NIO = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_schrodinger_nio/model_checkpoint_best_0.042415.pt"
CKPT_FNO = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_schrodinger_fno/model_checkpoint_best_0.048281.pt"
CKPT_UNET = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_schrodinger_unet/model_checkpoint_best_0.038187.pt"

BASE_OUT = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_eval"
os.makedirs(BASE_OUT, exist_ok=True)

# ====== 导入模型定义（与你单测脚本一致）======
from NIOModules import PermInvUNet_attn1D_bag
from NIOModules import NIOFP_schrodinger, NIOFP_FNO

# ============================================================
#  训练集缩放（严格照你的单测脚本）
# ============================================================

def load_data_np(path):
    data = np.load(path, allow_pickle=True).item()
    required = ["y", "V", "g", "kappa"]
    for k in required:
        if k not in data:
            raise KeyError(f"{path} 缺少键: {k}")
    return data

def compute_train_scalers(train_dict):
    """
    与你给出的推理代码一致：
      y_max = y.max() / 3
      V_max = V.max() / 3
      g_max = g.max()
      kappa_max = kappa.max()
    """
    y_max = train_dict["y"].max() / 3.0
    V_max = train_dict["V"].max() / 3.0
    g_max = train_dict["g"].max()
    kappa_max = train_dict["kappa"].max()
    scalers = {"y_max": y_max, "V_max": V_max, "g_max": g_max, "kappa_max": kappa_max}
    return scalers

def normalize_with_train_scalers(d, scalers):
    """
    与训练一致：只除以最大值，不减均值
    """
    y = d["y"] / scalers["y_max"]
    V = d["V"] / scalers["V_max"]
    g = d["g"] / scalers["g_max"]
    kappa = d["kappa"] / scalers["kappa_max"]
    return {"y": y, "V": V, "g": g, "kappa": kappa}

# ============================================================
#  模型构建 & 加载（与单测脚本一致）
# ============================================================

def build_model_nio(device):
    input_dimensions_trunk = 1
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 10
    modes = 30
    output_dim = 1
    model = NIOFP_schrodinger(
        input_dimensions_trunk,
        n_hidden_layers,
        neurons,
        n_basis,
        fno_layers,
        width,
        modes,
        output_dim,
        device
    ).to(device)
    return model

def build_model_fno(device):
    fno_layers = 3
    width = 10
    modes = 30
    output_dim = 1
    model = NIOFP_FNO(
        fno_layers,
        width,
        modes,
        output_dim,
        device
    ).to(device)
    return model

def build_model_unet(device):
    model = PermInvUNet_attn1D_bag(
        in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=128, device=device
    ).to(device)
    return model

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

# ============================================================
#  GPE（伪谱 + 分裂步）——复用并小幅封装你的实现
# ============================================================

def get_initial_condition(ic, x):
    if ic == 1:
        return np.exp(-x**2/10)
    elif ic == 2:
        return 2 * np.sin(x) / (np.exp(x) + np.exp(-x))
    elif ic == 3:
        return 2 * np.cos(x) / (np.exp(x) + np.exp(-x))
    else:
        raise ValueError("初值索引必须为 1, 2 或 3")

def sech(x):
    return 1/np.cosh(x)

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
    psi = step_nonlinear(psi, b1, V, g, kappa); psi = step_linear(psi, a1, k)
    psi = step_nonlinear(psi, b2, V, g, kappa); psi = step_linear(psi, a2, k)
    psi = step_nonlinear(psi, b1, V, g, kappa); psi = step_linear(psi, a2, k)
    psi = step_nonlinear(psi, b2, V, g, kappa); psi = step_linear(psi, a1, k)
    psi = step_nonlinear(psi, b1, V, g, kappa)
    return psi

def solve_GPE_custom(init_ic, V, g, kappa, Nx=128, dt=0.005, t_final=5.0, order=2):
    """
    在区间 x ∈ [-10,10]，时间 [0,t_final] 上，用给定 V(x), g, kappa 演化。
    返回:
      t: (Nt,)
      rho: (Nt, Nx) 密度 |psi|^2
    """
    assert V.ndim == 1, "V 必须是一维 (Nx,)"
    Nx = len(V)
    x = np.linspace(-10.0, 10.0, Nx)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    Nt = int(t_final/dt) + 1
    t = np.linspace(0.0, t_final, Nt)

    psi = get_initial_condition(init_ic, x).astype(complex)
    rho = np.zeros((Nt, Nx), dtype=float)
    rho[0] = (np.abs(psi)**2).real

    # 预乘 dt 系数，避免多次乘法
    if order == 2:
        for n in range(1, Nt):
            psi = step_strang(psi, dt, k, V, g, kappa)
            rho[n] = (np.abs(psi)**2).real
    elif order == 4:
        # 注意 step_fourth_order 内部的 a,b 系数现在写成“以 dt=1 计”，这里传入时乘 dt
        for n in range(1, Nt):
            psi = step_nonlinear(psi, (1.0/(2-2**(1/3)))*dt, V, g, kappa)
            psi = step_linear(psi,  (1.0/(2-2**(1/3)))*dt, k)
            psi = step_nonlinear(psi, (-2**(1/3)/(2-2**(1/3)))*dt, V, g, kappa)
            psi = step_linear(psi,  (-2**(1/3)/(2-2**(1/3)))*dt, k)
            psi = step_nonlinear(psi, (1.0/(2-2**(1/3)))*dt, V, g, kappa)
            psi = step_linear(psi,  (-2**(1/3)/(2-2**(1/3)))*dt, k)
            psi = step_nonlinear(psi, (-2**(1/3)/(2-2**(1/3)))*dt, V, g, kappa)
            psi = step_linear(psi,  (1.0/(2-2**(1/3)))*dt, k)
            psi = step_nonlinear(psi, (1.0/(2-2**(1/3)))*dt, V, g, kappa)
            rho[n] = (np.abs(psi)**2).real
    else:
        raise ValueError("仅支持 order=2 或 4")

    return t, x, rho

# ============================================================
#  误差（严格按题面公式：时间平均的 L2^2）
# ============================================================

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

    # 时间一致性检查 + 时间平均
    if not np.allclose(time_ref, time_pred):
        raise ValueError("time_ref 与 time_pred 不一致，请检查 GPE 调用参数")

    t = time_ref
    dt = np.diff(t)
    time_integral = np.sum(0.5 * (rel_L2_t[:-1] + rel_L2_t[1:]) * dt)
    T = t[-1] - t[0]
    return float(time_integral / T)


# ============================================================
#  主流程
# ============================================================

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 训练集缩放（严格与训练一致）
    train_dict_raw = load_data_np(TRAIN_FILE)
    scalers = compute_train_scalers(train_dict_raw)

    # 2) 测试集（原始 + 归一化）
    test_dict_raw = load_data_np(TEST_FILE)
    test_norm = normalize_with_train_scalers(test_dict_raw, scalers)

    num_total = test_norm["y"].shape[0]
    T_len, Nx = test_norm["y"].shape[1], test_norm["y"].shape[2]
    print(f"Test set: {num_total} samples, T={T_len}, Nx={Nx}")

    # 3) 三个模型
    model_nio  = load_checkpoint(build_model_nio(device), CKPT_NIO, device)
    model_fno  = load_checkpoint(build_model_fno(device), CKPT_FNO, device)
    model_unet = load_checkpoint(build_model_unet(device), CKPT_UNET, device)

    # NIO/FNO 需要的 [0,1] 归一化坐标网格
    grid_tensor = torch.linspace(0.0, 1.0, Nx).unsqueeze(-1).to(device)

    # 4) 采样样本索引
    rng = np.random.default_rng(42)
    num_eval = min(args.num_samples, num_total)
    if args.take_first:
        indices = np.arange(num_eval, dtype=int)
    else:
        indices = rng.choice(num_total, size=num_eval, replace=False)
    print(f"Evaluating {num_eval} samples, indices[0:10]={indices[:10]}")

    # 5) 评估 & 记录
    errs = {"nio": [], "fno": [], "unet": []}
    os.makedirs(os.path.join(BASE_OUT, "nio"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT, "fno"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT, "unet"), exist_ok=True)

    for idx in tqdm(indices, desc="Eval"):
        # ========== 准备输入 ==========
        # (T, Nx) 归一化轨道
        traj_norm = test_norm["y"][idx]  # 已按 y_max 归一化
        # 真值势（反归一化）
        V_true = test_norm["V"][idx] * scalers["V_max"]  # (Nx,)
        # 非线性系数（直接用原始值）
        g_true = float(test_dict_raw["g"][idx])
        kappa_true = float(test_dict_raw["kappa"][idx])

        # torch 输入
        inp = torch.tensor(traj_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Nx)

        # ========== 真值密度轨道 ==========
        t_ref, x_ref, rho_ref = solve_GPE_custom(
            init_ic=2, V=V_true, g=g_true, kappa=kappa_true,
            Nx=Nx, dt=args.dt, t_final=args.t_final, order=args.order
        )

        # ========== NIO ==========
        with torch.no_grad():
            pred_nio = model_nio(inp, grid_tensor).squeeze(0)  # (Nx,C)
        # NIO/FNO/UNet 单测里：取第 0 通道为 V 预测
        if pred_nio.ndim == 2:
            V_pred_nio = pred_nio[:, 0].detach().cpu().numpy()
        else:
            V_pred_nio = pred_nio.detach().cpu().numpy()
        V_pred_nio = V_pred_nio * scalers["V_max"]  # 反归一化
        # 演化
        t_nio, x_nio, rho_nio = solve_GPE_custom(
            init_ic=2, V=V_pred_nio, g=g_true, kappa=kappa_true,
            Nx=Nx, dt=args.dt, t_final=args.t_final, order=args.order
        )
        err_nio  = time_averaged_L2_error(t_ref,  rho_ref, t_nio,  rho_nio,  x_ref)
        errs["nio"].append(err_nio)
        np.save(os.path.join(BASE_OUT, "nio", f"V_pred_{idx}.npy"), V_pred_nio)

        # ========== FNO ==========
        with torch.no_grad():
            pred_fno = model_fno(inp, grid_tensor).squeeze(0)
        if pred_fno.ndim == 2:
            V_pred_fno = pred_fno[:, 0].detach().cpu().numpy()
        else:
            V_pred_fno = pred_fno.detach().cpu().numpy()
        V_pred_fno = V_pred_fno * scalers["V_max"]
        t_fno, x_fno, rho_fno = solve_GPE_custom(
            init_ic=2, V=V_pred_fno, g=g_true, kappa=kappa_true,
            Nx=Nx, dt=args.dt, t_final=args.t_final, order=args.order
        )
        err_fno  = time_averaged_L2_error(t_ref,  rho_ref, t_fno,  rho_fno,  x_ref)
        errs["fno"].append(err_fno)
        np.save(os.path.join(BASE_OUT, "fno", f"V_pred_{idx}.npy"), V_pred_fno)

        # ========== UNet ==========
        with torch.no_grad():
            pred_unet = model_unet(inp).squeeze(0)
        if pred_unet.ndim == 2:
            V_pred_unet = pred_unet[:, 0].detach().cpu().numpy()
        else:
            V_pred_unet = pred_unet.detach().cpu().numpy()
        V_pred_unet = V_pred_unet * scalers["V_max"]
        t_unet, x_unet, rho_unet = solve_GPE_custom(
            init_ic=2, V=V_pred_unet, g=g_true, kappa=kappa_true,
            Nx=Nx, dt=args.dt, t_final=args.t_final, order=args.order
        )
        err_unet = time_averaged_L2_error(t_ref,  rho_ref, t_unet, rho_unet, x_ref)
        errs["unet"].append(err_unet)
        np.save(os.path.join(BASE_OUT, "unet", f"V_pred_{idx}.npy"), V_pred_unet)

    # 6) 汇总与保存
    summary_lines = []
    for name in ["nio", "fno", "unet"]:
        arr = np.array(errs[name], dtype=np.float64)
        np.save(os.path.join(BASE_OUT, f"ErrL2_{name}_N{num_eval}.npy"), arr)
        mean_v = float(arr.mean()) if arr.size > 0 else float("nan")
        std_v  = float(arr.std())  if arr.size > 0 else float("nan")
        line = f"{name.upper():5s} | mean Err_L2 = {mean_v:.6e} | std = {std_v:.6e} | N = {arr.size}"
        print(line)
        summary_lines.append(line)

    with open(os.path.join(BASE_OUT, "summary.txt"), "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="从测试集抽取的样本数")
    parser.add_argument("--take_first", action="store_true", help="若给出则取前 num_samples 个样本，否则随机抽样")
    parser.add_argument("--dt", type=float, default=0.005, help="GPE 时间步长")
    parser.add_argument("--t_final", type=float, default=5.0, help="GPE 演化终止时间")
    parser.add_argument("--order", type=int, default=2, choices=[2,4], help="分裂步阶数")
    args = parser.parse_args()
    main(args)
