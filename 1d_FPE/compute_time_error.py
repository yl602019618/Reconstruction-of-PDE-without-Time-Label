import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from NIOModules import NIOFP, NIOFP_FNO, PermInvUNet_attn1D_bag
from fplanck import (
    fokker_planck,
    boundary,
    gaussian_pdf,
    combine,
    gaussian_potential,
    potential_from_data,
)

# ============================================================
#  路径与常数配置
# ============================================================

# 数据路径
TRAIN_FILE = "/home/ubuntu/unlabel_PDE_official/1d/dataset_1D_drift_diffusion.npz"
EVAL_FILE  = "/home/ubuntu/unlabel_PDE_official/1d/dataset_1D_drift_diffusion_evaluation.npz"

# 三个模型的 checkpoint
CKPT_NIO  = "/home/ubuntu/unlabel_PDE_official/1d/results_nio/model_checkpoint_best_0.281287.pt"
CKPT_FNO  = "/home/ubuntu/unlabel_PDE_official/1d/results_fno/model_checkpoint_best_0.342750.pt"
CKPT_UNET = "/home/ubuntu/unlabel_PDE_official/1d/results_unet_bag/model_checkpoint_best_0.240148.pt"

# 预测 potential / drag 存放目录
BASE_RESULT_DIR = "/home/ubuntu/unlabel_PDE_official/1d/result_fig"

# 与训练脚本一致的缩放系数
TRAJ_SCALE      = 1e5
POTENTIAL_SCALE = 1e20
DRAG_SCALE      = 1e5

# Fokker–Planck 物理参数
nm = 1e-9
viscosity = 8e-4
radius    = 50 * nm
drag_phys = 6 * np.pi * viscosity * radius   # 只用来构造 dummy potential 的 sim，真正的 drag 用数据或模型预测
temperature        = 300
extent             = 800 * nm
resolution         = 10 * nm
boundary_condition = boundary.reflecting
N_STEPS_FP         = 400
DT_FP              = 2e-3  # 与你原代码一致
INIT_WIDTH         = 50 * nm   # 初始 Gaussian 的宽度

# ============================================================
#  训练集统计量（与训练保持完全一致）
# ============================================================

def compute_train_stats(train_npz_path):
    """
    与你之前 eval 脚本中的 compute_train_stats 完全一致。
    注意：先乘缩放系数，再在指定维度上算 mean/std。
    """
    data = np.load(train_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32) * TRAJ_SCALE   # (M, 100, Nx)
    potential    = np.array(data["potential"],    dtype=np.float32) * POTENTIAL_SCALE  # (M, Nx)
    drag         = np.array(data["drag"],         dtype=np.float32) * DRAG_SCALE       # (M,)

    drag = drag[:, np.newaxis]  # (M,1)

    # 与训练完全一致的维度上做 mean/std
    traj_mean = trajectories.mean(axis=(0, 1), keepdims=True)  # (1,100,Nx)
    traj_std  = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

    pot_mean  = potential.mean(axis=0, keepdims=True)   # (1, Nx)
    pot_std   = potential.std(axis=0, keepdims=True) + 1e-8

    drag_mean = drag.mean(axis=0, keepdims=True)        # (1,1)
    drag_std  = drag.std(axis=0, keepdims=True) + 1e-8

    stats = {
        "traj_mean": traj_mean,
        "traj_std":  traj_std,
        "pot_mean":  pot_mean,
        "pot_std":   pot_std,
        "drag_mean": drag_mean,
        "drag_std":  drag_std
    }
    return stats

def normalize_traj_with_train_stats(traj_raw, stats):
    """
    traj_raw: (T, Nx) —— EVAL 数据集中原始物理量级的轨道
    与你原 eval 脚本中的 normalize_with_train_stats 完全一致。
    """
    traj_scaled = traj_raw * TRAJ_SCALE  # (T, Nx)
    traj_norm   = (traj_scaled - stats["traj_mean"].squeeze()) / stats["traj_std"].squeeze()
    return traj_norm.astype(np.float32)

def denormalize_outputs(pred_norm, stats):
    """
    pred_norm: (Nx, 2), 列 0 = normalized potential, 列 1 = normalized drag (逐点复制)
    与原 eval 脚本中的 denormalize_outputs 完全一致。
    """
    pot = pred_norm[:, 0] * stats["pot_std"].squeeze()  + stats["pot_mean"].squeeze()   # (Nx,)
    drg = pred_norm[:, 1] * stats["drag_std"].squeeze() + stats["drag_mean"].squeeze()  # (Nx,)

    pot = pot / POTENTIAL_SCALE
    drg = drg / DRAG_SCALE
    return pot, drg

# ============================================================
#  加载 EVAL 数据
# ============================================================

def load_eval_dataset(eval_npz_path):
    data = np.load(eval_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32)  # (M, 100, Nx)
    potential    = np.array(data["potential"],    dtype=np.float32)  # (M, Nx)
    drag         = np.array(data["drag"],         dtype=np.float32)  # (M,)
    return trajectories, potential, drag

# ============================================================
#  构造 grid（严格模仿你原来用 random_gaussian_params + combine 的方式）
#  只是为了拿到 sim.grid，保证与原脚本相同的空间离散
# ============================================================

def make_grid():
    # 随便造一个 potential，只是为了让 fokker_planck 生成 grid
    centers = np.array([0.0, 50 * nm, -50 * nm])
    widths  = np.array([40 * nm, 60 * nm, 50 * nm])
    As      = np.array([1e-20, 1.5e-20, 1.2e-20])

    U_dummy = combine(
        gaussian_potential(center=centers[0], width=widths[0], amplitude=As[0]),
        gaussian_potential(center=centers[1], width=widths[1], amplitude=As[1]),
        gaussian_potential(center=centers[2], width=widths[2], amplitude=As[2]),
    )

    sim_dummy = fokker_planck(
        temperature=temperature,
        drag=drag_phys,
        extent=extent,
        resolution=resolution,
        boundary=boundary_condition,
        potential=U_dummy,
    )
    return sim_dummy.grid  # 这就是后续 potential_from_data 使用的 grid

# ============================================================
#  三个模型的加载
# ============================================================

def load_model_nio(device):
    input_dimensions_trunk = 1
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 30
    modes = 15
    output_dim = 2

    model = NIOFP(
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

    ckpt = torch.load(CKPT_NIO, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def load_model_fno(device):
    fno_layers = 3
    width = 30
    modes = 15
    output_dim = 2

    model = NIOFP_FNO(
        fno_layers,
        width,
        modes,
        output_dim,
        device
    ).to(device)

    ckpt = torch.load(CKPT_FNO, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def load_model_unet(device):
    model = PermInvUNet_attn1D_bag(
        in_ch=1,
        out_ch=2,
        base_ch=1,
        depth=4,
        input_size=80,  # 与训练时一致
        device=device
    ).to(device)

    ckpt = torch.load(CKPT_UNET, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

# ============================================================
#  Fokker–Planck 演化 & time-averaged L² error
# ============================================================

def simulate_density_trajectory(U_np, drag_value, grid):
    """
    输入:
      U_np      : (Nx,) numpy array, potential on grid (原始物理量级)
      drag_value: scalar, drag 系数
      grid      : (Nx,) 空间网格
    输出:
      time: (Nt,) 时间点
      Pt  : (Nt, Nx) 演化密度
    """
    U_obj = potential_from_data(grid, U_np)

    sim = fokker_planck(
        temperature=temperature,
        drag=drag_value,
        extent=extent,
        resolution=resolution,
        boundary=boundary_condition,
        potential=U_obj
    )

    pdf = gaussian_pdf(center=(0.0 * nm), width=INIT_WIDTH)
    time, Pt = sim.propagate_interval(pdf, DT_FP, Nsteps=N_STEPS_FP)
    return time, Pt, sim.grid  # 返回 sim.grid 以防万一

def time_averaged_L2_error(time_ref, rho_ref, time_pred, rho_pred, grid, eps=1e-12):
    """
    实现公式：
        Err_L2 = 1/T ∫_0^T ( ||rho_pred(·,t) - rho_ref(·,t)||_2 / ||rho_ref(·,t)||_2 ) dt

    相对 L2 误差的时间平均。
    """
    # 简单 sanity check
    if rho_ref.shape != rho_pred.shape:
        raise ValueError(
            f"rho_ref shape {rho_ref.shape} != rho_pred shape {rho_pred.shape}, "
            "请检查时间步数和空间网格是否一致"
        )

    # ---- 处理 grid：支持 (Nx,) 和 (1, Nx) ----
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

    # ---- 空间方向：先算 |·|^2 再积分 ----
    # 差值平方
    sq_diff = (rho_pred - rho_ref) ** 2          # (Nt, Nx)
    sq_ref  = (rho_ref) ** 2                     # (Nt, Nx)

    # 对 x 积分：得到 L2 范数的“平方”
    diff_L2_sq = np.trapz(sq_diff, x=x, axis=1)  # (Nt,)
    ref_L2_sq  = np.trapz(sq_ref,  x=x, axis=1)  # (Nt,)

    # 取平方根得到 L2 范数
    diff_L2 = np.sqrt(np.maximum(diff_L2_sq, 0.0))   # 数值上避免负零
    ref_L2  = np.sqrt(np.maximum(ref_L2_sq,  0.0))

    # 每个时间点的相对 L2 误差： ||diff||_2 / ||ref||_2
    # 加上 eps 防止 ref_L2=0 导致除零
    rel_L2_t = diff_L2 / (ref_L2 + eps)   # (Nt,)

    # ---- 时间积分：对 rel_L2_t 做 time-average ----
    t = time_ref
    if not np.allclose(time_ref, time_pred):
        raise ValueError("time_ref 与 time_pred 不一致，请检查 Fokker–Planck 调用参数")

    dt = np.diff(t)
    # trapezoid rule: ∫ r(t) dt ≈ Σ 0.5 * (r_k + r_{k+1}) * Δt_k
    time_integral = np.sum(0.5 * (rel_L2_t[:-1] + rel_L2_t[1:]) * dt)
    T = t[-1] - t[0]
    Err_L2 = time_integral / T
    return Err_L2

# ============================================================
#  主流程：对多个 test 样本、三个模型计算 Err_L2
# ============================================================

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 训练集统计量（normalization 与训练完全一致）
    stats = compute_train_stats(TRAIN_FILE)
    print("Train stats computed.")

    # 2) 加载 test / evaluation 数据
    trajectories_eval, potential_eval, drag_eval = load_eval_dataset(EVAL_FILE)
    num_total_samples, T_len, Nx_data = trajectories_eval.shape
    print(f"Eval dataset: {num_total_samples} samples, T={T_len}, Nx={Nx_data}")

    # 3) 构造 grid（与之前 Fokker–Planck 脚本保持一致）
    grid = make_grid()
    Nx_grid = grid.shape[0]
    print(f"Spatial grid size from Fokker–Planck: {Nx_grid}")

    if Nx_grid != Nx_data:
        print("WARNING: Nx from dataset and grid size differ, "
              "这里假设 potential_from_data 会做插值。")

    # 4) 载入三个模型
    model_nio  = load_model_nio(device)
    model_fno  = load_model_fno(device)
    model_unet = load_model_unet(device)
    print("All models loaded.")

    # 归一化后的坐标 grid (0~1)，给 NIO / FNO 使用（与原 eval 脚本一致）
    grid_norm_tensor = torch.linspace(0.0, 1.0, Nx_data).unsqueeze(-1).to(device)

    # 5) 选择要评估的样本索引
    num_samples = min(args.num_samples, num_total_samples)
    np.random.seed(42)  # 固定随机性，可选
    sample_indices = np.random.choice(num_total_samples, size=num_samples, replace=False)
    print(f"Will evaluate {num_samples} randomly selected samples: indices = {sample_indices[:10]}{'...' if num_samples > 10 else ''}")

    # 结果字典
    err_L2_results = {
        "nio":  [],
        "fno":  [],
        "unet": []
    }

    # 确保输出目录存在
    for model_name in ["nio", "fno", "unet"]:
        os.makedirs(os.path.join(BASE_RESULT_DIR, model_name), exist_ok=True)

    # 6) 主循环：每个样本一次，内部对三个模型分别算 Err_L2
    for idx in tqdm(sample_indices):
        print(f"\n=== Sample {idx} ===")
        traj_raw = trajectories_eval[idx]      # (T, Nx)
        U_true   = potential_eval[idx]        # (Nx,)
        drag_true = float(drag_eval[idx])     # scalar

        # 6.1 计算真值密度轨道 rho_{theta*}(x,t)
        time_true, Pt_true, grid_true = simulate_density_trajectory(U_true, drag_true, grid)

        # 6.2 准备输入（标准化），供三个模型共享
        traj_norm = normalize_traj_with_train_stats(traj_raw, stats)  # (T, Nx)
        inputs = torch.tensor(traj_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, Nx)

        # ----- NIO -----
        with torch.no_grad():
            preds_nio = model_nio(inputs, grid_norm_tensor)  # (1, Nx, 2)
        preds_nio_np = preds_nio.squeeze(0).detach().cpu().numpy()  # (Nx, 2)
        pot_nio, drag_vec_nio = denormalize_outputs(preds_nio_np, stats)
        drag_nio_scalar = float(drag_vec_nio.mean())

        # 保存预测 potential / drag（和你原 eval 脚本相同格式）
        arr_nio = np.stack([pot_nio, drag_vec_nio], axis=1)
        np.save(os.path.join(BASE_RESULT_DIR, "nio", f"pred_sample_{idx}.npy"), arr_nio)

        time_nio, Pt_nio, grid_nio = simulate_density_trajectory(pot_nio, drag_nio_scalar, grid)
        err_nio = time_averaged_L2_error(time_true, Pt_true, time_nio, Pt_nio, grid_true)
        err_L2_results["nio"].append(err_nio)
        print(f"NIO   time-averaged L2 error: {err_nio:.4e}")

        # ----- FNO -----
        with torch.no_grad():
            preds_fno = model_fno(inputs, grid_norm_tensor)  # (1, Nx, 2)
        preds_fno_np = preds_fno.squeeze(0).detach().cpu().numpy()  # (Nx, 2)
        pot_fno, drag_vec_fno = denormalize_outputs(preds_fno_np, stats)
        drag_fno_scalar = float(drag_vec_fno.mean())

        arr_fno = np.stack([pot_fno, drag_vec_fno], axis=1)
        np.save(os.path.join(BASE_RESULT_DIR, "fno", f"pred_sample_{idx}.npy"), arr_fno)

        time_fno, Pt_fno, grid_fno = simulate_density_trajectory(pot_fno, drag_fno_scalar, grid)
        err_fno = time_averaged_L2_error(time_true, Pt_true, time_fno, Pt_fno, grid_true)
        err_L2_results["fno"].append(err_fno)
        print(f"FNO   time-averaged L2 error: {err_fno:.4e}")

        # ----- UNet -----
        with torch.no_grad():
            preds_unet = model_unet(inputs)  # (1, Nx, 2)  —— 与原 eval_unet_1d.py 一致
        preds_unet_np = preds_unet.squeeze(0).detach().cpu().numpy()  # (Nx, 2)
        pot_unet, drag_vec_unet = denormalize_outputs(preds_unet_np, stats)
        drag_unet_scalar = float(drag_vec_unet.mean())

        arr_unet = np.stack([pot_unet, drag_vec_unet], axis=1)
        np.save(os.path.join(BASE_RESULT_DIR, "unet", f"pred_sample_{idx}.npy"), arr_unet)

        time_unet, Pt_unet, grid_unet = simulate_density_trajectory(pot_unet, drag_unet_scalar, grid)
        err_unet = time_averaged_L2_error(time_true, Pt_true, time_unet, Pt_unet, grid_true)
        err_L2_results["unet"].append(err_unet)
        print(f"UNet  time-averaged L2 error: {err_unet:.4e}")

    # 7) 汇总结果并保存
    for model_name in ["nio", "fno", "unet"]:
        errs = np.array(err_L2_results[model_name], dtype=np.float64)
        mean_err = float(errs.mean())
        std_err  = float(errs.std())
        print(f"\n[{model_name.upper()}] over {num_samples} samples:")
        print(f"  mean Err_L2 = {mean_err:.4e}")
        print(f"  std  Err_L2 = {std_err:.4e}")

        # 保存到 npy
        np.save(os.path.join(BASE_RESULT_DIR, f"ErrL2_{model_name}_Nsamples_{num_samples}.npy"), errs)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100,
                        help="在 test/eval 集上评估的样本数量（从 index 0 开始）")
    args = parser.parse_args()
    main(args)
