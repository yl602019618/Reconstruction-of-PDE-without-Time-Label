# eval_unet_1d.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from NIOModules import NIOFP, NIOFP_FNO, PermInvUNet_attn1D, PermInvUNet_attn1D_bag

# -----------------------------
# 路径配置（按需修改）
# -----------------------------
train_file = "/home/ubuntu/unlabel_PDE_official/1d/dataset_1D_drift_diffusion.npz"
eval_file  = "/home/ubuntu/unlabel_PDE_official/1d/dataset_1D_drift_diffusion_evaluation.npz"
ckpt_path  = "/home/ubuntu/unlabel_PDE_official/1d/results_fno/model_checkpoint_best_0.342750.pt"

save_dir = "result_fig/fno"
os.makedirs(save_dir, exist_ok=True)

# 选择要预测的样本索引


# 与训练脚本一致的缩放系数
TRAJ_SCALE = 1e5
POTENTIAL_SCALE = 1e20
DRAG_SCALE = 1e5

# -----------------------------
# 1) 从训练集计算并保存标准化统计量（与训练严格一致）
# -----------------------------
def compute_train_stats(train_npz_path):
    data = np.load(train_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32) * TRAJ_SCALE   # (M, 100, Nx)
    potential    = np.array(data["potential"],    dtype=np.float32) * POTENTIAL_SCALE  # (M, Nx)
    drag         = np.array(data["drag"],         dtype=np.float32) * DRAG_SCALE       # (M,)

    drag = drag[:, np.newaxis]  # (M,1)

    # 与训练完全一致的维度上做 mean/std
    traj_mean = trajectories.mean(axis=(0, 1), keepdims=True)
    traj_std  = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

    pot_mean  = potential.mean(axis=(0), keepdims=True)   # (1, Nx)
    pot_std   = potential.std(axis=(0),  keepdims=True) + 1e-8

    drag_mean = drag.mean(axis=(0), keepdims=True)        # (1,1)
    drag_std  = drag.std(axis=(0),  keepdims=True) + 1e-8

    stats = {
        "traj_mean": traj_mean, "traj_std": traj_std,
        "pot_mean": pot_mean,   "pot_std": pot_std,
        "drag_mean": drag_mean, "drag_std": drag_std
    }
    return stats

# -----------------------------
# 2) 加载评测集（只需用于取一条样本 + 可视化用的真值）
# -----------------------------
def load_eval_sample(eval_npz_path, idx):
    data = np.load(eval_npz_path, allow_pickle=True)
    trajectories = np.array(data["trajectories"], dtype=np.float32)  # 原始物理量级
    potential    = np.array(data["potential"],    dtype=np.float32)
    drag         = np.array(data["drag"],         dtype=np.float32)

    # 取指定样本（保持原始物理量级的真值，用于最终对比）
    traj_raw = trajectories[idx]          # (100, Nx)
    pot_raw  = potential[idx]             # (Nx,)
    drag_raw = drag[idx]                  # scalar

    return traj_raw, pot_raw, drag_raw

# -----------------------------
# 3) 使用训练统计量对评测输入做标准化，喂给模型
# -----------------------------
def normalize_with_train_stats(traj_raw, stats):
    # 先按训练时同样的缩放
    traj_scaled = traj_raw * TRAJ_SCALE  # (100, Nx)

    # 再用训练集统计量做标准化
    traj_norm = (traj_scaled - stats["traj_mean"].squeeze()) / stats["traj_std"].squeeze()
    return traj_norm.astype(np.float32)

# -----------------------------
# 4) 反标准化模型输出，恢复到原始物理量级（用于可视化/保存）
#    模型输出的形状：(Nx, 2) —— [:,0] potential, [:,1] drag(每点一值，但训练时drag是常数复制)
# -----------------------------
def denormalize_outputs(pred_norm, stats):
    # pred_norm: (Nx, 2)
    # 先反标准化（注意训练时 potential/drag 是分别以各自统计量标准化的）
    pot = pred_norm[:, 0] * stats["pot_std"].squeeze()  + stats["pot_mean"].squeeze()   # (Nx,)
    drg = pred_norm[:, 1] * stats["drag_std"].squeeze() + stats["drag_mean"].squeeze()  # (Nx,)

    # 再把训练前的缩放还原回原始物理量级
    pot = pot / POTENTIAL_SCALE
    drg = drg / DRAG_SCALE
    return pot, drg

# -----------------------------
# 5) 载入模型
# -----------------------------
def load_model(device):
    # 与训练时参数保持一致
    fno_layers = 3
    width = 30
    modes = 15
    output_dim = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NIOFP_FNO(
        fno_layers,
        width,
        modes,
        output_dim,
        device
    ).to(device)
    return model

def main(args):

    sample_idx = args.sample_idx  # 可改成任意 test 样本索引
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 计算训练集统计量
    stats = compute_train_stats(train_file)

    # 加载评测样本（原始物理量级）
    traj_raw, pot_raw, drag_raw = load_eval_sample(eval_file, sample_idx)  # (100,Nx), (Nx,), scalar
    Nx = pot_raw.shape[0]
    

    # 标准化输入（使用训练统计量）
    traj_norm = normalize_with_train_stats(traj_raw, stats)  # (100, Nx)
    # 模型期望的张量形状与训练相同：batch维在前
    inputs = torch.tensor(traj_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1,100,Nx)

    # 加载模型
    model = load_model(device)
    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    grid = torch.linspace(0, 1, 80).unsqueeze(-1).to(device)
    with torch.no_grad():
        preds = model(inputs,grid)  # 预期 (1, Nx, 2)
    preds = preds.squeeze(0).detach().cpu().numpy()  # (Nx, 2)，列0=potential，列1=drag(逐点)

    # 反标准化到原始物理量级
    pred_potential, pred_drag_vec = denormalize_outputs(preds, stats)  # (Nx,), (Nx,)
    # drag 在训练时是常数沿 x 复制，这里画图时采用均值代表预测标量
    pred_drag_scalar = float(pred_drag_vec.mean())

    # -----------------------------
    # 可视化并保存
    # -----------------------------
    # 1) potential 对比
    grid = np.linspace(0.0, 1.0, Nx)
    plt.figure()
    plt.plot(grid, pot_raw, label="True Potential")
    plt.plot(grid, pred_potential, linestyle="--", label="Predicted Potential")
    plt.xlabel("Grid")
    plt.ylabel("Potential")
    plt.title(f"Potential: Sample {sample_idx}")
    plt.legend()
    plt.grid(True)
    pot_png = os.path.join(save_dir, f"potential_sample_{sample_idx}.png")
    plt.savefig(pot_png, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) drag 对比（真值是标量，沿 x 方向常数）
    plt.figure()
    plt.plot(grid, np.ones_like(grid) * drag_raw, label="True Drag")
    plt.plot(grid, np.ones_like(grid) * pred_drag_scalar, linestyle="--", label="Predicted Drag")
    plt.xlabel("Grid")
    plt.ylabel("Drag")
    plt.title(f"Drag: Sample {sample_idx}")
    plt.ylim(drag_raw - 2*abs(drag_raw) - 1, drag_raw + 2*abs(drag_raw) + 1)  # 给个相对宽松的范围
    plt.legend()
    plt.grid(True)
    drag_png = os.path.join(save_dir, f"drag_sample_{sample_idx}.png")
    plt.savefig(drag_png, dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 保存预测结果到 .npy
    # 保存的数组形状：(Nx, 2)，列0: potential，列1: drag（按点给出，但可用其均值作为标量）
    # -----------------------------
    pred_array = np.stack([pred_potential, pred_drag_vec], axis=1)  # (Nx, 2)
    npy_path = os.path.join(save_dir, f"pred_sample_{sample_idx}.npy")
    np.save(npy_path, pred_array)

    print(f"[Done] Figures saved to:\n  {pot_png}\n  {drag_png}")
    print(f"[Done] Prediction .npy saved to:\n  {npy_path}")
    print(f"Predicted drag (mean over x): {pred_drag_scalar:.6g}, True drag: {drag_raw:.6g}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=1000)
    args = parser.parse_args()
    main(args)