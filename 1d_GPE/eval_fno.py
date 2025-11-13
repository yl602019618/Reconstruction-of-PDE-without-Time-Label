#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
# ====== 路径配置 ======
TRAIN_FILE = "/home/ubuntu/unlabel_PDE_official/1dGPE/training_data_Schrodinger.npy"
TEST_FILE  = "/home/ubuntu/unlabel_PDE_official/1dGPE/test_data_Schrodinger.npy"
CKPT_PATH  = "/home/ubuntu/unlabel_PDE_official/1dGPE/results_schrodinger_fno/model_checkpoint_best_0.048281.pt"

OUT_DIR    = "result_fig/fno"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== 导入模型定义 ======
# 确保 NIOModules 在可导入路径中
from NIOModules import PermInvUNet_attn1D_bag  # 与训练时一致
from NIOModules import NIOFP_schrodinger
from NIOModules import NIOFP_schrodinger, NIOFP_FNO
def load_data_np(path):
    """读取 .npy(dict) 并返回字典"""
    data = np.load(path, allow_pickle=True).item()
    # 期望包含键：y, V, g, kappa
    required = ["y", "V", "g", "kappa"]
    for k in required:
        if k not in data:
            raise KeyError(f"{path} 缺少键: {k}")
    return data

def compute_train_scalers(train_dict):
    """
    训练集缩放参数，与训练脚本完全一致：
      y_max = y.max() / 3
      V_max = V.max() / 3
      g_max = g.max()
      kappa_max = kappa.max()
    返回字典
    """
    y_max = train_dict["y"].max() / 3.0
    V_max = train_dict["V"].max() / 3.0
    g_max = train_dict["g"].max()
    kappa_max = train_dict["kappa"].max()

    

    scalers = {
        "y_max": y_max,
        "V_max": V_max,
        "g_max": g_max,
        "kappa_max": kappa_max,
    }
    print("=== 训练集缩放因子 ===")
    for k, v in scalers.items():
        print(f"{k}: {v}")
    return scalers

def normalize_with_train_scalers(d, scalers):
    """
    用训练集的缩放因子对任意数据字典 d 进行归一化（与训练一致：只除以最大值，不减均值）
    """
    y = d["y"] / scalers["y_max"]
    V = d["V"] / scalers["V_max"]
    g = d["g"] / scalers["g_max"]
    kappa = d["kappa"] / scalers["kappa_max"]
    return {"y": y, "V": V, "g": g, "kappa": kappa}

def build_model(device):
    """
    按训练时的参数构建模型。
    训练代码中：PermInvUNet_attn1D_bag(in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=128, device=device)
    """
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

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容 DataParallel 等情况
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 加载训练集并计算缩放因子
    train_dict_raw = load_data_np(TRAIN_FILE)
    scalers = compute_train_scalers(train_dict_raw)

    # 2) 加载测试集，并用训练集缩放参数进行归一化
    test_dict_raw = load_data_np(TEST_FILE)
    test_norm = normalize_with_train_scalers(test_dict_raw, scalers)

    # 3) 取第一个样本（索引 0）
    idx = args.sample_idx  # 可改成任意 test 样本索引
    # test_norm["y"] 形状：(num_samples, num_time_samples, 128)
    # test_norm["V"] 形状：(num_samples, 128)
    traj = test_norm["y"][idx]        # (T, 128)  已归一化
    true_V_norm = test_norm["V"][idx] # (128,)    已归一化

    # 4) 构建并加载模型
    model = build_model(device)
    model = load_checkpoint(model, CKPT_PATH, device)
    grid = torch.linspace(0, 1, 128).unsqueeze(-1).to(device)
    # 5) 前向预测（与训练时的张量布局保持一致）
    # 训练时直接把 (batch, num_time_samples, 128) 喂入 model，所以这里按相同方式组织
    with torch.no_grad():
        # 增加 batch 维
        inp = torch.tensor(traj, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 128)
        pred = model(inp, grid)  # 期望输出形状：(1, 128, C)
        # 保险起见，把输出 squeeze 到 (128, C)
        pred = pred.squeeze(0)

        # 由于训练时只用 V 通道做监督，这里默认取第 0 个通道为 V 的预测
        # 如果你的模型把 V 放在其他通道，请相应修改索引
        if pred.dim() == 2:
            pred_V_norm = pred[:, 0].detach().cpu().numpy()  # (128,)
        elif pred.dim() == 1:
            # 如果模型实际上只输出一个通道
            pred_V_norm = pred.detach().cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    # 6) 反归一化回到原始物理尺度（与训练一致：乘以训练集的 V_max）
    V_max = scalers["V_max"]
    pred_V = pred_V_norm * V_max
    true_V = true_V_norm * V_max

    # 7) 画图并保存
    x = np.linspace(0.0, 1.0, true_V.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(x, true_V, label="True V", linewidth=2)
    plt.plot(x, pred_V, label="Predicted V", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Sample #0: V(x) Prediction vs Ground Truth")
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(OUT_DIR, "sample0_V_pred_vs_true.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存图像: {fig_path}")

    # 8) 保存预测结果到 .npy（包含网格、真值和预测，均为反归一化后的值）
    save_dict = {
        "x": x,                 # (128,)
        "pred_V": pred_V,       # (128,)
        "true_V": true_V,       # (128,)
        "V_max_used": V_max,    # 记录反归一化使用的系数
        "note": "Values are de-normalized using training set V_max (V_max = V_train.max()/3)."
    }
    npy_path = os.path.join(OUT_DIR, "sample_pred_V.npy")
    np.save(npy_path, save_dict)
    print(f"保存预测数据: {npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()
    main(args)
