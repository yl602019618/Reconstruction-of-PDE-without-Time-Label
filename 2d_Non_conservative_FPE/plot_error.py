import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Matplotlib 统一风格
# ----------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.fancybox": False,
    "legend.borderpad": 0.1,
    "legend.borderaxespad": 0.2,
    "lines.linewidth": 2,
})

# ----------------------------
# 读取数据
# ----------------------------
index = 2

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet/sample_000{index}_predictions.npy', allow_pickle=True).item()
Fx_blindno = np.asarray(data['Fx_pred'])
Fy_blindno = np.asarray(data['Fy_pred'])
Fx_gt      = np.asarray(data['Fx_true'])
Fy_gt      = np.asarray(data['Fy_true'])

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/nio/sample_000{index}_predictions.npy', allow_pickle=True).item()
Fx_nio = np.asarray(data['Fx_pred'])
Fy_nio = np.asarray(data['Fy_pred'])

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/fno/sample_000{index}_predictions.npy', allow_pickle=True).item()
Fx_fno = np.asarray(data['Fx_pred'])
Fy_fno = np.asarray(data['Fy_pred'])

# ----------------------------
# 工具：在一组数组上做统一 min-max 归一化 (0-1)
# ----------------------------
def normalize_group(arr_list, eps=0):
    vmin = min([a.min() for a in arr_list])
    vmax = max([a.max() for a in arr_list])
    scale = max(vmax - vmin, eps)
    return [(a - vmin) / scale for a in arr_list], (vmin, vmax)

# ----------------------------
# 原始场（可选：若仍需保存原 8 张图，保留这段；若只要误差图，可删）
# ----------------------------
Fx_list_raw = [Fx_gt, Fx_blindno, Fx_nio, Fx_fno]
Fy_list_raw = [Fy_gt, Fy_blindno, Fy_nio, Fy_fno]

Fx_list_norm, Fx_range = normalize_group(Fx_list_raw)
Fy_list_norm, Fy_range = normalize_group(Fy_list_raw)

# 归一化坐标轴：假设数据为 (Ny, Nx)
Ny, Nx = Fx_gt.shape
extent = (0.0, 1.0, 0.0, 1.0)
xticks = [0.0, 0.5, 1.0]
yticks = [0.0, 0.5, 1.0]

# ----------------------------
# 绘图保存工具
# ----------------------------
out_dir = "./error"  # 想存到别处可改成目标文件夹路径

def save_field_pdf(Z, fname, label="Fx"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        Z, origin="lower", cmap="viridis", vmin=0.0, vmax=0.1,
        extent=extent, interpolation=None, aspect="equal"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Error")
    fig.savefig(f"{out_dir}/{fname}")
    plt.close(fig)


# ============================
# 新增：误差图（|Z_pred - Z_true|）
# ============================

# 计算三个模型相对 GT 的绝对误差
Fx_err_blindno = np.abs(Fx_blindno - Fx_gt)
Fx_err_nio     = np.abs(Fx_nio     - Fx_gt)
Fx_err_fno     = np.abs(Fx_fno     - Fx_gt)

Fy_err_blindno = np.abs(Fy_blindno - Fy_gt)
Fy_err_nio     = np.abs(Fy_nio     - Fy_gt)
Fy_err_fno     = np.abs(Fy_fno     - Fy_gt)

# 分别对 Fx-误差组、Fy-误差组做统一归一化（确保三种模型可比）
Fx_err_list_norm, Fx_err_range = normalize_group(
    [Fx_err_blindno, Fx_err_nio, Fx_err_fno], eps=1e-12
)
Fy_err_list_norm, Fy_err_range = normalize_group(
    [Fy_err_blindno, Fy_err_nio, Fy_err_fno], eps=1e-12
)

# 文件名
Fx_err_names = ["Fx_err_blindno.pdf", "Fx_err_nio.pdf", "Fx_err_fno.pdf"]
Fy_err_names = ["Fy_err_blindno.pdf", "Fy_err_nio.pdf", "Fy_err_fno.pdf"]

# 保存误差图（颜色条标注成 |Fx_pred - Fx_true| / |Fy_pred - Fy_true|）
for Z, fname in zip(Fx_err_list_norm, Fx_err_names):
    save_field_pdf(Z, fname, label=r"|Fx_pred - Fx_true|")

for Z, fname in zip(Fy_err_list_norm, Fy_err_names):
    save_field_pdf(Z, fname, label=r"|Fy_pred - Fy_true|")

print("已保存误差图 6 张 PDF：",
      "Fx_err_blindno.pdf, Fx_err_nio.pdf, Fx_err_fno.pdf, "
      "Fy_err_blindno.pdf, Fy_err_nio.pdf, Fy_err_fno.pdf")
