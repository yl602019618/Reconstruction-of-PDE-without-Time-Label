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
index = 41

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/unet/sample_00{index}_predictions.npy', allow_pickle=True).item()
U_blindno   = np.asarray(data['drift_pred'])
drag_blindno= np.asarray(data['diffusion_pred'])
U_gt        = np.asarray(data['drift_true'])
drag_gt     = np.asarray(data['diffusion_true'])

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/nio/sample_00{index}_predictions.npy', allow_pickle=True).item()
U_nio       = np.asarray(data['drift_pred'])
drag_nio    = np.asarray(data['diffusion_pred'])

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/fno/sample_00{index}_predictions.npy', allow_pickle=True).item()
U_fno       = np.asarray(data['drift_pred'])
drag_fno    = np.asarray(data['diffusion_pred'])

# ----------------------------
# 工具：在一组数组上做统一 min-max 归一化 (0-1)
# ----------------------------
def normalize_group(arr_list, eps=0):
    vmin = min([a.min() for a in arr_list])
    vmax = max([a.max() for a in arr_list])
    scale = max(vmax - vmin, eps)
    return [(a - vmin) / scale for a in arr_list], (vmin, vmax)

# U 与 drag 分别归一化（保证可比性）
U_list_raw    = [U_gt, U_blindno, U_nio, U_fno]
drag_list_raw = [drag_gt, drag_blindno, drag_nio, drag_fno]

U_list_norm,    U_range    = normalize_group(U_list_raw)
drag_list_norm, drag_range = normalize_group(drag_list_raw)

# 归一化坐标轴：假设数据为 (Ny, Nx)
Ny, Nx = U_gt.shape
extent = (0.0, 1.0, 0.0, 1.0)
xticks = [0.0, 0.5, 1.0]
yticks = [0.0, 0.5, 1.0]

# ----------------------------
# 绘制并保存（每张一页 PDF）
# ----------------------------
items = [
    ("U_true.pdf",       U_list_norm[0]),
    ("U_blindno.pdf",    U_list_norm[1]),
    ("U_nio.pdf",        U_list_norm[2]),
    ("U_fno_nio.pdf",    U_list_norm[3]),
    ("drag_true.pdf",    drag_list_norm[0]),
    ("drag_blindno.pdf", drag_list_norm[1]),
    ("drag_nio.pdf",     drag_list_norm[2]),
    ("drag_fno_nio.pdf", drag_list_norm[3]),
]

for fname, Z in items:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=extent, interpolation=None, aspect="equal")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized value")

    # 单张图保存为 PDF
    fig.savefig(fname)
    plt.close(fig)

print("已保存 8 张 PDF：",
      "U_true.pdf, U_blindno.pdf, U_nio.pdf, U_fno_nio.pdf, "
      "drag_true.pdf, drag_blindno.pdf, drag_nio.pdf, drag_fno_nio.pdf")