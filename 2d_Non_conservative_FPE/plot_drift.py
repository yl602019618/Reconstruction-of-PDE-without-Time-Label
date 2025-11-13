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

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_blindno   = np.asarray(data['Fx_pred'])
Fy_blindno  = np.asarray(data['Fy_pred'])
Fx_gt        = np.asarray(data['Fx_true'])
Fy_gt     = np.asarray(data['Fy_true'])

data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/nio/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_nio   = np.asarray(data['Fx_pred'])
Fy_nio  = np.asarray(data['Fy_pred'])


data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/fno/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_fno   = np.asarray(data['Fx_pred'])
Fy_fno  = np.asarray(data['Fy_pred'])




# ----------------------------
# 工具：在一组数组上做统一 min-max 归一化 (0-1)
# ----------------------------
def normalize_group(arr_list, eps=0):
    vmin = min([a.min() for a in arr_list])
    vmax = max([a.max() for a in arr_list])
    scale = max(vmax - vmin, eps)
    return [(a - vmin) / scale for a in arr_list], (vmin, vmax)

# U 与 drag 分别归一化（保证可比性）
Fx_list_raw    = [Fx_gt, Fx_blindno, Fx_nio, Fx_fno]
Fy_list_raw    = [Fy_gt, Fy_blindno, Fy_nio, Fy_fno]

Fx_list_norm,    Fx_range    = normalize_group(Fx_list_raw)
Fy_list_norm,    Fy_range    = normalize_group(Fy_list_raw)

# 归一化坐标轴：假设数据为 (Ny, Nx)
Ny, Nx = Fx_gt.shape
extent = (0.0, 1.0, 0.0, 1.0)
xticks = [0.0, 0.5, 1.0]
yticks = [0.0, 0.5, 1.0]

# ----------------------------
# 绘制并保存（每张一页 PDF）
# ----------------------------
out_dir = "."  # 想存到别处可改成目标文件夹路径
Fx_names = ["Fx_true.pdf", "Fx_blindno.pdf", "Fx_nio.pdf", "Fx_fno.pdf"]
Fy_names = ["Fy_true.pdf", "Fy_blindno.pdf", "Fy_nio.pdf", "Fy_fno.pdf"]

# 对应到已归一化的数据
Fx_norm_list = Fx_list_norm  # [GT, blindno, nio, fno]
Fy_norm_list = Fy_list_norm

def save_field_pdf(Z, fname, label="Fx"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        Z, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
        extent=extent, interpolation=None, aspect="equal"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Normalized {label}")
    fig.savefig(f"{out_dir}/{fname}")
    plt.close(fig)

# 保存 Fx
for Z, fname in zip(Fx_norm_list, Fx_names):
    save_field_pdf(Z, fname, label="Fx")

# 保存 Fy
for Z, fname in zip(Fy_norm_list, Fy_names):
    save_field_pdf(Z, fname, label="Fy")

print("已保存 8 张 PDF：",
      "Fx_true.pdf, Fx_blindno.pdf, Fx_nio.pdf, Fx_fno.pdf, "
      "Fy_true.pdf, Fy_blindno.pdf, Fy_nio.pdf, Fy_fno.pdf")