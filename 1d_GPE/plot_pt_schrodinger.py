import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# 路径
# ----------------------------
pt_ref_path     = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig/traj.npy"


# ----------------------------
# 加载数据 (T, Nx) = (400, 80)
# ----------------------------
Pt_ref     = np.load(pt_ref_path,allow_pickle = True).item()['psi']        # (T, Nx)

# ----------------------------
# 误差（逐点相对误差）
# ----------------------------



# ----------------------------
# 统一风格（与前述图一致）
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

# 颜色映射（密度/误差统一使用同一 colormap）
cmap_name = "viridis"

# 工具函数：画 2D 图并保存（坐标归一化到 [0,1]）
def plot_map(Z, fname, vmin=None, vmax=None, cmap=cmap_name):
    T, Nx = Z.shape
    extent = [0, 1, 0, 1]  # x in [0,1], t in [0,1]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel(r"Normalized $X$")
    ax.set_ylabel(r"Normalized $t$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved: {fname}")

# 1) 参考密度图（使用自身数据范围做 vmin/vmax）
plot_map(
    Pt_ref,
    "Pt_ref_schrodinger.pdf",
    vmin=float(Pt_ref.min()),
    vmax=float(Pt_ref.max()),
)

