import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useMathText=True)  # 用 MathText 保证 1e-3 形式
formatter.set_powerlimits((-3, 3))  # 在 10^-3 到 10^3 范围内使用科学计数法

# ----------------------------
# 路径
# ----------------------------
pt_ref_path = "/home/ubuntu/unlabelPDE_official/2d_diffusion/result_fig/Pt_41.npy"

# ----------------------------
# 加载数据 (T, Nx*Ny) 或 (T, Ny, Nx)
# ----------------------------
Pt_ref = np.load(pt_ref_path)

# 取三个时刻
Pt_0   = Pt_ref[0]
Pt_mid = Pt_ref[Pt_ref.shape[0] // 2]
Pt_end = Pt_ref[-1]

def ensure_2d(arr):
    """将输入转成 (Ny, Nx) 的 2D 数组"""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        n = arr.size
        s = int(np.sqrt(n))
        if s * s != n:
            raise ValueError(f"无法将长度为 {n} 的数组重塑成方阵。")
        return arr.reshape(s, s)
    raise ValueError(f"不支持的维度：{arr.ndim}")

Z0   = ensure_2d(Pt_0)
Zmid = ensure_2d(Pt_mid)
Zend = ensure_2d(Pt_end)

Ny, Nx = Z0.shape
x = np.linspace(0.0, 1.0, Nx)
y = np.linspace(0.0, 1.0, Ny)
X, Y = np.meshgrid(x, y)

# ----------------------------
# 统一风格
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

# cmap 和颜色范围统一


datasets = [
    (Z0, "fp_t0.pdf"),
    (Zmid, "fp_tmid.pdf"),
    (Zend, "fp_tend.pdf")
]

# ----------------------------
# 逐张保存
# ----------------------------
for Z, filename in datasets:
    # 1) 自动布局，给标签和 colorbar 预留空间
    cmap = get_cmap("viridis")
    vmin =Z.min()
    vmax = Z0.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # 让 Z 轴在右侧，同时避免遮挡
    ax.view_init(elev=35, azim=35)

    facecolors = cmap(norm(Z))
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        shade=False
    )

    # 0–1 归一化坐标与刻度
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    # 2) 避免 x 与 y 的 0.0 重叠：隐藏其中一个（这里隐藏 y=0 的标签）
    ax.set_yticklabels(["", "0.5", "1.0"])

    # 3) 适当的文字与轴的距离，减少重叠
    ax.tick_params(axis='x', pad=6)
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='z', pad=4)

    # Z 轴科学计数法
    #ax.zaxis.set_major_formatter(formatter)

    # Z 轴标签：给足 labelpad，且不使用旋转的 3D 文本（更不易被裁）
    #ax.zaxis.set_rotate_label(False)
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$y$")
    # ax.set_zlabel(r"$p(x,y)$", labelpad=18)

    # 独立 colorbar（constrained_layout 会自动留白，不会裁切）
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z)
    # cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("Density")

    # 保存：不要用 bbox_inches='tight'，避免裁掉 zlabel
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

print("三张 PDF 图片已保存：fp_t0.pdf, fp_tmid.pdf, fp_tend.pdf")