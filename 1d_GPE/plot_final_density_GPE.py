# ========= 第三张图：最终时刻的 density 对比 =========
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams.update({
    "font.family": "DejaVu Sans",       # 通用无衬线，兼容性好；数学用 STIX，外观接近论文风格
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

# 颜色与线型（参考示例图：蓝/青/绿/灰 + 红色虚线参考）
c_blindno = "#0B346E"   # 深蓝
c_nio     = "#00A7A7"   # 青色
c_fno     = "#2F7D32"   # 绿色
c_other   = "#9E9E9E"   # 灰
c_ref     = "#D32F2F"   # 红（参考/GT）
# ----------------------------
# 路径
# ----------------------------
pt_ref_path = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/traj.npy"
pt_BlindNO_path = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/unet/traj.npy"
pt_nio_path  = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/nio/traj.npy"
pt_fno_path  = "/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/fno/traj.npy"

# ----------------------------
# 加载数据 (T, Nx) = (400, 80)
# ----------------------------
Pt_ref    = np.load(pt_ref_path,allow_pickle = True).item()['psi']       # (T, Nx)
Pt_BlindNO = np.load(pt_BlindNO_path,allow_pickle = True) .item()['psi']  # (T, Nx)
Pt_nio    = np.load(pt_nio_path,allow_pickle = True) .item()['psi']       # (T, Nx)
Pt_fno    = np.load(pt_fno_path,allow_pickle = True)  .item()['psi']      # (T, Nx)

# 取最后一个时刻 t = -1
T = Pt_ref.shape[0]
ref_last   = Pt_ref[-1]
blind_last = Pt_BlindNO[-1]
nio_last   = Pt_nio[-1]
fno_last   = Pt_fno[-1]

# 横坐标：0-1 归一化
Nx = ref_last.shape[0]
x = np.linspace(0, 1, Nx)

# y 轴范围：根据四条曲线自动选择，并留出 8% 边距
y_all = np.concatenate([ref_last, blind_last, nio_last, fno_last])
ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
pad = 0.2 * max(ymax - ymin, 1e-12)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

ax.plot(x, blind_last, color=c_blindno, label="BlinDNO")
ax.plot(x, nio_last,   color=c_nio,     label="NIO")
ax.plot(x, fno_last,   color=c_fno,     label="FNO-NIO")
ax.plot(x, ref_last,   color=c_ref, ls="--", label="GT")

ax.set_xlim(0, 1)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel(r"Normalized $X$")
ax.set_ylabel(r"$\rho(T)$")
ax.legend(loc="upper right", ncols=2, columnspacing=0.5)

plt.savefig("density_compare_GPE.pdf", dpi=300)
plt.close(fig)
print("Saved: density_compare_GPE.pdf")
