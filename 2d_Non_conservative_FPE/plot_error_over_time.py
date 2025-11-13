import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
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
index = 2
pt_ref_path = f"/home/ubuntu/unlabelPDE_official/2d_force/result_fig/Pt_{index}.npy"
pt_blindno_path = f"/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet/Pt_{index}.npy"
pt_nio_path  = f"/home/ubuntu/unlabelPDE_official/2d_force/result_fig/nio/Pt_{index}.npy"
pt_fno_path  = f"/home/ubuntu/unlabelPDE_official/2d_force/result_fig/fno/Pt_{index}.npy"

# ----------------------------
# 加载数据
# 期望形状 (T, Nx) = (400, 80)
# ----------------------------
Pt_ref  = np.load(pt_ref_path)*1e10   # (400, 80)
Pt_blindno = np.load(pt_blindno_path)*1e10  # (400, 80)
Pt_nio  = np.load(pt_nio_path)*1e10   # (400, 80)
Pt_fno  = np.load(pt_fno_path)*1e10   # (400, 80)

# ----------------------------
# 计算每个时间步的相对 L2 误差
# ----------------------------
eps = 1e-20
def relative_l2_over_time(P_model, P_ref, eps=eps):
    # (T, Nx)
    num = np.linalg.norm(P_model - P_ref, axis=(1,2), ord=2)
    den = np.linalg.norm(P_ref, axis=(1,2), ord=2) 
    return num / den  # (T,)

rel_blindno = relative_l2_over_time(Pt_blindno, Pt_ref, eps)
rel_nio     = relative_l2_over_time(Pt_nio,     Pt_ref, eps)
rel_fno     = relative_l2_over_time(Pt_fno,     Pt_ref, eps)

# ----------------------------
# 折线图：三个模型放在同一张图（风格与前图一致）
# ----------------------------
T = Pt_ref.shape[0]
t_idx = np.linspace(0, 1, T)  # 归一化时间

# 自动 y 轴范围：从 0 到略高于最大值
ymax = float(np.max([rel_blindno.max(), rel_nio.max(), rel_fno.max()]))
if ymax <= 0:
    ymax = 1.0
pad = max(0.15 * ymax, 1e-6)
ymin_auto, ymax_auto = 0.0, ymax + pad

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

ax.plot(t_idx, rel_blindno, color=c_blindno, label="BlinDNO")
ax.plot(t_idx, rel_nio,     color=c_nio,     label="NIO")
ax.plot(t_idx, rel_fno,     color=c_fno,     label="FNO-NIO")

ax.set_xlim(0, 1)
ax.set_ylim(ymin_auto, ymax_auto)
ax.set_xlabel(r"Normalized $t$")
ax.set_ylabel(r"Relative $L^2$ Error")
ax.legend(loc="upper right", ncols=2, columnspacing=0.5)

plt.savefig("relL2_over_time.pdf", dpi=300)
plt.close(fig)

print("Saved: relL2_over_time.pdf")