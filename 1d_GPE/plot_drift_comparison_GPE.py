import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ========= 读入数据（按你的路径） =========
U_gt = np.load('/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/unet/sample_pred_V.npy',allow_pickle = True).item()["true_V"]

U_BlinDNO = np.load('/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/unet/sample_pred_V.npy',allow_pickle = True).item()["pred_V"]




U_nio = np.load('/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/nio/sample_pred_V.npy',allow_pickle = True).item()["pred_V"]

U_fno = np.load('/home/ubuntu/unlabel_PDE_official/1dGPE/result_fig_GPE/fno/sample_pred_V.npy',allow_pickle = True).item()["pred_V"]


# ========= 统一风格（尽量贴近示例图） =========
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

# ========= 横坐标：0-1 归一化 =========
x = np.linspace(0, 1, len(U_gt))

# ========= U：按最大=0、最小=-1 归一化 =========
Umax, Umin = U_gt.max(), U_gt.min()
def norm_U(U):
    return (U - Umax) / (Umax - Umin)

U_gt_n   = norm_U(U_gt)
U_bln_n  = norm_U(U_BlinDNO)
U_nio_n  = norm_U(U_nio)
U_fno_n  = norm_U(U_fno)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
ax.plot(x, U_bln_n,  color=c_blindno, label="BlinDNO")
ax.plot(x, U_nio_n,  color=c_nio,     label="NIO")
ax.plot(x, U_fno_n,  color=c_fno,     label="FNO-NIO")
ax.plot(x, U_gt_n,   color=c_ref, ls="--", label="GT")

ax.set_xlim(0, 1)
ax.set_ylim(-1.1, 0.3)
ax.set_xlabel(r"Normalized $X$")
ax.set_ylabel(r"Normalized $V$")
ax.legend(loc="upper right", ncols=2,columnspacing=0.5)
plt.savefig("V_compare_GPE.pdf", dpi=300)
plt.close(fig)
