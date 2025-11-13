import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os
index = 2
# 基本参数设置
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius  # 固定的 drag 值
temperature = 300
extent = [800 * nm, 800 * nm]
resolution = 10 * nm
boundary_condition = boundary.reflecting

# 定义 F 函数，默认参数作为占位
def F(x, y, L=100*nm, a=1, b=1, c=1, d=1):
    rad = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    Fphi = 1e-12 * rad / L * np.exp(-rad / L * b) * a
    Frad = 1e-12 * (1 - rad / L) * np.exp(-rad / L * d) * c
    Fx = -np.sin(phi) * Fphi + np.cos(phi) * Frad
    Fy =  np.cos(phi) * Fphi + np.sin(phi) * Frad
    return np.array([Fx, Fy])

def potential_from_data(grid, data):
    """
    grid:
      - 要么是长度为 ndim 的 tuple/list，每个元素是一维坐标轴 (严格单调)
      - 要么是形如 (ndim, n1, n2[, ...]) 的网格数组（比如 np.meshgrid 的输出堆叠）
    data:
      - 形状 (n1, n2[, ...]) 的标量场
    返回：potential(*args) 可在任意点插值
    """
    # 情况 1：已经是 (x_axis, y_axis, ...) 的一维轴
    if isinstance(grid, (list, tuple)) and all(np.ndim(g) == 1 for g in grid):
        axes = list(grid)
        arr = np.asarray(data)
        # 保证严格单调，必要时翻转 data
        for d, ax in enumerate(axes):
            if np.any(np.diff(ax) == 0):
                raise ValueError(f"Axis {d} has repeated points; axes must be strictly monotonic.")
            if ax[0] > ax[-1]:
                axes[d] = ax[::-1]
                arr = np.flip(arr, axis=d)
        f = RegularGridInterpolator(tuple(axes), arr, bounds_error=False, fill_value=None)

    else:
        # 情况 2：是堆叠网格 (ndim, n1, n2)
        G = np.asarray(grid)
        if G.ndim != (np.asarray(data).ndim + 1):
            raise ValueError(f"grid (ndim={G.ndim}) and data (ndim={np.asarray(data).ndim}) are incompatible.")

        # 这里以 2D 为例（你的数据就是 2D），推广到更高维也可以类推
        if G.shape[0] != 2:
            raise ValueError("Only 2D grids are supported in this helper. Got first dim size != 2.")

        X, Y = G[0], G[1]                 # 形状都是 (n1, n2)
        arr = np.asarray(data)
        n1, n2 = arr.shape

        # 从网格提取一维坐标轴（不假设 'ij' 或 'xy'，而是自适应）
        # 尝试把第一维的轴取为 X 的第一列；第二维的轴取为 Y 的第一行
        x_try = X[:, 0]
        y_try = Y[0, :]
        ok_x = (x_try.size == n1)
        ok_y = (y_try.size == n2)

        # 如果不匹配，试另一种取法（相当于网格用的是另一种 indexing）
        if not (ok_x and ok_y):
            x_try2 = X[0, :]
            y_try2 = Y[:, 0]
            if x_try2.size == n1 and y_try2.size == n2:
                x_try, y_try = x_try2, y_try2
            else:
                raise ValueError("Cannot infer 1D axes from the stacked grid; shapes don't match data.")

        # 确保严格单调，必要时翻转 data
        def ensure_monotonic(ax, axis):
            nonlocal arr
            if np.any(np.diff(ax) == 0):
                raise ValueError("Axis has repeated points; axes must be strictly monotonic.")
            if ax[0] > ax[-1]:
                ax = ax[::-1]
                arr = np.flip(arr, axis=axis)
            return ax

        x_axis = ensure_monotonic(x_try, axis=0)
        y_axis = ensure_monotonic(y_try, axis=1)

        f = RegularGridInterpolator((x_axis, y_axis), arr, bounds_error=False, fill_value=None)

    def potential(*args):
        # 支持传入 (x, y) 或形如 [[x1, y1], [x2, y2], ...] 的点
        pts = np.stack(args, axis=-1) if len(args) > 1 else args[0]
        return f(pts)

    return potential

L_rand = np.random.uniform(50 * nm, 150 * nm)
a_rand = np.random.uniform(0.5, 2)
b_rand = np.random.uniform(0.5, 2)
c_rand = np.random.uniform(0.5, 2)
d_rand = np.random.uniform(0.5, 2)

# 固定参数传入 force 参数（注意：通过 lambda 固定参数）
force_func = lambda x, y: F(x, y, L=L_rand, a=a_rand, b=b_rand, c=c_rand, d=d_rand)

# 创建模拟对象，使用固定的 drag
sim = fokker_planck(temperature=temperature,
                    drag=drag,
                    extent=extent,
                    resolution=resolution,
                    boundary=boundary_condition,
                    force=force_func)
grid = sim.grid   
fig_path = '/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_np = data['Fx_pred']
Fy_np = data['Fy_pred']
# 先各自构造标量插值函数：Fx(x,y) 与 Fy(x,y)
Fx_func = potential_from_data(grid, Fx_np)  # grid 可用 sim.grid（你的代码里已有）
Fy_func = potential_from_data(grid, Fy_np)

def F_from_data(x, y):
    """
    向量力场插值：返回 [Fx(x,y), Fy(x,y)]。
    兼容标量或数组输入，输出形状与输入广播后的形状一致：
      - 若 x,y 为标量 -> 返回形状 (2,)
      - 若 x,y 为同形数组 -> 返回形状 (2, *x.shape)
    """
    Fx_val = Fx_func(x, y)
    Fy_val = Fy_func(x, y)
    # 保持与原先 F(x,y) 返回 np.array([Fx, Fy]) 的风格一致
    return np.array([Fx_val, Fy_val])

sim = fokker_planck(
    temperature=temperature,
    drag=drag,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    force=F_from_data,   # 关键：这里用向量插值出来的力场
)

# 初始 PDF（这里采用与例子中相同的参数）
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

# 使用 1000 个时间步长进行传播，时间间隔取 10e-3（与原 F 模拟代码保持一致）
Nsteps = 500
dt = 10e-3
time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
F_field = F_from_data(*sim.grid)  # 形状 (2, Nx, Ny)
Fx_true, Fy_true = F_field[0], F_field[1]

# 预测力场（来自网络推理结果）
Fx_pred, Fy_pred = Fx_np, Fy_np  # 形状 (Nx, Ny)

# 统一 imshow 的坐标范围
x_min, x_max = grid[0].min(), grid[0].max()
y_min, y_max = grid[1].min(), grid[1].max()
extent_xy = (x_min, x_max, y_min, y_max)

# --- Fx / Fy 对比可视化 ---
# 对每个分量采用对称色标，便于直观比较
vmax_fx = float(np.max(np.abs([Fx_true, Fx_pred])))
vmin_fx = -vmax_fx
vmax_fy = float(np.max(np.abs([Fy_true, Fy_pred])))
vmin_fy = -vmax_fy

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im00 = axes[0, 0].imshow(Fx_true, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 0].set_title("Fx (True)")
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

im01 = axes[0, 1].imshow(Fx_pred, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 1].set_title("Fx (Pred)")
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

im10 = axes[1, 0].imshow(Fy_true, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 0].set_title("Fy (True)")
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(Fy_pred, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 1].set_title("Fy (Pred)")
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"F_comparison_{index}.png"), dpi=300)
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=extent_xy)
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"Pt_{index}.png"), dpi=300)
plt.close()

# 同时保存 P_t 全时序（你前面已经保存，这里保留）
np.save(os.path.join(fig_path, f'Pt_{index}.npy'), Pt)









fig_path = '/home/ubuntu/unlabelPDE_official/2d_force/result_fig/nio'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/nio/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_np = data['Fx_pred']
Fy_np = data['Fy_pred']
# 先各自构造标量插值函数：Fx(x,y) 与 Fy(x,y)
Fx_func = potential_from_data(grid, Fx_np)  # grid 可用 sim.grid（你的代码里已有）
Fy_func = potential_from_data(grid, Fy_np)

def F_from_data(x, y):
    """
    向量力场插值：返回 [Fx(x,y), Fy(x,y)]。
    兼容标量或数组输入，输出形状与输入广播后的形状一致：
      - 若 x,y 为标量 -> 返回形状 (2,)
      - 若 x,y 为同形数组 -> 返回形状 (2, *x.shape)
    """
    Fx_val = Fx_func(x, y)
    Fy_val = Fy_func(x, y)
    # 保持与原先 F(x,y) 返回 np.array([Fx, Fy]) 的风格一致
    return np.array([Fx_val, Fy_val])

sim = fokker_planck(
    temperature=temperature,
    drag=drag,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    force=F_from_data,   # 关键：这里用向量插值出来的力场
)

# 初始 PDF（这里采用与例子中相同的参数）
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

# 使用 1000 个时间步长进行传播，时间间隔取 10e-3（与原 F 模拟代码保持一致）
Nsteps = 500
dt = 10e-3
time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
F_field = F_from_data(*sim.grid)  # 形状 (2, Nx, Ny)
Fx_true, Fy_true = F_field[0], F_field[1]

# 预测力场（来自网络推理结果）
Fx_pred, Fy_pred = Fx_np, Fy_np  # 形状 (Nx, Ny)

# 统一 imshow 的坐标范围
x_min, x_max = grid[0].min(), grid[0].max()
y_min, y_max = grid[1].min(), grid[1].max()
extent_xy = (x_min, x_max, y_min, y_max)

# --- Fx / Fy 对比可视化 ---
# 对每个分量采用对称色标，便于直观比较
vmax_fx = float(np.max(np.abs([Fx_true, Fx_pred])))
vmin_fx = -vmax_fx
vmax_fy = float(np.max(np.abs([Fy_true, Fy_pred])))
vmin_fy = -vmax_fy

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im00 = axes[0, 0].imshow(Fx_true, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 0].set_title("Fx (True)")
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

im01 = axes[0, 1].imshow(Fx_pred, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 1].set_title("Fx (Pred)")
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

im10 = axes[1, 0].imshow(Fy_true, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 0].set_title("Fy (True)")
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(Fy_pred, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 1].set_title("Fy (Pred)")
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"F_comparison_{index}.png"), dpi=300)
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=extent_xy)
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"Pt_{index}.png"), dpi=300)
plt.close()

# 同时保存 P_t 全时序（你前面已经保存，这里保留）
np.save(os.path.join(fig_path, f'Pt_{index}.npy'), Pt)







fig_path = '/home/ubuntu/unlabelPDE_official/2d_force/result_fig/fno'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/fno/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_np = data['Fx_pred']
Fy_np = data['Fy_pred']
# 先各自构造标量插值函数：Fx(x,y) 与 Fy(x,y)
Fx_func = potential_from_data(grid, Fx_np)  # grid 可用 sim.grid（你的代码里已有）
Fy_func = potential_from_data(grid, Fy_np)

def F_from_data(x, y):
    """
    向量力场插值：返回 [Fx(x,y), Fy(x,y)]。
    兼容标量或数组输入，输出形状与输入广播后的形状一致：
      - 若 x,y 为标量 -> 返回形状 (2,)
      - 若 x,y 为同形数组 -> 返回形状 (2, *x.shape)
    """
    Fx_val = Fx_func(x, y)
    Fy_val = Fy_func(x, y)
    # 保持与原先 F(x,y) 返回 np.array([Fx, Fy]) 的风格一致
    return np.array([Fx_val, Fy_val])

sim = fokker_planck(
    temperature=temperature,
    drag=drag,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    force=F_from_data,   # 关键：这里用向量插值出来的力场
)

# 初始 PDF（这里采用与例子中相同的参数）
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

# 使用 1000 个时间步长进行传播，时间间隔取 10e-3（与原 F 模拟代码保持一致）
Nsteps = 500
dt = 10e-3
time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
F_field = F_from_data(*sim.grid)  # 形状 (2, Nx, Ny)
Fx_true, Fy_true = F_field[0], F_field[1]

# 预测力场（来自网络推理结果）
Fx_pred, Fy_pred = Fx_np, Fy_np  # 形状 (Nx, Ny)

# 统一 imshow 的坐标范围
x_min, x_max = grid[0].min(), grid[0].max()
y_min, y_max = grid[1].min(), grid[1].max()
extent_xy = (x_min, x_max, y_min, y_max)

# --- Fx / Fy 对比可视化 ---
# 对每个分量采用对称色标，便于直观比较
vmax_fx = float(np.max(np.abs([Fx_true, Fx_pred])))
vmin_fx = -vmax_fx
vmax_fy = float(np.max(np.abs([Fy_true, Fy_pred])))
vmin_fy = -vmax_fy

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im00 = axes[0, 0].imshow(Fx_true, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 0].set_title("Fx (True)")
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

im01 = axes[0, 1].imshow(Fx_pred, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 1].set_title("Fx (Pred)")
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

im10 = axes[1, 0].imshow(Fy_true, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 0].set_title("Fy (True)")
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(Fy_pred, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 1].set_title("Fy (Pred)")
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"F_comparison_{index}.png"), dpi=300)
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=extent_xy)
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"Pt_{index}.png"), dpi=300)
plt.close()

# 同时保存 P_t 全时序（你前面已经保存，这里保留）
np.save(os.path.join(fig_path, f'Pt_{index}.npy'), Pt)




fig_path = '/home/ubuntu/unlabelPDE_official/2d_force/result_fig/'
data = np.load(f'/home/ubuntu/unlabelPDE_official/2d_force/result_fig/unet/sample_000{index}_predictions.npy',allow_pickle = True).item()
Fx_np = data['Fx_true']
Fy_np = data['Fy_true']
# 先各自构造标量插值函数：Fx(x,y) 与 Fy(x,y)
Fx_func = potential_from_data(grid, Fx_np)  # grid 可用 sim.grid（你的代码里已有）
Fy_func = potential_from_data(grid, Fy_np)

def F_from_data(x, y):
    """
    向量力场插值：返回 [Fx(x,y), Fy(x,y)]。
    兼容标量或数组输入，输出形状与输入广播后的形状一致：
      - 若 x,y 为标量 -> 返回形状 (2,)
      - 若 x,y 为同形数组 -> 返回形状 (2, *x.shape)
    """
    Fx_val = Fx_func(x, y)
    Fy_val = Fy_func(x, y)
    # 保持与原先 F(x,y) 返回 np.array([Fx, Fy]) 的风格一致
    return np.array([Fx_val, Fy_val])

sim = fokker_planck(
    temperature=temperature,
    drag=drag,
    extent=extent,
    resolution=resolution,
    boundary=boundary_condition,
    force=F_from_data,   # 关键：这里用向量插值出来的力场
)

# 初始 PDF（这里采用与例子中相同的参数）
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

# 使用 1000 个时间步长进行传播，时间间隔取 10e-3（与原 F 模拟代码保持一致）
Nsteps = 500
dt = 10e-3
time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
F_field = F_from_data(*sim.grid)  # 形状 (2, Nx, Ny)
Fx_true, Fy_true = F_field[0], F_field[1]

# 预测力场（来自网络推理结果）
Fx_pred, Fy_pred = Fx_np, Fy_np  # 形状 (Nx, Ny)

# 统一 imshow 的坐标范围
x_min, x_max = grid[0].min(), grid[0].max()
y_min, y_max = grid[1].min(), grid[1].max()
extent_xy = (x_min, x_max, y_min, y_max)

# --- Fx / Fy 对比可视化 ---
# 对每个分量采用对称色标，便于直观比较
vmax_fx = float(np.max(np.abs([Fx_true, Fx_pred])))
vmin_fx = -vmax_fx
vmax_fy = float(np.max(np.abs([Fy_true, Fy_pred])))
vmin_fy = -vmax_fy

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im00 = axes[0, 0].imshow(Fx_true, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 0].set_title("Fx (True)")
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

im01 = axes[0, 1].imshow(Fx_pred, origin='lower', extent=extent_xy, vmin=vmin_fx, vmax=vmax_fx)
axes[0, 1].set_title("Fx (Pred)")
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

im10 = axes[1, 0].imshow(Fy_true, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 0].set_title("Fy (True)")
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(Fy_pred, origin='lower', extent=extent_xy, vmin=vmin_fy, vmax=vmax_fy)
axes[1, 1].set_title("Fy (Pred)")
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"F_comparison_{index}.png"), dpi=300)
plt.close()

# --- P_t 最后一帧可视化 ---
Pt_last = Pt[-1]  # 最后一帧
plt.figure(figsize=(5, 4))
im2 = plt.imshow(Pt_last, origin='lower', extent=extent_xy)
plt.title("P_t Last Frame")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, f"Pt_{index}.png"), dpi=300)
plt.close()

# 同时保存 P_t 全时序（你前面已经保存，这里保留）
np.save(os.path.join(fig_path, f'Pt_{index}.npy'), Pt)